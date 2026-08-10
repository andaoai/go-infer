// Package detector 封装 YOLO ONNX 模型的加载、预处理、推理与后处理。
//
// 适配 ultralytics 导出的 ONNX 检测模型：
//   - 输入:  float32[N,3,H,W]，NCHW，像素归一化到 0~1（letterbox 填充为 114）
//   - 输出:  两种常见排布均支持：
//     [1, 4+nc, anchors]  —— 官方默认导出
//     [1, anchors, 4+nc]  —nms 后或转置后的排布
package detector

import (
	"fmt"
	"image"
	"sort"
	"sync"

	"github.com/yalue/onnxruntime_go"
)

// Detection 是单个检测结果，坐标映射回原图。
type Detection struct {
	ClassID    int     `json:"class_id"`
	ClassName  string  `json:"class_name"`
	Confidence float32 `json:"confidence"`
	X1         float32 `json:"x1"`
	Y1         float32 `json:"y1"`
	X2         float32 `json:"x2"`
	Y2         float32 `json:"y2"`
}

// Config 是检测器配置。
type Config struct {
	ModelPath  string
	InputW     int      // 模型输入宽
	InputH     int      // 模型输入高
	Classes    []string // 类别名（下标即 class id）
	ConfThresh float32  // 置信度阈值
	IoUThresh  float32  // NMS IoU 阈值
}

// Detector 持有 ONNX session 与输入输出张量。
type Detector struct {
	cfg       Config
	mu        sync.RWMutex // 保护运行时可变参数（如 conf 阈值）
	runMu     sync.Mutex   // 序列化推理：输入/输出张量缓冲为共享资源
	session   *onnxruntime_go.AdvancedSession
	input     *onnxruntime_go.Tensor[float32]
	output    *onnxruntime_go.Tensor[float32]
	inputBuf  []float32
	outputBuf []float32
	// 输出张量形状参数，初始化时解析
	outAnchors    int
	outAttrs      int  // 4 + nc
	outTransposed bool // true: [1, anchors, attrs]; false: [1, attrs, anchors]
}

// SetConfThresh 动态调整置信度阈值（线程安全）。
func (d *Detector) SetConfThresh(t float32) {
	d.mu.Lock()
	d.cfg.ConfThresh = t
	d.mu.Unlock()
}

// New 创建并初始化检测器。
//
// 从模型元数据读取真实输入/输出名与输出形状，兼容官方导出的
// [1, 4+nc, anchors] 与转置后的 [1, anchors, 4+nc] 两种排布。
func New(cfg Config) (*Detector, error) {
	if cfg.ConfThresh <= 0 {
		cfg.ConfThresh = 0.25
	}
	if cfg.IoUThresh <= 0 {
		cfg.IoUThresh = 0.45
	}
	if cfg.InputW <= 0 || cfg.InputH <= 0 {
		return nil, fmt.Errorf("input size must be positive, got %dx%d", cfg.InputW, cfg.InputH)
	}

	inputs, outputs, err := onnxruntime_go.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("read model IO info: %w", err)
	}
	if len(inputs) != 1 || len(outputs) != 1 {
		return nil, fmt.Errorf("expect 1 input / 1 output, got %d/%d", len(inputs), len(outputs))
	}
	inName, outInfo := inputs[0].Name, outputs[0]

	outAnchors, outAttrs, outTransposed, err := parseOutputShape(outInfo.Dimensions, len(cfg.Classes))
	if err != nil {
		return nil, err
	}

	inputShape := onnxruntime_go.NewShape(1, 3, int64(cfg.InputH), int64(cfg.InputW))
	inputBuf := make([]float32, 1*3*cfg.InputH*cfg.InputW)
	input, err := onnxruntime_go.NewTensor(inputShape, inputBuf)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outputShape := onnxruntime_go.NewShape(1)
	if outTransposed {
		outputShape = append(outputShape, int64(outAnchors), int64(outAttrs))
	} else {
		outputShape = append(outputShape, int64(outAttrs), int64(outAnchors))
	}
	outputBuf := make([]float32, outAnchors*outAttrs)
	output, err := onnxruntime_go.NewTensor(outputShape, outputBuf)
	if err != nil {
		input.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	options, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		input.Destroy()
		output.Destroy()
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer options.Destroy()

	session, err := onnxruntime_go.NewAdvancedSession(
		cfg.ModelPath,
		[]string{inName},
		[]string{outInfo.Name},
		[]onnxruntime_go.Value{input},
		[]onnxruntime_go.Value{output},
		options,
	)
	if err != nil {
		input.Destroy()
		output.Destroy()
		return nil, fmt.Errorf("create session: %w", err)
	}

	return &Detector{
		cfg:           cfg,
		session:       session,
		input:         input,
		output:        output,
		inputBuf:      inputBuf,
		outputBuf:     outputBuf,
		outAnchors:    outAnchors,
		outAttrs:      outAttrs,
		outTransposed: outTransposed,
	}, nil
}

// parseOutputShape 将模型输出形状解析为 anchors / attrs / 是否转置。
func parseOutputShape(dims onnxruntime_go.Shape, nc int) (anchors, attrs int, transposed bool, err error) {
	if len(dims) != 3 {
		return 0, 0, false, fmt.Errorf("unsupported output rank %v, expect [1,...,...]", dims)
	}
	if dims[0] != 1 {
		return 0, 0, false, fmt.Errorf("batch dim must be 1, got %v", dims)
	}
	expectedAttrs := 4 + nc
	dim1, dim2 := int(dims[1]), int(dims[2])
	// -1 表示动态维度（少见，但兜底）
	if dim1 < 0 || dim2 < 0 {
		return 0, 0, false, fmt.Errorf("dynamic output dim %v not supported; re-export with fixed shape", dims)
	}
	switch {
	case dim1 == expectedAttrs:
		return dim2, expectedAttrs, false, nil
	case dim2 == expectedAttrs:
		return dim1, expectedAttrs, true, nil
	default:
		return 0, 0, false, fmt.Errorf("output shape %v does not match 4+%d classes", dims, nc)
	}
}

// Close 释放 ONNX 资源。
func (d *Detector) Close() error {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.input != nil {
		d.input.Destroy()
	}
	if d.output != nil {
		d.output.Destroy()
	}
	return nil
}

// Option 配置单次推理。
type Option func(*predictOpts)

type predictOpts struct {
	confThresh float32 // <=0 表示用检测器默认值
}

// WithConfThresh 为本次推理覆盖置信度阈值（不影响检测器默认值）。
func WithConfThresh(t float32) Option {
	return func(o *predictOpts) { o.confThresh = t }
}

// Predict 对一张图做检测，返回映射回原图坐标的结果。
func (d *Detector) Predict(img image.Image, opts ...Option) ([]Detection, error) {
	o := predictOpts{}
	for _, fn := range opts {
		fn(&o)
	}

	// 共享输入/输出张量缓冲，推理必须串行。
	d.runMu.Lock()
	defer d.runMu.Unlock()

	letter, err := letterbox(img, d.cfg.InputW, d.cfg.InputH)
	if err != nil {
		return nil, fmt.Errorf("preprocess: %w", err)
	}
	// 写入 NCHW float32 输入缓冲并归一化
	fillNCHW(d.inputBuf, letter.rgba, d.cfg.InputW, d.cfg.InputH)

	if err := d.session.Run(); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	raw := d.postprocess(o.confThresh)

	d.mu.RLock()
	classNames := d.cfg.Classes
	d.mu.RUnlock()

	// letterbox 坐标映射回原图
	scale := letter.scale
	dets := make([]Detection, 0, len(raw))
	for _, r := range raw {
		x1 := (r.x1 - letter.padX) / scale
		y1 := (r.y1 - letter.padY) / scale
		x2 := (r.x2 - letter.padX) / scale
		y2 := (r.y2 - letter.padY) / scale
		// clamp 到原图边界
		b := img.Bounds()
		w, h := float32(b.Dx()), float32(b.Dy())
		x1 = clampF(x1, 0, w)
		y1 = clampF(y1, 0, h)
		x2 = clampF(x2, 0, w)
		y2 = clampF(y2, 0, h)
		name := ""
		if r.classID < len(classNames) {
			name = classNames[r.classID]
		}
		dets = append(dets, Detection{
			ClassID:    r.classID,
			ClassName:  name,
			Confidence: r.conf,
			X1:         x1, Y1: y1, X2: x2, Y2: y2,
		})
	}
	return dets, nil
}

type rawDet struct {
	classID        int
	conf           float32
	x1, y1, x2, y2 float32
}

// postprocess 解析输出张量，按类取最大置信度并做 NMS。
// confOverride <=0 时使用检测器默认阈值。
func (d *Detector) postprocess(confOverride float32) []rawDet {
	attrs := d.outAttrs
	nc := attrs - 4
	d.mu.RLock()
	thresh := d.cfg.ConfThresh
	iouThresh := d.cfg.IoUThresh
	d.mu.RUnlock()
	if confOverride > 0 {
		thresh = confOverride
	}

	cands := make([]rawDet, 0, 256)
	for a := 0; a < d.outAnchors; a++ {
		// cx,cy,w,h 在 attrs 维度前 4 个；类别分数在后
		var cx, cy, w, h float32
		if d.outTransposed {
			// [anchors, attrs]
			base := a * attrs
			cx = d.outputBuf[base]
			cy = d.outputBuf[base+1]
			w = d.outputBuf[base+2]
			h = d.outputBuf[base+3]
		} else {
			// [attrs, anchors]
			cx = d.outputBuf[a]
			cy = d.outputBuf[d.outAnchors+a]
			w = d.outputBuf[2*d.outAnchors+a]
			h = d.outputBuf[3*d.outAnchors+a]
		}

		// 找最大类别分数
		var cls int = -1
		var best float32 = thresh
		for c := 0; c < nc; c++ {
			var s float32
			if d.outTransposed {
				s = d.outputBuf[a*attrs+4+c]
			} else {
				s = d.outputBuf[(4+c)*d.outAnchors+a]
			}
			if s > best {
				best = s
				cls = c
			}
		}
		if cls < 0 {
			continue
		}
		cands = append(cands, rawDet{
			classID: cls,
			conf:    best,
			x1:      cx - w/2,
			y1:      cy - h/2,
			x2:      cx + w/2,
			y2:      cy + h/2,
		})
	}
	return nms(cands, iouThresh)
}

// nms 按置信度排序后做非极大值抑制（按类分组）。
func nms(dets []rawDet, iouThresh float32) []rawDet {
	sort.SliceStable(dets, func(i, j int) bool {
		return dets[i].conf > dets[j].conf
	})
	keep := make([]rawDet, 0, len(dets))
	suppressed := make([]bool, len(dets))
	for i := 0; i < len(dets); i++ {
		if suppressed[i] {
			continue
		}
		keep = append(keep, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if suppressed[j] || dets[j].classID != dets[i].classID {
				continue
			}
			if iou(dets[i], dets[j]) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func iou(a, b rawDet) float32 {
	ix1 := maxF(a.x1, b.x1)
	iy1 := maxF(a.y1, b.y1)
	ix2 := minF(a.x2, b.x2)
	iy2 := minF(a.y2, b.y2)
	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	areaA := (a.x2 - a.x1) * (a.y2 - a.y1)
	areaB := (b.x2 - b.x1) * (b.y2 - b.y1)
	return inter / (areaA + areaB - inter + 1e-7)
}

func maxF(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
func minF(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
func clampF(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
