// Package seg 实现基于 ONNX Runtime 的 YOLOv8 实例分割引擎。
//
// 适配 ultralytics 导出的 YOLOv8-seg ONNX：
//   - 输入:  float32[N,3,H,W]，NCHW，letterbox 归一化（同 detect）
//   - 输出1: 检测头 [N,4+nc+nm,anchors]（或转置），nm 为掩膜系数维度（通常 32）
//   - 输出2: 原型掩膜 [N,nm,mh,mw]（通常 [N,32,160,160]）
//
// 每个保留实例：掩膜 = sigmoid(系数 · 原型)，按框裁切，上采样回输入分辨率，
// 0.5 二值化后追踪外轮廓多边形，再经 letterbox 映射回原图。N 由模型 batch 维
// 决定，支持 dynamic batching，预处理在 Prepare 中于设备锁之外完成。
package seg

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/ortbatch"
	"github.com/andaoai/go-infer/internal/postprocess"
	"github.com/andaoai/go-infer/internal/preprocess"
	ort "github.com/yalue/onnxruntime_go"
)

// Config 是 ONNX YOLOv8-seg 引擎的配置。
type Config struct {
	Name       string
	ModelPath  string
	InputW     int
	InputH     int
	Classes    []string
	ConfThresh float32
	IoUThresh  float32
	MaskThresh float32 // 掩膜二值化阈值，默认 0.5
	// MaxBatch 仅在模型输入 batch 维为动态（<=0）时生效，作为合批上限。
	MaxBatch int
}

// Engine 是 engine.Engine / engine.BatchEngine 的 ONNX Runtime YOLOv8-seg 实现。
// 输入缓冲、session、pool 与设备锁由内嵌的 *ortbatch.Runtime 管理；本结构只保留
// 检测头/原型输出形状与输出缓冲。
type Engine struct {
	*ortbatch.Runtime
	cfg Config

	detBuf      []float32 // maxBatch*detStride
	protoBuf    []float32 // maxBatch*protoStride
	detStride   int       // anchors*attrs，每图检测头元素数
	protoStride int       // nm*mh*mw，每图原型元素数

	// 检测头布局
	anchors    int
	attrs      int // 4 + nc + nm
	nc         int
	nm         int
	transposed bool
	// 原型掩膜
	maskH, maskW int
}

// New 创建并初始化 seg 引擎。
func New(cfg Config) (*Engine, error) {
	if cfg.Name == "" {
		cfg.Name = "yolov8n-seg"
	}
	if cfg.ConfThresh <= 0 {
		cfg.ConfThresh = 0.25
	}
	if cfg.IoUThresh <= 0 {
		cfg.IoUThresh = 0.45
	}
	if cfg.MaskThresh <= 0 {
		cfg.MaskThresh = 0.5
	}
	if cfg.InputW <= 0 || cfg.InputH <= 0 {
		return nil, fmt.Errorf("input size must be positive, got %dx%d", cfg.InputW, cfg.InputH)
	}

	inputs, outputs, err := ort.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("read model IO info: %w", err)
	}
	if len(inputs) != 1 {
		return nil, fmt.Errorf("expect 1 input, got %d", len(inputs))
	}
	if len(outputs) != 2 {
		return nil, fmt.Errorf("seg model expects 2 outputs (det+proto), got %d", len(outputs))
	}
	nc := len(cfg.Classes)
	detInfo, protoInfo, err := classifyOutputs(outputs, nc)
	if err != nil {
		return nil, err
	}

	anchors, attrs, nm, transposed, err := parseDetShape(detInfo.Dimensions, nc)
	if err != nil {
		return nil, err
	}
	mh, mw, err := parseProtoShape(protoInfo.Dimensions, nm)
	if err != nil {
		return nil, err
	}

	rt, err := ortbatch.New(ortbatch.Config{
		ModelPath:   cfg.ModelPath,
		InputNames:  []string{inputs[0].Name},
		OutputNames: []string{detInfo.Name, protoInfo.Name},
		InputDims:   inputs[0].Dimensions,
		W:           cfg.InputW,
		H:           cfg.InputH,
		MaxBatch:    cfg.MaxBatch,
		DefaultConf: cfg.ConfThresh,
	})
	if err != nil {
		return nil, err
	}

	detStride := anchors * attrs
	protoStride := nm * mh * mw
	return &Engine{
		Runtime:     rt,
		cfg:         cfg,
		detBuf:      make([]float32, rt.MaxBatch()*detStride),
		protoBuf:    make([]float32, rt.MaxBatch()*protoStride),
		detStride:   detStride,
		protoStride: protoStride,
		anchors:     anchors,
		attrs:       attrs,
		nc:          nc,
		nm:          nm,
		transposed:  transposed,
		maskH:       mh,
		maskW:       mw,
	}, nil
}

func (e *Engine) Name() string      { return e.cfg.Name }
func (e *Engine) Task() engine.Task { return engine.TaskSegmentation }
func (e *Engine) Framework() string { return "onnxruntime" }

// Run 是单图便捷路径。
func (e *Engine) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	return e.Runtime.RunOnce(ctx, req, e.buildOutput, e.postSlot)
}

// RunBatch 在一次底层推理中处理 n 张已预处理的图。
func (e *Engine) RunBatch(ctx context.Context, batch []*engine.Prepared) ([]engine.Result, error) {
	return e.Runtime.RunBatch(ctx, batch, e.buildOutput, e.postSlot)
}

// buildOutput 在引擎自有 detBuf/protoBuf 上按本次 batch n 构造检测头与原型两个输出张量。
func (e *Engine) buildOutput(n int) (ortbatch.Output, error) {
	var detShape ort.Shape
	if e.transposed {
		detShape = ort.NewShape(int64(n), int64(e.anchors), int64(e.attrs))
	} else {
		detShape = ort.NewShape(int64(n), int64(e.attrs), int64(e.anchors))
	}
	detOut, err := ort.NewTensor(detShape, e.detBuf[:n*e.detStride])
	if err != nil {
		return ortbatch.Output{}, fmt.Errorf("create det tensor: %w", err)
	}
	protoTensor, err := ort.NewTensor(
		ort.NewShape(int64(n), int64(e.nm), int64(e.maskH), int64(e.maskW)),
		e.protoBuf[:n*e.protoStride],
	)
	if err != nil {
		detOut.Destroy()
		return ortbatch.Output{}, fmt.Errorf("create proto tensor: %w", err)
	}
	return ortbatch.Output{
		Values: []ort.Value{detOut, protoTensor},
		Done:   func() { detOut.Destroy(); protoTensor.Destroy() },
	}, nil
}

// postSlot 对第 slot 个输出解码、NMS、生成掩膜并映射回原图。
func (e *Engine) postSlot(slot int, p *engine.Prepared, elapsed time.Duration) (engine.Result, error) {
	m := p.Meta.(*ortbatch.Meta)
	cands := e.decodeDetections(slot, m.Conf)
	kept := e.nms(cands)
	instances := e.makeMasks(slot, kept, m)
	return &engine.SegmentationResult{Elapsed: elapsed, Instances: instances}, nil
}

type cand struct {
	box    engine.Box
	coeffs []float32 // 长度 nm
}

// decodeDetections 解析第 slot 个检测头：cx cy w h + nc 类别分 + nm 掩膜系数。
func (e *Engine) decodeDetections(slot int, conf float32) []cand {
	nc, nm, anchors := e.nc, e.nm, e.anchors
	base0 := slot * e.detStride
	out := make([]cand, 0, 128)
	for a := 0; a < anchors; a++ {
		var cx, cy, w, h float32
		if e.transposed {
			base := base0 + a*e.attrs
			cx = e.detBuf[base]
			cy = e.detBuf[base+1]
			w = e.detBuf[base+2]
			h = e.detBuf[base+3]
		} else {
			cx = e.detBuf[base0+a]
			cy = e.detBuf[base0+anchors+a]
			w = e.detBuf[base0+2*anchors+a]
			h = e.detBuf[base0+3*anchors+a]
		}
		cls, best := -1, conf
		for c := 0; c < nc; c++ {
			var s float32
			if e.transposed {
				s = e.detBuf[base0+a*e.attrs+4+c]
			} else {
				s = e.detBuf[base0+(4+c)*anchors+a]
			}
			if s > best {
				best, cls = s, c
			}
		}
		if cls < 0 {
			continue
		}
		coeffs := make([]float32, nm)
		for k := 0; k < nm; k++ {
			if e.transposed {
				coeffs[k] = e.detBuf[base0+a*e.attrs+4+nc+k]
			} else {
				coeffs[k] = e.detBuf[base0+(4+nc+k)*anchors+a]
			}
		}
		out = append(out, cand{
			box: engine.Box{
				ClassID: cls, Confidence: best,
				X1: cx - w/2, Y1: cy - h/2, X2: cx + w/2, Y2: cy + h/2,
			},
			coeffs: coeffs,
		})
	}
	return out
}

func (e *Engine) nms(cands []cand) []cand {
	// 候选携带掩膜系数，不能直接用 postprocess.NMS；按置信度降序稳定排序，
	// 同类 IoU 超阈值则抑制，IoU 复用 postprocess.IoU。
	for i := 1; i < len(cands); i++ {
		for j := i; j > 0 && cands[j-1].box.Confidence < cands[j].box.Confidence; j-- {
			cands[j-1], cands[j] = cands[j], cands[j-1]
		}
	}
	suppressed := make([]bool, len(cands))
	keep := make([]cand, 0, len(cands))
	for i := 0; i < len(cands); i++ {
		if suppressed[i] {
			continue
		}
		keep = append(keep, cands[i])
		for j := i + 1; j < len(cands); j++ {
			if suppressed[j] || cands[j].box.ClassID != cands[i].box.ClassID {
				continue
			}
			if postprocess.IoU(cands[i].box, cands[j].box) > e.cfg.IoUThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

// makeMasks 对每个保留实例计算二值掩膜、追踪轮廓并映射回原图。
func (e *Engine) makeMasks(slot int, cands []cand, m *ortbatch.Meta) []engine.Instance {
	origW, origH := float32(m.Orig.Dx()), float32(m.Orig.Dy())
	inW, inH := float32(e.cfg.InputW), float32(e.cfg.InputH)
	mW, mH := float32(e.maskW), float32(e.maskH)
	sx, sy := mW/inW, mH/inH

	instances := make([]engine.Instance, 0, len(cands))
	for _, c := range cands {
		b := c.box
		ix1 := preprocess.Clamp(b.X1, 0, inW)
		iy1 := preprocess.Clamp(b.Y1, 0, inH)
		ix2 := preprocess.Clamp(b.X2, 0, inW)
		iy2 := preprocess.Clamp(b.Y2, 0, inH)
		bw := int(ix2 - ix1)
		bh := int(iy2 - iy1)
		if bw < 2 || bh < 2 {
			continue
		}

		mask := make([]bool, bw*bh)
		for py := 0; py < bh; py++ {
			my := (float32(py) + 0.5 + iy1) * sy // 掩膜空间 y（像素中心）
			for px := 0; px < bw; px++ {
				mx := (float32(px) + 0.5 + ix1) * sx
				v := e.sampleProto(slot, mx, my, c.coeffs)
				if v >= e.cfg.MaskThresh {
					mask[py*bw+px] = true
				}
			}
		}

		polys := contour(mask, bw, bh)
		if len(polys) == 0 {
			continue
		}

		mapped := make([][]engine.Point, 0, len(polys))
		for _, poly := range polys {
			mp := make([]engine.Point, 0, len(poly))
			for _, p := range poly {
				gx := float32(p.x) + ix1
				gy := float32(p.y) + iy1
				ox := preprocess.Clamp((gx-m.PadX)/m.Scale, 0, origW)
				oy := preprocess.Clamp((gy-m.PadY)/m.Scale, 0, origH)
				mp = append(mp, engine.Point{X: ox, Y: oy})
			}
			mapped = append(mapped, mp)
		}

		name := ""
		if b.ClassID < len(e.cfg.Classes) {
			name = e.cfg.Classes[b.ClassID]
		}
		x1 := preprocess.Clamp((b.X1-m.PadX)/m.Scale, 0, origW)
		y1 := preprocess.Clamp((b.Y1-m.PadY)/m.Scale, 0, origH)
		x2 := preprocess.Clamp((b.X2-m.PadX)/m.Scale, 0, origW)
		y2 := preprocess.Clamp((b.Y2-m.PadY)/m.Scale, 0, origH)

		instances = append(instances, engine.Instance{
			Detection: engine.Detection{
				ClassID: b.ClassID, ClassName: name, Confidence: b.Confidence,
				X1: x1, Y1: y1, X2: x2, Y2: y2,
			},
			Mask: mapped,
		})
	}
	return instances
}

// sampleProto 在第 slot 个原型掩膜上做双线性采样，返回 sigmoid(sum_k coeffs[k]*proto[k])。
func (e *Engine) sampleProto(slot int, mx, my float32, coeffs []float32) float32 {
	x0 := int(math.Floor(float64(mx)))
	y0 := int(math.Floor(float64(my)))
	x1 := x0 + 1
	y1 := y0 + 1
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x0 >= e.maskW {
		x0 = e.maskW - 1
	}
	if y0 >= e.maskH {
		y0 = e.maskH - 1
	}
	if x1 >= e.maskW {
		x1 = e.maskW - 1
	}
	if y1 >= e.maskH {
		y1 = e.maskH - 1
	}
	dx := mx - float32(x0)
	dy := my - float32(y0)
	w00 := (1 - dx) * (1 - dy)
	w10 := dx * (1 - dy)
	w01 := (1 - dx) * dy
	w11 := dx * dy

	batchOff := slot * e.protoStride
	plane := e.maskW * e.maskH
	var sum float32
	for k := 0; k < e.nm; k++ {
		base := batchOff + k*plane
		p := coeffs[k] * (w00*e.protoBuf[base+y0*e.maskW+x0] +
			w10*e.protoBuf[base+y0*e.maskW+x1] +
			w01*e.protoBuf[base+y1*e.maskW+x0] +
			w11*e.protoBuf[base+y1*e.maskW+x1])
		sum += p
	}
	return sigmoid(sum)
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// ---------- 输出形状解析 ----------

// classifyOutputs 在两个输出中区分检测头（3D）与原型（4D）。batch 维任意。
func classifyOutputs(outputs []ort.InputOutputInfo, nc int) (det, proto ort.InputOutputInfo, err error) {
	for _, o := range outputs {
		d := o.Dimensions
		if len(d) == 4 {
			proto = o
		} else if len(d) == 3 {
			det = o
		}
	}
	if det.Name == "" || proto.Name == "" {
		return det, proto, fmt.Errorf("cannot identify det/proto outputs among shapes %v, %v",
			outputs[0].Dimensions, outputs[1].Dimensions)
	}
	return det, proto, nil
}

// parseDetShape 解析检测头形状，返回 anchors/attrs/nm/是否转置。dims[0] 为 batch 维，任意。
func parseDetShape(dims ort.Shape, nc int) (anchors, attrs, nm int, transposed bool, err error) {
	if len(dims) != 3 {
		return 0, 0, 0, false, fmt.Errorf("unsupported det output shape %v", dims)
	}
	d1, d2 := int(dims[1]), int(dims[2])
	if d1 <= 0 || d2 <= 0 {
		return 0, 0, 0, false, fmt.Errorf("non-batch det dims must be fixed, got %v", dims)
	}
	// attrs = 4+nc+nm 是较小维（~116），anchors 是较大维（~8400）。
	switch {
	case d1 < d2:
		attrs, anchors, transposed = d1, d2, false // [B, attrs, anchors]
	case d2 < d1:
		attrs, anchors, transposed = d2, d1, true // [B, anchors, attrs]
	default:
		return 0, 0, 0, false, fmt.Errorf("cannot distinguish anchors/attrs in square det shape %v", dims)
	}
	if attrs <= 4+nc {
		return 0, 0, 0, false, fmt.Errorf("det output shape %v: attrs=%d does not exceed 4+%d classes", dims, attrs, nc)
	}
	return anchors, attrs, attrs - 4 - nc, transposed, nil
}

// parseProtoShape 解析原型输出 [B,nm,mh,mw]。dims[0] 为 batch 维，任意。
func parseProtoShape(dims ort.Shape, nmExpected int) (mh, mw int, err error) {
	if len(dims) != 4 {
		return 0, 0, fmt.Errorf("unsupported proto shape %v", dims)
	}
	nm, mh64, mw64 := int(dims[1]), int(dims[2]), int(dims[3])
	if nm != nmExpected {
		return 0, 0, fmt.Errorf("proto nm=%d but det coeffs=%d", nm, nmExpected)
	}
	if mh64 <= 0 || mw64 <= 0 {
		return 0, 0, fmt.Errorf("non-batch proto dims must be fixed, got %v", dims)
	}
	return mh64, mw64, nil
}
