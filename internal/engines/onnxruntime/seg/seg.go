// Package seg 实现基于 ONNX Runtime 的 YOLOv8 实例分割引擎。
//
// 适配 ultralytics 导出的 YOLOv8-seg ONNX：
//   - 输入:  float32[1,3,H,W]，NCHW，letterbox 归一化（同 detect）
//   - 输出1: 检测头 [1,4+nc+nm,anchors]（或转置），nm 为掩膜系数维度（通常 32）
//   - 输出2: 原型掩膜 [1,nm,mh,mw]（通常 [1,32,160,160]）
//
// 每个保留实例：掩膜 = sigmoid(系数 · 原型)，按框裁切，上采样回输入分辨率，
// 0.5 二值化后追踪外轮廓多边形，再经 letterbox 映射回原图。
package seg

import (
	"context"
	"fmt"
	"image"
	"math"
	"sync"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
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
}

// Engine 是 engine.Engine 的 ONNX Runtime YOLOv8-seg 实现。
type Engine struct {
	cfg   Config
	runMu sync.Mutex // 共享输入/输出张量，推理串行

	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	detOut  *ort.Tensor[float32]
	proto   *ort.Tensor[float32]

	inputBuf []float32
	detBuf   []float32
	protoBuf []float32

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

	inputBuf := make([]float32, 3*cfg.InputH*cfg.InputW)
	input, err := ort.NewTensor(ort.NewShape(1, 3, int64(cfg.InputH), int64(cfg.InputW)), inputBuf)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	detShape := ort.NewShape(1)
	if transposed {
		detShape = append(detShape, int64(anchors), int64(attrs))
	} else {
		detShape = append(detShape, int64(attrs), int64(anchors))
	}
	detBuf := make([]float32, anchors*attrs)
	detOut, err := ort.NewTensor(detShape, detBuf)
	if err != nil {
		input.Destroy()
		return nil, fmt.Errorf("create det tensor: %w", err)
	}

	protoBuf := make([]float32, nm*mh*mw)
	protoTensor, err := ort.NewTensor(ort.NewShape(1, int64(nm), int64(mh), int64(mw)), protoBuf)
	if err != nil {
		input.Destroy()
		detOut.Destroy()
		return nil, fmt.Errorf("create proto tensor: %w", err)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		input.Destroy()
		detOut.Destroy()
		protoTensor.Destroy()
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewAdvancedSession(
		cfg.ModelPath,
		[]string{inputs[0].Name},
		[]string{detInfo.Name, protoInfo.Name},
		[]ort.Value{input},
		[]ort.Value{detOut, protoTensor},
		options,
	)
	if err != nil {
		input.Destroy()
		detOut.Destroy()
		protoTensor.Destroy()
		return nil, fmt.Errorf("create session: %w", err)
	}

	return &Engine{
		cfg:        cfg,
		session:    session,
		input:      input,
		detOut:     detOut,
		proto:      protoTensor,
		inputBuf:   inputBuf,
		detBuf:     detBuf,
		protoBuf:   protoBuf,
		anchors:    anchors,
		attrs:      attrs,
		nc:         nc,
		nm:         nm,
		transposed: transposed,
		maskH:      mh,
		maskW:      mw,
	}, nil
}

func (e *Engine) Name() string      { return e.cfg.Name }
func (e *Engine) Task() engine.Task { return engine.TaskSegmentation }
func (e *Engine) Framework() string { return "onnxruntime" }

// Run 执行一次实例分割。
func (e *Engine) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	if req.Image == nil {
		return nil, fmt.Errorf("segmentation requires an image")
	}
	conf := e.cfg.ConfThresh
	if v, ok := req.Params["conf"]; ok && v > 0 {
		conf = v
	}

	e.runMu.Lock()
	defer e.runMu.Unlock()

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	lb := preprocess.Letterbox(req.Image, e.cfg.InputW, e.cfg.InputH)
	preprocess.FillNCHW(e.inputBuf, lb.RGBA, e.cfg.InputW, e.cfg.InputH)

	start := time.Now()
	if err := e.session.Run(); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}
	inferElapsed := time.Since(start)

	cands := e.decodeDetections(conf)
	kept := e.nms(cands)
	instances := e.makeMasks(kept, lb, req.Image.Bounds())

	return &engine.SegmentationResult{Elapsed: inferElapsed, Instances: instances}, nil
}

type cand struct {
	box    engine.Box
	coeffs []float32 // 长度 nm
}

// decodeDetections 解析检测头：cx cy w h + nc 类别分 + nm 掩膜系数。
func (e *Engine) decodeDetections(conf float32) []cand {
	nc, nm, anchors := e.nc, e.nm, e.anchors
	out := make([]cand, 0, 128)
	for a := 0; a < anchors; a++ {
		var cx, cy, w, h float32
		if e.transposed {
			base := a * e.attrs
			cx = e.detBuf[base]
			cy = e.detBuf[base+1]
			w = e.detBuf[base+2]
			h = e.detBuf[base+3]
		} else {
			cx = e.detBuf[a]
			cy = e.detBuf[anchors+a]
			w = e.detBuf[2*anchors+a]
			h = e.detBuf[3*anchors+a]
		}
		cls, best := -1, conf
		for c := 0; c < nc; c++ {
			var s float32
			if e.transposed {
				s = e.detBuf[a*e.attrs+4+c]
			} else {
				s = e.detBuf[(4+c)*anchors+a]
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
				coeffs[k] = e.detBuf[a*e.attrs+4+nc+k]
			} else {
				coeffs[k] = e.detBuf[(4+nc+k)*anchors+a]
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
	// 按置信度降序稳定排序，同类 IoU 超阈值则抑制。候选数量小，插入排序即可。
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
		bi := preprocess.NMSBox{
			ClassID: cands[i].box.ClassID, Confidence: cands[i].box.Confidence,
			X1: cands[i].box.X1, Y1: cands[i].box.Y1, X2: cands[i].box.X2, Y2: cands[i].box.Y2,
		}
		for j := i + 1; j < len(cands); j++ {
			if suppressed[j] || cands[j].box.ClassID != cands[i].box.ClassID {
				continue
			}
			bj := preprocess.NMSBox{
				ClassID: cands[j].box.ClassID, Confidence: cands[j].box.Confidence,
				X1: cands[j].box.X1, Y1: cands[j].box.Y1, X2: cands[j].box.X2, Y2: cands[j].box.Y2,
			}
			if preprocess.IoU(bi, bj) > e.cfg.IoUThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

// makeMasks 对每个保留实例计算二值掩膜、追踪轮廓并映射回原图。
func (e *Engine) makeMasks(cands []cand, lb *preprocess.Letterboxed, orig image.Rectangle) []engine.Instance {
	origW, origH := float32(orig.Dx()), float32(orig.Dy())
	inW, inH := float32(e.cfg.InputW), float32(e.cfg.InputH)
	mW, mH := float32(e.maskW), float32(e.maskH)
	sx, sy := mW/inW, mH/inH

	instances := make([]engine.Instance, 0, len(cands))
	for _, c := range cands {
		b := c.box
		// 输入空间框，clamp 到画布。
		ix1 := preprocess.Clamp(b.X1, 0, inW)
		iy1 := preprocess.Clamp(b.Y1, 0, inH)
		ix2 := preprocess.Clamp(b.X2, 0, inW)
		iy2 := preprocess.Clamp(b.Y2, 0, inH)
		bw := int(ix2 - ix1)
		bh := int(iy2 - iy1)
		if bw < 2 || bh < 2 {
			continue
		}

		// 在掩膜原型空间内只处理落在框内的像素，双线性采样加权和并 sigmoid。
		mask := make([]bool, bw*bh)
		for py := 0; py < bh; py++ {
			my := (float32(py) + 0.5 + iy1) * sy // 掩膜空间 y（像素中心，注意用 iy1 而非 ix1）
			for px := 0; px < bw; px++ {
				mx := (float32(px) + 0.5 + ix1) * sx
				v := e.sampleProto(mx, my, c.coeffs)
				if v >= e.cfg.MaskThresh {
					mask[py*bw+px] = true
				}
			}
		}

		polys := contour(mask, bw, bh)
		if len(polys) == 0 {
			continue
		}

		// 轮廓点从框局部坐标 → 输入空间 → 原图。
		mapped := make([][]engine.Point, 0, len(polys))
		for _, poly := range polys {
			mp := make([]engine.Point, 0, len(poly))
			for _, p := range poly {
				gx := float32(p.x) + ix1
				gy := float32(p.y) + iy1
				ox := preprocess.Clamp((gx-lb.PadX)/lb.Scale, 0, origW)
				oy := preprocess.Clamp((gy-lb.PadY)/lb.Scale, 0, origH)
				mp = append(mp, engine.Point{X: ox, Y: oy})
			}
			mapped = append(mapped, mp)
		}

		name := ""
		if b.ClassID < len(e.cfg.Classes) {
			name = e.cfg.Classes[b.ClassID]
		}
		// 框也映射回原图。
		x1 := preprocess.Clamp((b.X1-lb.PadX)/lb.Scale, 0, origW)
		y1 := preprocess.Clamp((b.Y1-lb.PadY)/lb.Scale, 0, origH)
		x2 := preprocess.Clamp((b.X2-lb.PadX)/lb.Scale, 0, origW)
		y2 := preprocess.Clamp((b.Y2-lb.PadY)/lb.Scale, 0, origH)

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

// sampleProto 在原型掩膜上做双线性采样，返回 sigmoid(sum_k coeffs[k]*proto[k])。
func (e *Engine) sampleProto(mx, my float32, coeffs []float32) float32 {
	x0 := int(math.Floor(float64(mx)))
	y0 := int(math.Floor(float64(my)))
	x1 := x0 + 1
	y1 := y0 + 1
	// 四个角都 clamp 到掩膜网格内：像素中心可能恰好落在右/下边界上，
	// 此时 x0/y0 本身也会越界。
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

	plane := e.maskW * e.maskH
	var sum float32
	for k := 0; k < e.nm; k++ {
		base := k * plane
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

func (e *Engine) Close() error {
	if e.session != nil {
		e.session.Destroy()
	}
	if e.input != nil {
		e.input.Destroy()
	}
	if e.detOut != nil {
		e.detOut.Destroy()
	}
	if e.proto != nil {
		e.proto.Destroy()
	}
	return nil
}

// ---------- 输出形状解析 ----------

// classifyOutputs 在两个输出中区分检测头（3D，末维含 4+nc）与原型（4D）。
func classifyOutputs(outputs []ort.InputOutputInfo, nc int) (det, proto ort.InputOutputInfo, err error) {
	for _, o := range outputs {
		d := o.Dimensions
		if len(d) == 4 && d[0] == 1 {
			proto = o
		} else if len(d) == 3 && d[0] == 1 {
			det = o
		}
	}
	if det.Name == "" || proto.Name == "" {
		return det, proto, fmt.Errorf("cannot identify det/proto outputs among shapes %v, %v",
			outputs[0].Dimensions, outputs[1].Dimensions)
	}
	return det, proto, nil
}

// parseDetShape 解析检测头形状，返回 anchors/attrs/nm/是否转置。
func parseDetShape(dims ort.Shape, nc int) (anchors, attrs, nm int, transposed bool, err error) {
	if len(dims) != 3 || dims[0] != 1 {
		return 0, 0, 0, false, fmt.Errorf("unsupported det output shape %v", dims)
	}
	d1, d2 := int(dims[1]), int(dims[2])
	if d1 < 0 || d2 < 0 {
		return 0, 0, 0, false, fmt.Errorf("dynamic dim %v not supported", dims)
	}
	// attrs = 4+nc+nm 是较小维（~116），anchors 是较大维（~8400）。
	switch {
	case d1 < d2:
		// 官方排布 [1, attrs, anchors]
		attrs, anchors, transposed = d1, d2, false
	case d2 < d1:
		// 转置排布 [1, anchors, attrs]
		attrs, anchors, transposed = d2, d1, true
	default:
		return 0, 0, 0, false, fmt.Errorf("cannot distinguish anchors/attrs in square det shape %v", dims)
	}
	if attrs <= 4+nc {
		return 0, 0, 0, false, fmt.Errorf("det output shape %v: attrs=%d does not exceed 4+%d classes", dims, attrs, nc)
	}
	return anchors, attrs, attrs - 4 - nc, transposed, nil
}

// parseProtoShape 解析原型输出 [1,nm,mh,mw]。
func parseProtoShape(dims ort.Shape, nmExpected int) (mh, mw int, err error) {
	if len(dims) != 4 || dims[0] != 1 {
		return 0, 0, fmt.Errorf("unsupported proto shape %v", dims)
	}
	nm, mh64, mw64 := int(dims[1]), int(dims[2]), int(dims[3])
	if nm != nmExpected {
		return 0, 0, fmt.Errorf("proto nm=%d but det coeffs=%d", nm, nmExpected)
	}
	if mh64 <= 0 || mw64 <= 0 {
		return 0, 0, fmt.Errorf("dynamic proto dim %v not supported", dims)
	}
	return mh64, mw64, nil
}
