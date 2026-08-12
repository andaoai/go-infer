// Package detect 实现基于 ONNX Runtime 的 YOLO 检测引擎。
//
// 适配 ultralytics 导出的 ONNX 检测模型：
//   - 输入:  float32[N,3,H,W]，NCHW，像素归一化 0~1（letterbox 填充 114）
//   - 输出:  官方 [N,4+nc,anchors] 或转置 [N,anchors,4+nc]
//
// N 由模型输入 batch 维决定：固定 1 则只能单图；动态维则可在配置上限内
// 合批。预处理（letterbox/归一化）在 Prepare 中于设备锁之外完成，RunBatch
// 只做拷贝、建张量、推理、后处理，供 sched 做流水线与 dynamic batching。
package detect

import (
	"context"
	"fmt"
	"image"
	"sync"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/ortutil"
	"github.com/andaoai/go-infer/internal/preprocess"
	ort "github.com/yalue/onnxruntime_go"
)

// Config 是 ONNX YOLO 检测引擎的配置。
type Config struct {
	Name       string // 引擎实例名，空则用 "yolo-onnx"
	ModelPath  string
	InputW     int
	InputH     int
	Classes    []string
	ConfThresh float32
	IoUThresh  float32
	// MaxBatch 仅在模型输入 batch 维为动态（<=0）时生效，作为合批上限；
	// 固定 batch 的模型忽略此值。<=0 用 defaultMaxBatch。
	MaxBatch int
}

// detMeta 是每个预处理项的引擎私有元信息，只含标量，不引用借出的缓冲。
type detMeta struct {
	scale float32
	padX  float32
	padY  float32
	orig  image.Rectangle
	conf  float32
}

// Engine 是 engine.Engine / engine.BatchEngine 的 ONNX Runtime YOLO 实现。
type Engine struct {
	cfg   Config
	runMu sync.Mutex // 共享批缓冲与 session，RunBatch 串行

	session *ort.DynamicAdvancedSession

	maxBatch  int
	planeSize int // 3*H*W，每图输入元素数
	outStride int // anchors*attrs，每图输出元素数

	inputBuf  []float32 // maxBatch*planeSize
	outputBuf []float32 // maxBatch*outStride
	pool      sync.Pool // 借出 []float32，len=planeSize

	outAnchors    int
	outAttrs      int // 4 + nc
	outTransposed bool
}

// New 创建并初始化引擎，从模型元数据读取真实 IO 名与输出形状。
func New(cfg Config) (*Engine, error) {
	if cfg.Name == "" {
		cfg.Name = "yolo-onnx"
	}
	if cfg.ConfThresh <= 0 {
		cfg.ConfThresh = 0.25
	}
	if cfg.IoUThresh <= 0 {
		cfg.IoUThresh = 0.45
	}
	if cfg.InputW <= 0 || cfg.InputH <= 0 {
		return nil, fmt.Errorf("input size must be positive, got %dx%d", cfg.InputW, cfg.InputH)
	}

	inputs, outputs, err := ort.GetInputOutputInfo(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("read model IO info: %w", err)
	}
	if len(inputs) != 1 || len(outputs) != 1 {
		return nil, fmt.Errorf("expect 1 input / 1 output, got %d/%d", len(inputs), len(outputs))
	}
	inInfo, outInfo := inputs[0], outputs[0]

	anchors, attrs, transposed, err := parseOutputShape(outInfo.Dimensions, len(cfg.Classes))
	if err != nil {
		return nil, err
	}

	maxBatch, err := ortutil.ResolveMaxBatch(inInfo.Dimensions, cfg.MaxBatch)
	if err != nil {
		return nil, err
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{inInfo.Name},
		[]string{outInfo.Name},
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	planeSize := 3 * cfg.InputH * cfg.InputW
	outStride := anchors * attrs
	e := &Engine{
		cfg:           cfg,
		session:       session,
		maxBatch:      maxBatch,
		planeSize:     planeSize,
		outStride:     outStride,
		inputBuf:      make([]float32, maxBatch*planeSize),
		outputBuf:     make([]float32, maxBatch*outStride),
		outAnchors:    anchors,
		outAttrs:      attrs,
		outTransposed: transposed,
	}
	e.pool.New = func() any { return make([]float32, planeSize) }
	return e, nil
}

func (e *Engine) Name() string      { return e.cfg.Name }
func (e *Engine) Task() engine.Task { return engine.TaskDetection }
func (e *Engine) Framework() string { return "onnxruntime" }
func (e *Engine) MaxBatch() int     { return e.maxBatch }

// Prepare 在设备锁之外完成一张图的 letterbox/归一化，借出缓冲由 Release 归还。
func (e *Engine) Prepare(ctx context.Context, req *engine.Request) (*engine.Prepared, error) {
	if req.Image == nil {
		return nil, fmt.Errorf("detection requires an image")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	conf := e.cfg.ConfThresh
	if v, ok := req.Params["conf"]; ok && v > 0 {
		conf = v
	}
	buf := e.pool.Get().([]float32)
	lb := preprocess.Letterbox(req.Image, e.cfg.InputW, e.cfg.InputH)
	preprocess.FillNCHW(buf, lb.RGBA, e.cfg.InputW, e.cfg.InputH)
	return &engine.Prepared{
		Input: buf,
		Meta: &detMeta{
			scale: lb.Scale,
			padX:  lb.PadX,
			padY:  lb.PadY,
			orig:  req.Image.Bounds(),
			conf:  conf,
		},
	}, nil
}

// Release 归还 Prepare 借出的缓冲，对 nil 安全、可重复调用。
func (e *Engine) Release(p *engine.Prepared) {
	if p == nil || p.Input == nil {
		return
	}
	e.pool.Put(p.Input)
	p.Input = nil
}

// Run 是单图便捷路径，等价于 Prepare + RunBatch(1) + Release。
func (e *Engine) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	p, err := e.Prepare(ctx, req)
	if err != nil {
		return nil, err
	}
	defer e.Release(p)
	results, err := e.RunBatch(ctx, []*engine.Prepared{p})
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// RunBatch 在一次底层推理中处理 n 张已预处理的图，返回等长结果。
// n∈[1,MaxBatch]；整批成功或整批失败。后处理在锁内逐槽完成。
func (e *Engine) RunBatch(ctx context.Context, batch []*engine.Prepared) ([]engine.Result, error) {
	n := len(batch)
	if n == 0 || n > e.maxBatch {
		return nil, fmt.Errorf("invalid batch size %d (max %d)", n, e.maxBatch)
	}

	e.runMu.Lock()
	defer e.runMu.Unlock()

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// 拷贝各图输入到批缓冲。C 张量指向批缓冲，而非 per-request 缓冲，
	// 因此 RunBatch 返回后调度器即可安全归还 per-request 缓冲。
	for i, p := range batch {
		copy(e.inputBuf[i*e.planeSize:(i+1)*e.planeSize], p.Input)
	}

	in, err := ort.NewTensor(
		ort.NewShape(int64(n), 3, int64(e.cfg.InputH), int64(e.cfg.InputW)),
		e.inputBuf[:n*e.planeSize],
	)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer in.Destroy()

	var outShape ort.Shape
	if e.outTransposed {
		outShape = ort.NewShape(int64(n), int64(e.outAnchors), int64(e.outAttrs))
	} else {
		outShape = ort.NewShape(int64(n), int64(e.outAttrs), int64(e.outAnchors))
	}
	out, err := ort.NewTensor(outShape, e.outputBuf[:n*e.outStride])
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer out.Destroy()

	start := time.Now()
	if err := e.session.Run([]ort.Value{in}, []ort.Value{out}); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}
	inferElapsed := time.Since(start)

	results := make([]engine.Result, n)
	for i, p := range batch {
		m := p.Meta.(*detMeta)
		boxes := e.postprocess(i, m.conf)
		detections := e.mapToImage(boxes, m)
		results[i] = &engine.DetectionResult{Elapsed: inferElapsed, Detections: detections}
	}
	return results, nil
}

// postprocess 解析第 slot 个输出，按类取最大分数并 NMS（按图独立，不跨图抑制）。
func (e *Engine) postprocess(slot int, conf float32) []engine.Box {
	attrs, anchors, nc := e.outAttrs, e.outAnchors, e.outAttrs-4
	base0 := slot * e.outStride
	cands := make([]engine.Box, 0, 256)
	for a := 0; a < anchors; a++ {
		var cx, cy, w, h float32
		if e.outTransposed {
			base := base0 + a*attrs
			cx = e.outputBuf[base]
			cy = e.outputBuf[base+1]
			w = e.outputBuf[base+2]
			h = e.outputBuf[base+3]
		} else {
			cx = e.outputBuf[base0+a]
			cy = e.outputBuf[base0+anchors+a]
			w = e.outputBuf[base0+2*anchors+a]
			h = e.outputBuf[base0+3*anchors+a]
		}
		cls, best := -1, conf
		for c := 0; c < nc; c++ {
			var s float32
			if e.outTransposed {
				s = e.outputBuf[base0+a*attrs+4+c]
			} else {
				s = e.outputBuf[base0+(4+c)*anchors+a]
			}
			if s > best {
				best, cls = s, c
			}
		}
		if cls < 0 {
			continue
		}
		cands = append(cands, engine.Box{
			ClassID:    cls,
			Confidence: best,
			X1:         cx - w/2,
			Y1:         cy - h/2,
			X2:         cx + w/2,
			Y2:         cy + h/2,
		})
	}
	return nms(cands, e.cfg.IoUThresh)
}

// mapToImage 将 letterbox 坐标映射回原图并转为公共 Detection。
func (e *Engine) mapToImage(boxes []engine.Box, m *detMeta) []engine.Detection {
	w, h := float32(m.orig.Dx()), float32(m.orig.Dy())
	out := make([]engine.Detection, 0, len(boxes))
	for _, box := range boxes {
		x1 := preprocess.Clamp((box.X1-m.padX)/m.scale, 0, w)
		y1 := preprocess.Clamp((box.Y1-m.padY)/m.scale, 0, h)
		x2 := preprocess.Clamp((box.X2-m.padX)/m.scale, 0, w)
		y2 := preprocess.Clamp((box.Y2-m.padY)/m.scale, 0, h)
		name := ""
		if box.ClassID < len(e.cfg.Classes) {
			name = e.cfg.Classes[box.ClassID]
		}
		out = append(out, engine.Detection{
			ClassID:    box.ClassID,
			ClassName:  name,
			Confidence: box.Confidence,
			X1:         x1, Y1: y1, X2: x2, Y2: y2,
		})
	}
	return out
}

func (e *Engine) Close() error {
	if e.session != nil {
		return e.session.Destroy()
	}
	return nil
}

// parseOutputShape 将模型输出形状解析为 anchors/attrs/是否转置。
// dims[0] 是 batch 维（固定 1、动态 <=0、或固定 N 均接受）；dim1/dim2 必须固定。
func parseOutputShape(dims ort.Shape, nc int) (anchors, attrs int, transposed bool, err error) {
	if len(dims) != 3 {
		return 0, 0, false, fmt.Errorf("unsupported output shape %v, expect [B, ..., ...]", dims)
	}
	expected := 4 + nc
	d1, d2 := int(dims[1]), int(dims[2])
	if d1 <= 0 || d2 <= 0 {
		return 0, 0, false, fmt.Errorf("non-batch output dims must be fixed, got %v", dims)
	}
	switch {
	case d1 == expected:
		return d2, expected, false, nil // [B, attrs, anchors]
	case d2 == expected:
		return d1, expected, true, nil // [B, anchors, attrs]
	default:
		return 0, 0, false, fmt.Errorf("output shape %v does not match 4+%d classes", dims, nc)
	}
}
