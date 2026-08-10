// Package detect 实现基于 ONNX Runtime 的 YOLO 检测引擎。
//
// 适配 ultralytics 导出的 ONNX 检测模型：
//   - 输入:  float32[1,3,H,W]，NCHW，像素归一化 0~1（letterbox 填充 114）
//   - 输出:  官方 [1,4+nc,anchors] 或转置 [1,anchors,4+nc]
//
// 它是 go-infer 的第一个引擎实现，不绑定具体模型/类别；后续可在此包
// 或新增 engines/ 下扩展 TensorRT、NCNN 等后端。
package detect

import (
	"context"
	"fmt"
	"image"
	"sync"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
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
}

// Engine 是 engine.Engine 的 ONNX Runtime YOLO 实现。
type Engine struct {
	cfg       Config
	runMu     sync.Mutex // 共享输入/输出张量，推理串行
	session   *ort.AdvancedSession
	input     *ort.Tensor[float32]
	output    *ort.Tensor[float32]
	inputBuf  []float32
	outputBuf []float32

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
	inName, outInfo := inputs[0].Name, outputs[0]

	anchors, attrs, transposed, err := parseOutputShape(outInfo.Dimensions, len(cfg.Classes))
	if err != nil {
		return nil, err
	}

	inputBuf := make([]float32, 3*cfg.InputH*cfg.InputW)
	input, err := ort.NewTensor(ort.NewShape(1, 3, int64(cfg.InputH), int64(cfg.InputW)), inputBuf)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outShape := ort.NewShape(1)
	if transposed {
		outShape = append(outShape, int64(anchors), int64(attrs))
	} else {
		outShape = append(outShape, int64(attrs), int64(anchors))
	}
	outputBuf := make([]float32, anchors*attrs)
	output, err := ort.NewTensor(outShape, outputBuf)
	if err != nil {
		input.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		input.Destroy()
		output.Destroy()
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewAdvancedSession(
		cfg.ModelPath,
		[]string{inName},
		[]string{outInfo.Name},
		[]ort.Value{input},
		[]ort.Value{output},
		options,
	)
	if err != nil {
		input.Destroy()
		output.Destroy()
		return nil, fmt.Errorf("create session: %w", err)
	}

	return &Engine{
		cfg:           cfg,
		session:       session,
		input:         input,
		output:        output,
		inputBuf:      inputBuf,
		outputBuf:     outputBuf,
		outAnchors:    anchors,
		outAttrs:      attrs,
		outTransposed: transposed,
	}, nil
}

func (e *Engine) Name() string      { return e.cfg.Name }
func (e *Engine) Task() engine.Task { return engine.TaskDetection }
func (e *Engine) Framework() string { return "onnxruntime" }

// Run 执行一次检测。conf 阈值可通过 req.Params["conf"] 按请求覆盖。
func (e *Engine) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	if req.Image == nil {
		return nil, fmt.Errorf("detection requires an image")
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

	boxes := e.postprocess(conf)
	detections := e.mapToImage(boxes, lb, req.Image.Bounds())

	return &engine.DetectionResult{
		Elapsed:    inferElapsed,
		Detections: detections,
	}, nil
}

// postprocess 解析输出张量，按类取最大分数并 NMS。
func (e *Engine) postprocess(conf float32) []engine.Box {
	attrs, anchors, nc := e.outAttrs, e.outAnchors, e.outAttrs-4
	cands := make([]engine.Box, 0, 256)
	for a := 0; a < anchors; a++ {
		var cx, cy, w, h float32
		if e.outTransposed {
			base := a * attrs
			cx = e.outputBuf[base]
			cy = e.outputBuf[base+1]
			w = e.outputBuf[base+2]
			h = e.outputBuf[base+3]
		} else {
			cx = e.outputBuf[a]
			cy = e.outputBuf[anchors+a]
			w = e.outputBuf[2*anchors+a]
			h = e.outputBuf[3*anchors+a]
		}
		cls, best := -1, conf
		for c := 0; c < nc; c++ {
			var s float32
			if e.outTransposed {
				s = e.outputBuf[a*attrs+4+c]
			} else {
				s = e.outputBuf[(4+c)*anchors+a]
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
func (e *Engine) mapToImage(boxes []engine.Box, lb *preprocess.Letterboxed, b image.Rectangle) []engine.Detection {
	w, h := float32(b.Dx()), float32(b.Dy())
	out := make([]engine.Detection, 0, len(boxes))
	for _, box := range boxes {
		x1 := preprocess.Clamp((box.X1-lb.PadX)/lb.Scale, 0, w)
		y1 := preprocess.Clamp((box.Y1-lb.PadY)/lb.Scale, 0, h)
		x2 := preprocess.Clamp((box.X2-lb.PadX)/lb.Scale, 0, w)
		y2 := preprocess.Clamp((box.Y2-lb.PadY)/lb.Scale, 0, h)
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
		e.session.Destroy()
	}
	if e.input != nil {
		e.input.Destroy()
	}
	if e.output != nil {
		e.output.Destroy()
	}
	return nil
}

// parseOutputShape 将模型输出形状解析为 anchors/attrs/是否转置。
func parseOutputShape(dims ort.Shape, nc int) (anchors, attrs int, transposed bool, err error) {
	if len(dims) != 3 || dims[0] != 1 {
		return 0, 0, false, fmt.Errorf("unsupported output shape %v, expect [1, ..., ...]", dims)
	}
	expected := 4 + nc
	d1, d2 := int(dims[1]), int(dims[2])
	if d1 < 0 || d2 < 0 {
		return 0, 0, false, fmt.Errorf("dynamic output dim %v not supported; re-export with fixed shape", dims)
	}
	switch {
	case d1 == expected:
		return d2, expected, false, nil
	case d2 == expected:
		return d1, expected, true, nil
	default:
		return 0, 0, false, fmt.Errorf("output shape %v does not match 4+%d classes", dims, nc)
	}
}
