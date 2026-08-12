// Package detect 实现基于 ONNX Runtime 的 YOLO 检测引擎。
//
// 适配 ultralytics 导出的 ONNX 检测模型：
//   - 输入:  float32[N,3,H,W]，NCHW，像素归一化 0~1（letterbox 填充 114）
//   - 输出:  官方 [N,4+nc,anchors] 或转置 [N,anchors,4+nc]
//
// N 由模型输入 batch 维决定：固定 1 则只能单图；动态维则可在配置上限内
// 合批。预处理与批处理运行时由内嵌的 ortbatch.Runtime 提供，本包只负责输出
// 形状解析、输出张量构造与逐槽后处理。
package detect

import (
	"context"
	"fmt"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/ortbatch"
	"github.com/andaoai/go-infer/internal/postprocess"
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

// Engine 是 engine.Engine / engine.BatchEngine 的 ONNX Runtime YOLO 实现。
// 输入缓冲、session、pool 与设备锁由内嵌的 *ortbatch.Runtime 管理；本结构只保留
// 输出形状与输出缓冲。
type Engine struct {
	*ortbatch.Runtime
	cfg Config

	outputBuf     []float32 // maxBatch*outStride
	outStride     int       // anchors*attrs，每图输出元素数
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

	rt, err := ortbatch.New(ortbatch.Config{
		ModelPath:   cfg.ModelPath,
		InputNames:  []string{inInfo.Name},
		OutputNames: []string{outInfo.Name},
		InputDims:   inInfo.Dimensions,
		W:           cfg.InputW,
		H:           cfg.InputH,
		MaxBatch:    cfg.MaxBatch,
		DefaultConf: cfg.ConfThresh,
	})
	if err != nil {
		return nil, err
	}

	outStride := anchors * attrs
	return &Engine{
		Runtime:       rt,
		cfg:           cfg,
		outputBuf:     make([]float32, rt.MaxBatch()*outStride),
		outStride:     outStride,
		outAnchors:    anchors,
		outAttrs:      attrs,
		outTransposed: transposed,
	}, nil
}

func (e *Engine) Name() string      { return e.cfg.Name }
func (e *Engine) Task() engine.Task { return engine.TaskDetection }
func (e *Engine) Framework() string { return "onnxruntime" }

// Run 是单图便捷路径。
func (e *Engine) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	return e.Runtime.RunOnce(ctx, req, e.buildOutput, e.postSlot)
}

// RunBatch 在一次底层推理中处理 n 张已预处理的图。
func (e *Engine) RunBatch(ctx context.Context, batch []*engine.Prepared) ([]engine.Result, error) {
	return e.Runtime.RunBatch(ctx, batch, e.buildOutput, e.postSlot)
}

// buildOutput 在引擎自有 outputBuf 上按本次 batch n 构造输出张量。
func (e *Engine) buildOutput(n int) (ortbatch.Output, error) {
	var shape ort.Shape
	if e.outTransposed {
		shape = ort.NewShape(int64(n), int64(e.outAnchors), int64(e.outAttrs))
	} else {
		shape = ort.NewShape(int64(n), int64(e.outAttrs), int64(e.outAnchors))
	}
	out, err := ort.NewTensor(shape, e.outputBuf[:n*e.outStride])
	if err != nil {
		return ortbatch.Output{}, fmt.Errorf("create output tensor: %w", err)
	}
	return ortbatch.Output{
		Values: []ort.Value{out},
		Done:   func() { out.Destroy() },
	}, nil
}

// postSlot 对第 slot 个输出做解码、NMS 与坐标映射。
func (e *Engine) postSlot(slot int, p *engine.Prepared, elapsed time.Duration) (engine.Result, error) {
	m := p.Meta.(*ortbatch.Meta)
	boxes := e.postprocess(slot, m.Conf)
	detections := e.mapToImage(boxes, m)
	return &engine.DetectionResult{Elapsed: elapsed, Detections: detections}, nil
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
	return postprocess.NMS(cands, e.cfg.IoUThresh)
}

// mapToImage 将 letterbox 坐标映射回原图并转为公共 Detection。
func (e *Engine) mapToImage(boxes []engine.Box, m *ortbatch.Meta) []engine.Detection {
	w, h := float32(m.Orig.Dx()), float32(m.Orig.Dy())
	out := make([]engine.Detection, 0, len(boxes))
	for _, box := range boxes {
		x1 := preprocess.Clamp((box.X1-m.PadX)/m.Scale, 0, w)
		y1 := preprocess.Clamp((box.Y1-m.PadY)/m.Scale, 0, h)
		x2 := preprocess.Clamp((box.X2-m.PadX)/m.Scale, 0, w)
		y2 := preprocess.Clamp((box.Y2-m.PadY)/m.Scale, 0, h)
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
