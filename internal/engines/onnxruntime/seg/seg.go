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
	"sync"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/ortbatch"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/yolohead"
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

// segView 是一次 RunBatch 的输出视图：detBuf/protoBuf 是同一块池化缓冲
// 的两段切片，postSlot 在设备锁之外同步读取；Done 后整块回池。
type segView struct {
	detBuf      []float32 // 前 n*detStride 个元素
	protoBuf    []float32 // 紧接其后 n*protoStride 个元素
	detStride   int
	protoStride int
}

// maskScratch 是 makeMasks/contour 复用的掩膜与连通域标记缓冲，按输入面积
// 一次性分配，经 sync.Pool 在并发的 postSlot 之间复用。
type maskScratch struct {
	mask   []bool
	labels []int
}

// Engine 是 engine.Engine / engine.BatchEngine 的 ONNX Runtime YOLOv8-seg 实现。
// 输入缓冲、session、pool 与设备锁由内嵌的 *ortbatch.Runtime 管理；本结构只保留
// 检测头/原型输出形状、池化输出缓冲与掩膜临时缓冲。
type Engine struct {
	*ortbatch.Runtime
	cfg Config

	outPool     sync.Pool // 借出 []float32，len = MaxBatch()*(detStride+protoStride)
	scratchPool sync.Pool // 借出 *maskScratch，mask/labels 容量为 InputW*InputH

	detStride   int // anchors*attrs，每图检测头元素数
	protoStride int // nm*mh*mw，每图原型元素数

	// 检测头布局
	anchors    int
	attrs      int // 4 + nc + nm
	nc         int
	nm         int
	transposed bool
	// 原型掩膜
	maskH, maskW int
	// maskLogitThr = -log(1/thr - 1)；掩膜 logit 和 >= 它等价于 sigmoid(sum) >= thr。
	maskLogitThr float32
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
	e := &Engine{
		Runtime:      rt,
		cfg:          cfg,
		detStride:    detStride,
		protoStride:  protoStride,
		anchors:      anchors,
		attrs:        attrs,
		nc:           nc,
		nm:           nm,
		transposed:   transposed,
		maskH:        mh,
		maskW:        mw,
		maskLogitThr: logitFromProb(cfg.MaskThresh),
	}
	e.outPool.New = func() any {
		return make([]float32, rt.MaxBatch()*(detStride+protoStride))
	}
	area := cfg.InputW * cfg.InputH
	e.scratchPool.New = func() any {
		return &maskScratch{
			mask:   make([]bool, area),
			labels: make([]int, area),
		}
	}
	return e, nil
}

// logitFromProb 计算 logit(p)=log(p/(1-p))=-log(1/p-1)，对 p 做 [1e-4,1-1e-4]
// 保护，避免用户传 0/1 产生 ±Inf 让掩膜全删/全保。
func logitFromProb(p float32) float32 {
	const eps = 1e-4
	if p < eps {
		p = eps
	}
	if p > 1-eps {
		p = 1 - eps
	}
	return float32(-math.Log(float64(1.0/p - 1.0)))
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

// buildOutput 从池中借出一整块输出缓冲，前 n*detStride 给检测头、紧接其后的
// n*protoStride 给原型，分别建两个张量别名这两段；Done 销毁两个张量并整块回池。
func (e *Engine) buildOutput(n int) (ortbatch.Output, any, error) {
	full := e.outPool.Get().([]float32)
	detBuf := full[:n*e.detStride]
	protoBuf := full[n*e.detStride : n*(e.detStride+e.protoStride)]

	var detShape ort.Shape
	if e.transposed {
		detShape = ort.NewShape(int64(n), int64(e.anchors), int64(e.attrs))
	} else {
		detShape = ort.NewShape(int64(n), int64(e.attrs), int64(e.anchors))
	}
	detOut, err := ort.NewTensor(detShape, detBuf)
	if err != nil {
		e.outPool.Put(full)
		return ortbatch.Output{}, nil, fmt.Errorf("create det tensor: %w", err)
	}
	protoTensor, err := ort.NewTensor(
		ort.NewShape(int64(n), int64(e.nm), int64(e.maskH), int64(e.maskW)),
		protoBuf,
	)
	if err != nil {
		detOut.Destroy()
		e.outPool.Put(full)
		return ortbatch.Output{}, nil, fmt.Errorf("create proto tensor: %w", err)
	}
	return ortbatch.Output{
			Values: []ort.Value{detOut, protoTensor},
			Done: func() {
				detOut.Destroy()
				protoTensor.Destroy()
				e.outPool.Put(full)
			},
		}, &segView{
			detBuf:      detBuf,
			protoBuf:    protoBuf,
			detStride:   e.detStride,
			protoStride: e.protoStride,
		}, nil
}

// postSlot 对第 slot 个输出解码、NMS、生成掩膜并映射回原图。
func (e *Engine) postSlot(slot int, p *engine.Prepared, view any, elapsed time.Duration) (engine.Result, error) {
	m := p.Meta.(*ortbatch.Meta)
	v := view.(*segView)
	cands := e.decodeDetections(v, slot, m.Conf)
	kept := e.nms(cands)
	instances := e.makeMasks(v, slot, kept, m)
	return &engine.SegmentationResult{Elapsed: elapsed, Instances: instances}, nil
}

type cand struct {
	box    engine.Box
	coeffs []float32 // 长度 nm；从检测头复制一份，供 makeMasks 在锁外使用
}

// decodeDetections 解析第 slot 个检测头：cx cy w h + nc 类别分 + nm 掩膜系数。
// 输出头遍历统一走 yolohead.Walk，coeffOff 指向本 anchor 掩膜系数的起始偏移。
func (e *Engine) decodeDetections(v *segView, slot int, conf float32) []cand {
	nc, nm := e.nc, e.nm
	detBuf := v.detBuf
	out := make([]cand, 0, 128)
	yolohead.Walk(detBuf, slot, v.detStride, e.anchors, e.attrs, nc, e.transposed, conf,
		func(_, cls int, cx, cy, w, h, score float32, coeffs []float32) {
			cf := make([]float32, nm)
			copy(cf, coeffs)
			out = append(out, cand{
				box: engine.Box{
					ClassID: cls, Confidence: score,
					X1: cx - w/2, Y1: cy - h/2, X2: cx + w/2, Y2: cy + h/2,
				},
				coeffs: cf,
			})
		})
	return out
}

// nms 对携带掩膜系数的候选做按类 NMS，统一走 postprocess.NMSIndexed
// （内部用 slices.SortStableFunc 排索引），删除手写插入排序。
func (e *Engine) nms(cands []cand) []cand {
	keep := postprocess.NMSIndexed(len(cands),
		func(i int) float32 { return cands[i].box.Confidence },
		func(i, j int) bool { return cands[i].box.ClassID == cands[j].box.ClassID },
		func(i, j int) float32 { return postprocess.IoU(cands[i].box, cands[j].box) },
		e.cfg.IoUThresh,
	)
	out := make([]cand, 0, len(keep))
	for _, i := range keep {
		out = append(out, cands[i])
	}
	return out
}

// makeMasks 对每个保留实例计算二值掩膜、追踪轮廓并映射回原图。
//
// 优化：
//   - 用 maskLogitThr 直接比较 logit 和，删除逐像素 sigmoid/math.Exp；
//   - 对框网格预先一次性算出每列/每行的 x0,x1,dx 与 y0,y1,dy（floor/clamp
//     只做一次），nm 内层循环只做 4 次 protoBuf 取值与乘加；
//   - 掩膜 mask 与连通域 labels 缓冲从 scratchPool 复用。
func (e *Engine) makeMasks(v *segView, slot int, cands []cand, m *ortbatch.Meta) []engine.Instance {
	origW, origH := float32(m.Orig.Dx()), float32(m.Orig.Dy())
	inW, inH := float32(e.cfg.InputW), float32(e.cfg.InputH)
	mW, mH := float32(e.maskW), float32(e.maskH)
	sx, sy := mW/inW, mH/inH

	sc := e.scratchPool.Get().(*maskScratch)
	defer e.scratchPool.Put(sc)

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
		n := bw * bh
		if n > len(sc.mask) {
			// 理论上不会超过输入面积；防御性扩容。
			sc.mask = make([]bool, n)
			sc.labels = make([]int, n)
		}
		mask := sc.mask[:n]
		labels := sc.labels[:n]
		for i := range mask {
			mask[i] = false
		}

		// 预计算每列 x 网格的 floor/clamp/权重（px 维）。
		type xg struct {
			x0, x1 int
			dx     float32
		}
		xs := make([]xg, bw)
		for px := 0; px < bw; px++ {
			mx := (float32(px) + 0.5 + ix1) * sx
			x0 := int(math.Floor(float64(mx)))
			x1 := x0 + 1
			if x0 < 0 {
				x0 = 0
			}
			if x0 >= e.maskW {
				x0 = e.maskW - 1
			}
			if x1 >= e.maskW {
				x1 = e.maskW - 1
			}
			xs[px] = xg{x0: x0, x1: x1, dx: mx - float32(x0)}
		}
		// 预计算每行 y 网格（py 维）。
		type yg struct {
			y0, y1 int
			dy     float32
		}
		ys := make([]yg, bh)
		for py := 0; py < bh; py++ {
			my := (float32(py) + 0.5 + iy1) * sy
			y0 := int(math.Floor(float64(my)))
			y1 := y0 + 1
			if y0 < 0 {
				y0 = 0
			}
			if y0 >= e.maskH {
				y0 = e.maskH - 1
			}
			if y1 >= e.maskH {
				y1 = e.maskH - 1
			}
			ys[py] = yg{y0: y0, y1: y1, dy: my - float32(y0)}
		}

		batchOff := slot * v.protoStride
		plane := e.maskW * e.maskH
		coeffs := c.coeffs
		logitThr := e.maskLogitThr
		for py := 0; py < bh; py++ {
			yy := ys[py]
			y0w, y1w := yy.y0*e.maskW, yy.y1*e.maskW
			dy1 := 1 - yy.dy
			row := py * bw
			for px := 0; px < bw; px++ {
				xx := xs[px]
				dx1 := 1 - xx.dx
				w00 := dx1 * dy1
				w10 := xx.dx * dy1
				w01 := dx1 * yy.dy
				w11 := xx.dx * yy.dy
				var sum float32
				for k := 0; k < e.nm; k++ {
					base := batchOff + k*plane
					ck := coeffs[k]
					sum += ck * (w00*v.protoBuf[base+y0w+xx.x0] +
						w10*v.protoBuf[base+y0w+xx.x1] +
						w01*v.protoBuf[base+y1w+xx.x0] +
						w11*v.protoBuf[base+y1w+xx.x1])
				}
				if sum >= logitThr {
					mask[row+px] = true
				}
			}
		}

		polys := contour(mask, labels, bw, bh)
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
