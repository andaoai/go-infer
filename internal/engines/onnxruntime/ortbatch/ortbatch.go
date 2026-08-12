// Package ortbatch 提供 ONNX Runtime 动态 batch 推理的共享运行时：session 生命周期、
// 输入批缓冲、per-request 缓冲池、串行设备锁，以及 "Prepare 拷贝输入 -> 建输出张量 ->
// session.Run -> 逐槽后处理" 的统一骨架。
//
// detect/seg 等具体引擎内嵌 *Runtime，只需提供输出形状解析、自有的输出缓冲，以及两个
// 回调：BuildOutput（按本次 batch n 在自有输出缓冲上建张量）和 PostSlot（逐槽解码）。
// 这避免了每个 YOLO 头重复抄写约 150 行相同的加锁/copy/张量/计时样板。
package ortbatch

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

// Meta 是每个预处理项的引擎私有元信息（letterbox 缩放/填充、原图边界、conf）。
// 所有 ONNX 视觉引擎共用这一份标量集合。
type Meta struct {
	Scale float32
	PadX  float32
	PadY  float32
	Orig  image.Rectangle
	Conf  float32
}

// Output 是一次 RunBatch 的输出张量集合：Values 传给 session.Run，Done 在推理
// 返回后销毁这些张量（输出缓冲本身归引擎所有，Done 不释放底层 []float32）。
type Output struct {
	Values []ort.Value
	Done   func()
}

// BuildOutput 由引擎提供：基于本次 batch n，在引擎自有的输出缓冲切片上构造具体
// 形状的输出张量。返回的 Output.Values 顺序必须与 Config.OutputNames 一致。
type BuildOutput func(n int) (Output, error)

// PostSlot 由引擎提供：对第 slot 个输出做引擎私有的后处理（解码/NMS/坐标映射/掩膜），
// 返回该槽的结果。p.Meta 的动态类型为 *Meta。
type PostSlot func(slot int, p *engine.Prepared, elapsed time.Duration) (engine.Result, error)

// Config 构造共享运行时。InputDims 仅用于解析 batch 上限（dim0）。
type Config struct {
	ModelPath   string
	InputNames  []string
	OutputNames []string
	InputDims   ort.Shape
	W, H        int
	MaxBatch    int // 仅动态 batch 维模型生效；固定 batch 模型忽略
	DefaultConf float32
}

// Runtime 管理一个动态 batch ONNX session 的共享资源。零值不可用，用 New 构造。
type Runtime struct {
	session *ort.DynamicAdvancedSession

	maxBatch    int
	w, h        int
	planeSize   int // 3*H*W，每图输入元素数
	defaultConf float32

	inputBuf []float32  // maxBatch*planeSize，C 输入张量指向这里
	pool     sync.Pool  // 借出 []float32，len=planeSize
	runMu    sync.Mutex // 共享 inputBuf 与 session，RunBatch 串行
}

// New 创建 session 并按模型 batch 维分配输入缓冲与 per-request 池。
func New(cfg Config) (*Runtime, error) {
	if cfg.W <= 0 || cfg.H <= 0 {
		return nil, fmt.Errorf("input size must be positive, got %dx%d", cfg.W, cfg.H)
	}
	maxBatch, err := ortutil.ResolveMaxBatch(cfg.InputDims, cfg.MaxBatch)
	if err != nil {
		return nil, err
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewDynamicAdvancedSession(cfg.ModelPath, cfg.InputNames, cfg.OutputNames, options)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	planeSize := 3 * cfg.H * cfg.W
	r := &Runtime{
		session:     session,
		maxBatch:    maxBatch,
		w:           cfg.W,
		h:           cfg.H,
		planeSize:   planeSize,
		defaultConf: cfg.DefaultConf,
		inputBuf:    make([]float32, maxBatch*planeSize),
	}
	r.pool.New = func() any { return make([]float32, planeSize) }
	return r, nil
}

// MaxBatch 返回一次 RunBatch 能接受的最大图片数。
func (r *Runtime) MaxBatch() int { return r.maxBatch }

// Prepare 在设备锁之外完成一张图的 letterbox/归一化，借出缓冲由 Release 归还。
func (r *Runtime) Prepare(ctx context.Context, req *engine.Request) (*engine.Prepared, error) {
	if req.Image == nil {
		return nil, fmt.Errorf("inference requires an image")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	conf := r.defaultConf
	if v, ok := req.Params["conf"]; ok && v > 0 {
		conf = v
	}
	buf := r.pool.Get().([]float32)
	lb := preprocess.Letterbox(req.Image, r.w, r.h)
	preprocess.FillNCHW(buf, lb.RGBA, r.w, r.h)
	return &engine.Prepared{
		Input: buf,
		Meta: &Meta{
			Scale: lb.Scale,
			PadX:  lb.PadX,
			PadY:  lb.PadY,
			Orig:  req.Image.Bounds(),
			Conf:  conf,
		},
	}, nil
}

// Release 归还 Prepare 借出的缓冲，对 nil 安全、可重复调用。
func (r *Runtime) Release(p *engine.Prepared) {
	if p == nil || p.Input == nil {
		return
	}
	r.pool.Put(p.Input)
	p.Input = nil
}

// RunOnce 是单图便捷路径：Prepare + RunBatch(1) + Release。
func (r *Runtime) RunOnce(ctx context.Context, req *engine.Request, buildOut BuildOutput, post PostSlot) (engine.Result, error) {
	p, err := r.Prepare(ctx, req)
	if err != nil {
		return nil, err
	}
	defer r.Release(p)
	results, err := r.RunBatch(ctx, []*engine.Prepared{p}, buildOut, post)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// RunBatch 在一次底层推理中处理 n 张已预处理的图：加锁 -> copy 输入 -> 建输入/输出
// 张量 -> session.Run -> 逐槽 post。整批成功或整批失败；n∈[1,MaxBatch]。
func (r *Runtime) RunBatch(ctx context.Context, batch []*engine.Prepared, buildOut BuildOutput, post PostSlot) ([]engine.Result, error) {
	n := len(batch)
	if n == 0 || n > r.maxBatch {
		return nil, fmt.Errorf("invalid batch size %d (max %d)", n, r.maxBatch)
	}

	r.runMu.Lock()
	defer r.runMu.Unlock()

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// 拷贝各图输入到批缓冲。C 张量指向批缓冲，而非 per-request 缓冲，
	// 因此 RunBatch 返回后调度器即可安全归还 per-request 缓冲。
	for i, p := range batch {
		copy(r.inputBuf[i*r.planeSize:(i+1)*r.planeSize], p.Input)
	}

	in, err := ort.NewTensor(
		ort.NewShape(int64(n), 3, int64(r.h), int64(r.w)),
		r.inputBuf[:n*r.planeSize],
	)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer in.Destroy()

	out, err := buildOut(n)
	if err != nil {
		return nil, err
	}
	if out.Done != nil {
		defer out.Done()
	}

	start := time.Now()
	if err := r.session.Run([]ort.Value{in}, out.Values); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}
	elapsed := time.Since(start)

	results := make([]engine.Result, n)
	for i, p := range batch {
		results[i], err = post(i, p, elapsed)
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

// Close 销毁底层 session。
func (r *Runtime) Close() error {
	if r.session != nil {
		return r.session.Destroy()
	}
	return nil
}
