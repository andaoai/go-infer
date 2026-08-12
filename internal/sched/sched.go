// Package sched 是引擎之上的并发调度层：有界 worker pool + 背压、CPU 预处理与
// 设备推理的流水线重叠、以及延迟预算内的 dynamic batching。
//
// Scheduler 本身实现 engine.Engine，作为装饰器包在真实引擎外面注册，HTTP/采集/
// 可视化链路无需感知。若被包装引擎实现 engine.BatchEngine，则走
//
//	入队 -> N worker 并发 Prepare(CPU) -> batcher 攒批 -> 一次 RunBatch(设备)
//
// 的流水线；否则退化为 N worker 直接调用 Run，仍提供有界队列与背压。
package sched

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
)

// Config 控制调度器行为。零值字段会在 New 中填充默认值。
type Config struct {
	// Workers 是并发预处理（batch 路径）或并发推理（回退路径）的 worker 数。
	// <=0 时取 runtime.NumCPU()。
	Workers int
	// QueueDepth 是入队请求的缓冲上限；满时 Run 立即返回 engine.ErrBusy。<=0 → 64。
	QueueDepth int
	// MaxBatch 是一次 RunBatch 的图片数上限，0 表示用引擎自身 MaxBatch()。
	// 固定 batch=1 的模型会忽略此值（只能为 1）。
	MaxBatch int
	// MaxWait 是为凑满一批额外等待的时间；0 → 5ms。MaxBatch==1 时忽略（立即刷批）。
	MaxWait time.Duration
}

// Scheduler 包装一个 engine.Engine，提供并发调度。
type Scheduler struct {
	inner engine.Engine
	batch engine.BatchEngine // 非 nil 时走流水线

	workers  int
	maxBatch int
	maxWait  time.Duration

	jobs     chan *job
	prepared chan *preparedJob // 仅 batch 路径使用，缓冲给流水线重叠用
	done     chan struct{}     // Close 时关闭
	wgWorker sync.WaitGroup
	wgBatch  sync.WaitGroup

	accepting atomic.Bool
	stopOnce  sync.Once
}

type resultOrErr struct {
	res engine.Result
	err error
}

type job struct {
	ctx context.Context
	req *engine.Request
	fut chan resultOrErr // cap 1，resolve 永不阻塞
}

func (j *job) resolve(res engine.Result, err error) {
	select {
	case j.fut <- resultOrErr{res: res, err: err}:
	default: // 调用方已离开，结果丢弃
	}
}

// preparedJob 把一个已预处理项与其请求 future 绑定，供 batcher 路由结果。
type preparedJob struct {
	job *job
	p   *engine.Prepared
}

func (pj *preparedJob) resolve(res engine.Result, err error) { pj.job.resolve(res, err) }

// New 构造调度器并启动 worker/batcher。调用方需在关闭时调用 Close。
func New(inner engine.Engine, cfg Config) *Scheduler {
	if inner == nil {
		panic("sched: nil engine")
	}
	workers := cfg.Workers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	queue := cfg.QueueDepth
	if queue <= 0 {
		queue = 64
	}
	maxWait := cfg.MaxWait
	if maxWait <= 0 {
		maxWait = 5 * time.Millisecond
	}

	s := &Scheduler{
		inner:   inner,
		workers: workers,
		maxWait: maxWait,
		jobs:    make(chan *job, queue),
		done:    make(chan struct{}),
	}
	if be, ok := inner.(engine.BatchEngine); ok {
		s.batch = be
		s.maxBatch = be.MaxBatch()
		if cfg.MaxBatch > 0 && cfg.MaxBatch < s.maxBatch {
			s.maxBatch = cfg.MaxBatch
		}
		if s.maxBatch < 1 {
			s.maxBatch = 1
		}
		// 关键：prepared 必须有缓冲。batcher 卡在 RunBatch(设备) 期间，
		// worker 仍能持续 Prepare 并投递，从而实现 CPU/设备重叠。
		s.prepared = make(chan *preparedJob, workers+s.maxBatch)
		s.wgBatch.Add(1)
		go s.runBatcher()
	}
	s.wgWorker.Add(workers)
	for i := 0; i < workers; i++ {
		go s.runWorker()
	}
	s.accepting.Store(true)
	return s
}

// Name/Task/Framework 透传给被包装引擎。
func (s *Scheduler) Name() string      { return s.inner.Name() }
func (s *Scheduler) Task() engine.Task { return s.inner.Task() }
func (s *Scheduler) Framework() string { return s.inner.Framework() }

// Run 入队一次推理请求。队列满返回 engine.ErrBusy，调度器关闭返回 engine.ErrClosed，
// ctx 取消返回 ctx.Err。结果/错误来自被包装引擎。
func (s *Scheduler) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	if !s.accepting.Load() {
		return nil, engine.ErrClosed
	}
	if req == nil || req.Image == nil {
		return nil, fmt.Errorf("inference requires an image")
	}
	j := &job{ctx: ctx, req: req, fut: make(chan resultOrErr, 1)}
	select {
	case s.jobs <- j:
	case <-s.done:
		return nil, engine.ErrClosed
	default:
		return nil, engine.ErrBusy
	}
	select {
	case r := <-j.fut:
		return r.res, r.err
	case <-ctx.Done():
		// 仍在排队/预处理/推理中：future 是缓冲的，结果会被安全丢弃。
		return nil, ctx.Err()
	}
}

func (s *Scheduler) runWorker() {
	defer s.wgWorker.Done()
	if s.batch != nil {
		for {
			var j *job
			select {
			case <-s.done:
				return
			case j = <-s.jobs:
			}
			if s.stopped() {
				// 关闭期间只做必要的快速失败，不再浪费 CPU 预处理。
				j.resolve(nil, engine.ErrClosed)
				continue
			}
			if j.ctx.Err() != nil {
				j.resolve(nil, j.ctx.Err())
				continue
			}
			p, err := s.batch.Prepare(j.ctx, j.req)
			if err != nil {
				j.resolve(nil, err)
				continue
			}
			pj := &preparedJob{job: j, p: p}
			select {
			case s.prepared <- pj:
			case <-s.done:
				s.batch.Release(p)
				j.resolve(nil, engine.ErrClosed)
			}
		}
	}
	for {
		var j *job
		select {
		case <-s.done:
			return
		case j = <-s.jobs:
		}
		if s.stopped() {
			j.resolve(nil, engine.ErrClosed)
			continue
		}
		res, err := s.inner.Run(j.ctx, j.req)
		j.resolve(res, err)
	}
}

func (s *Scheduler) stopped() bool {
	select {
	case <-s.done:
		return true
	default:
		return false
	}
}

func (s *Scheduler) runBatcher() {
	defer s.wgBatch.Done()
	for {
		// 不在此 select done：关闭期间 buffered 的 prepared 必须排空并 Release，
		// 否则泄漏缓冲、future 无人 resolve。prepared 在 worker 全部退出后关闭，
		// 届时这里自然返回。
		pj, ok := <-s.prepared
		if !ok {
			return
		}
		first := pj
		// 首项可能在排队期间已被取消。
		if err := first.job.ctx.Err(); err != nil {
			s.batch.Release(first.p)
			first.resolve(nil, err)
			continue
		}
		batch := []*preparedJob{first}
		if s.maxBatch > 1 {
			timer := time.NewTimer(s.maxWait)
		collect:
			for len(batch) < s.maxBatch {
				select {
				case pj, ok := <-s.prepared:
					if !ok {
						timer.Stop()
						break collect
					}
					if pj.job.ctx.Err() != nil {
						s.batch.Release(pj.p)
						pj.resolve(nil, pj.job.ctx.Err())
						continue
					}
					batch = append(batch, pj)
				case <-timer.C:
					break collect
				case <-s.done:
					timer.Stop()
					break collect
				}
			}
			timer.Stop()
		}
		s.runAndDeliver(batch)
	}
}

// runAndDeliver 执行一批推理并把结果路由到各请求；任何路径都释放全部 Prepared。
func (s *Scheduler) runAndDeliver(batch []*preparedJob) {
	// 二次过滤：从入队到拼装之间可能有请求被取消。
	live := batch[:0]
	for _, pj := range batch {
		if err := pj.job.ctx.Err(); err != nil {
			s.batch.Release(pj.p)
			pj.resolve(nil, err)
			continue
		}
		live = append(live, pj)
	}
	batch = live
	if len(batch) == 0 {
		return
	}

	defer func() {
		for _, pj := range batch {
			s.batch.Release(pj.p)
		}
		if r := recover(); r != nil {
			err := fmt.Errorf("engine panic: %v", r)
			for _, pj := range batch {
				pj.resolve(nil, err)
			}
		}
	}()

	// 单个客户端断开不能中止共享批；ORT Run 本身也不可打断。
	ctx := context.WithoutCancel(batch[0].job.ctx)
	inputs := make([]*engine.Prepared, len(batch))
	for i, pj := range batch {
		inputs[i] = pj.p
	}
	results, err := s.batch.RunBatch(ctx, inputs)
	if err != nil {
		for _, pj := range batch {
			pj.resolve(nil, err)
		}
		return
	}
	if len(results) != len(batch) {
		err := fmt.Errorf("engine returned %d results for a batch of %d", len(results), len(batch))
		for _, pj := range batch {
			pj.resolve(nil, err)
		}
		return
	}
	for i, pj := range batch {
		pj.resolve(results[i], nil)
	}
}

// Close 停止接收新请求并等待 worker/batcher 退出。可重复调用。
// 注意：Close 只停调度器自身的 goroutine，不关闭被包装引擎（由外层按
// 先 Close 调度器、再 Close 引擎的顺序管理生命周期）。
func (s *Scheduler) Close() error {
	s.stopOnce.Do(func() {
		s.accepting.Store(false)
		close(s.done) // 通知所有 goroutine 停止；jobs 刻意不关闭，避免与 Run 的发送竞争
		s.wgWorker.Wait()
		// worker 都已退出，把队列里没被取走的请求以 ErrClosed 收尾，避免调用方永挂。
		for {
			select {
			case j := <-s.jobs:
				j.resolve(nil, engine.ErrClosed)
			default:
				goto drained
			}
		}
	drained:
		if s.batch != nil {
			// prepared 不再有生产者；关闭后 batcher 排空剩余项后退出。
			close(s.prepared)
			s.wgBatch.Wait()
		}
	})
	return nil
}
