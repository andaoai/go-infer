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

// Stats 是调度器某一时刻的可观测快照。
type Stats struct {
	Workers      int    `json:"workers"`       // worker 数
	QueueLen     int    `json:"queue_len"`     // 排队中请求数
	QueueCap     int    `json:"queue_cap"`     // 队列容量
	InFlight     int64  `json:"in_flight"`     // 已被 worker 取走、尚未返回结果的请求数
	MaxBatch     int    `json:"max_batch"`     // 实际生效的合批上限
	Accepted     uint64 `json:"accepted"`      // 累计成功入队
	RejectedBusy uint64 `json:"rejected_busy"` // 累计因队列满返回 ErrBusy
	Succeeded    uint64 `json:"succeeded"`     // 累计成功返回结果
	Failed       uint64 `json:"failed"`        // 累计返回错误（含客户端取消）
	Panics       uint64 `json:"panics"`        // 累计 RunBatch panic（已恢复）
	Batches      uint64 `json:"batches"`       // 累计执行的 RunBatch 次数
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

	accepted     atomic.Uint64
	rejectedBusy atomic.Uint64
	succeeded    atomic.Uint64
	failed       atomic.Uint64
	panics       atomic.Uint64
	batches      atomic.Uint64
	inFlight     atomic.Int64

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
func (pj *preparedJob) canceled() bool                       { return pj.job.ctx.Err() != nil }

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
		s.wgWorker.Add(workers)
		for i := 0; i < workers; i++ {
			go s.runBatchWorker()
		}
	} else {
		s.maxBatch = 1
		s.wgWorker.Add(workers)
		for i := 0; i < workers; i++ {
			go s.runPlainWorker()
		}
	}
	s.accepting.Store(true)
	return s
}

// Name/Task/Framework 透传给被包装引擎。
func (s *Scheduler) Name() string      { return s.inner.Name() }
func (s *Scheduler) Task() engine.Task { return s.inner.Task() }
func (s *Scheduler) Framework() string { return s.inner.Framework() }

// Stats 返回调度器的瞬时计数快照（并发安全，不加锁）。返回 any 以让上层
// （如 api 层）通过鸭子接口识别，而无需反向依赖 sched 包。
func (s *Scheduler) Stats() any { return s.Snapshot() }

// Snapshot 返回带类型的计数快照，供本包测试与内部调用方使用。
func (s *Scheduler) Snapshot() Stats {
	return Stats{
		Workers:      s.workers,
		QueueLen:     len(s.jobs),
		QueueCap:     cap(s.jobs),
		InFlight:     s.inFlight.Load(),
		MaxBatch:     s.maxBatch,
		Accepted:     s.accepted.Load(),
		RejectedBusy: s.rejectedBusy.Load(),
		Succeeded:    s.succeeded.Load(),
		Failed:       s.failed.Load(),
		Panics:       s.panics.Load(),
		Batches:      s.batches.Load(),
	}
}

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
		s.accepted.Add(1)
	case <-s.done:
		return nil, engine.ErrClosed
	default:
		s.rejectedBusy.Add(1)
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

// runBatchWorker 从 jobs 取请求、在设备锁之外并发 Prepare，再投递给 batcher。
func (s *Scheduler) runBatchWorker() {
	defer s.wgWorker.Done()
	for {
		var j *job
		select {
		case <-s.done:
			return
		case j = <-s.jobs:
		}
		s.inFlight.Add(1)
		if s.stopped() {
			// 关闭期间只做必要的快速失败，不再浪费 CPU 预处理。
			s.finishJob(j, nil, engine.ErrClosed)
			continue
		}
		if err := j.ctx.Err(); err != nil {
			s.finishJob(j, nil, err)
			continue
		}
		p, err := s.batch.Prepare(j.ctx, j.req)
		if err != nil {
			s.finishJob(j, nil, err)
			continue
		}
		pj := &preparedJob{job: j, p: p}
		select {
		case s.prepared <- pj:
			// 移交 batcher：由 batcher 在交付结果时 finishJob 并 Release。
		case <-s.done:
			s.batch.Release(p)
			s.finishJob(j, nil, engine.ErrClosed)
		}
	}
}

// runPlainWorker 服务不实现 BatchEngine 的引擎：直接并发调用 Run。
func (s *Scheduler) runPlainWorker() {
	defer s.wgWorker.Done()
	for {
		var j *job
		select {
		case <-s.done:
			return
		case j = <-s.jobs:
		}
		s.inFlight.Add(1)
		if s.stopped() {
			s.finishJob(j, nil, engine.ErrClosed)
			continue
		}
		res, err := s.inner.Run(j.ctx, j.req)
		s.finishJob(j, res, err)
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

// runBatcher 是唯一的 RunBatch 调用者：攒批 -> 推理 -> 路由结果。
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
		// 首项可能在排队期间已被取消。
		if pj.canceled() {
			s.batch.Release(pj.p)
			s.finishPrepared(pj, nil, pj.job.ctx.Err())
			continue
		}
		batch := s.collectBatch(pj)
		s.runAndDeliver(batch)
	}
}

// collectBatch 从首个已就绪项开始，在 MaxWait 预算内继续凑批直到满批/超时/关闭。
// maxBatch<=1 时立即返回，不等待（消除固定 batch 模型的额外延迟）。
func (s *Scheduler) collectBatch(first *preparedJob) []*preparedJob {
	batch := []*preparedJob{first}
	if s.maxBatch <= 1 {
		return batch
	}
	timer := time.NewTimer(s.maxWait)
	defer timer.Stop()
	for len(batch) < s.maxBatch {
		select {
		case pj, ok := <-s.prepared:
			if !ok {
				return batch
			}
			if pj.canceled() {
				s.batch.Release(pj.p)
				s.finishPrepared(pj, nil, pj.job.ctx.Err())
				continue
			}
			batch = append(batch, pj)
		case <-timer.C:
			return batch
		case <-s.done:
			return batch
		}
	}
	return batch
}

// runAndDeliver 执行一批推理并把结果路由到各请求；任何路径都释放全部 Prepared。
func (s *Scheduler) runAndDeliver(batch []*preparedJob) {
	// 二次过滤：从入队到拼装之间可能有请求被取消。
	live := batch[:0]
	for _, pj := range batch {
		if pj.canceled() {
			s.batch.Release(pj.p)
			s.finishPrepared(pj, nil, pj.job.ctx.Err())
			continue
		}
		live = append(live, pj)
	}
	batch = live
	if len(batch) == 0 {
		return
	}

	var results []engine.Result
	var err error
	func() {
		// Release 始终执行；panic 转成整批错误，batcher 不崩。
		defer func() {
			for _, pj := range batch {
				s.batch.Release(pj.p)
			}
			if r := recover(); r != nil {
				s.panics.Add(1)
				err = fmt.Errorf("engine panic: %v", r)
			}
		}()
		s.batches.Add(1)
		// 单个客户端断开不能中止共享批；ORT Run 本身也不可打断。
		ctx := context.WithoutCancel(batch[0].job.ctx)
		inputs := make([]*engine.Prepared, len(batch))
		for i, pj := range batch {
			inputs[i] = pj.p
		}
		results, err = s.batch.RunBatch(ctx, inputs)
	}()

	if err != nil {
		for _, pj := range batch {
			s.finishPrepared(pj, nil, err)
		}
		return
	}
	if len(results) != len(batch) {
		err := fmt.Errorf("engine returned %d results for a batch of %d", len(results), len(batch))
		for _, pj := range batch {
			s.finishPrepared(pj, nil, err)
		}
		return
	}
	for i, pj := range batch {
		s.finishPrepared(pj, results[i], nil)
	}
}

// finishJob 交付单个 job 的结果并更新计数/inFlight。
func (s *Scheduler) finishJob(j *job, res engine.Result, err error) {
	j.resolve(res, err)
	s.inFlight.Add(-1)
	if err != nil {
		s.failed.Add(1)
	} else {
		s.succeeded.Add(1)
	}
}

func (s *Scheduler) finishPrepared(pj *preparedJob, res engine.Result, err error) {
	s.finishJob(pj.job, res, err)
}

// drainJobs 把 jobs 通道中剩余的请求全部以 ErrClosed 收尾（worker 已退出后调用）。
func (s *Scheduler) drainJobs() {
	for {
		select {
		case j := <-s.jobs:
			j.resolve(nil, engine.ErrClosed)
		default:
			return
		}
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
		s.drainJobs()
		if s.batch != nil {
			// prepared 不再有生产者；关闭后 batcher 排空剩余项后退出。
			close(s.prepared)
			s.wgBatch.Wait()
		}
	})
	return nil
}
