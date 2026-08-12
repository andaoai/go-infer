package sched

import (
	"context"
	"errors"
	"image"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
)

// fakeBatch 是可控的 BatchEngine：Prepare/RunBatch 可注入延迟与错误，
// 统计调用次数并记录每次 RunBatch 的实际批大小。Input 不被并发复用
// （每个 Prepare 新建小切片），所以 Prepare 本身天然并发安全。
type fakeBatch struct {
	name       string
	maxBatch   int
	prepareDur time.Duration
	runDur     time.Duration
	failRun    error
	panicRun   bool

	prepareCalls atomic.Int64
	runCalls     atomic.Int64
	releaseCalls atomic.Int64

	mu        sync.Mutex
	batchSize []int

	runGate chan struct{} // 非 nil 时 RunBatch 阻塞到此通道关闭
	onRun   func()        // 可选：进入 RunBatch 后、阻塞前调用一次
}

func (f *fakeBatch) Name() string      { return f.name }
func (f *fakeBatch) Task() engine.Task { return engine.TaskDetection }
func (f *fakeBatch) Framework() string { return "fake" }
func (f *fakeBatch) MaxBatch() int     { return f.maxBatch }
func (f *fakeBatch) Close() error      { return nil }

// Run 满足 engine.Engine（调度器走的是 Batch 路径，不会调用它）。
func (f *fakeBatch) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	p, err := f.Prepare(ctx, req)
	if err != nil {
		return nil, err
	}
	defer f.Release(p)
	res, err := f.RunBatch(ctx, []*engine.Prepared{p})
	if err != nil {
		return nil, err
	}
	return res[0], nil
}

func (f *fakeBatch) Prepare(ctx context.Context, req *engine.Request) (*engine.Prepared, error) {
	f.prepareCalls.Add(1)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if f.prepareDur > 0 {
		select {
		case <-time.After(f.prepareDur):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	// 用 Text 携带调用方身份，回传到 Result 里便于路由校验。
	return &engine.Prepared{Input: make([]float32, 3), Meta: req.Text}, nil
}

func (f *fakeBatch) RunBatch(ctx context.Context, batch []*engine.Prepared) ([]engine.Result, error) {
	if f.onRun != nil {
		f.onRun()
	}
	if f.runGate != nil {
		select {
		case <-f.runGate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	f.runCalls.Add(1)
	f.mu.Lock()
	f.batchSize = append(f.batchSize, len(batch))
	f.mu.Unlock()
	if f.panicRun {
		panic("boom")
	}
	if f.runDur > 0 {
		select {
		case <-time.After(f.runDur):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if f.failRun != nil {
		return nil, f.failRun
	}
	out := make([]engine.Result, len(batch))
	for i, p := range batch {
		id, _ := p.Meta.(string)
		out[i] = &engine.DetectionResult{
			Detections: []engine.Detection{{ClassName: id}},
		}
	}
	return out, nil
}

func (f *fakeBatch) Release(p *engine.Prepared) {
	if p == nil {
		return
	}
	f.releaseCalls.Add(1)
}

func (f *fakeBatch) sizes() []int {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]int, len(f.batchSize))
	copy(out, f.batchSize)
	return out
}

// req 构造一个携带身份的合法请求。
func req(id string) *engine.Request {
	return &engine.Request{Image: image.NewRGBA(image.Rect(0, 0, 4, 4)), Text: id}
}

func runCtx(s *Scheduler, id string) (engine.Result, error) {
	return s.Run(context.Background(), req(id))
}

func TestResultsRoutedCorrectly(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 4}
	s := New(be, Config{Workers: 4, MaxWait: 10 * time.Millisecond})
	defer s.Close()

	const n = 50
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		id := string(rune('a' + i%26))
		go func() {
			defer wg.Done()
			res, err := runCtx(s, id)
			if err != nil {
				errs <- err
				return
			}
			dr := res.(*engine.DetectionResult)
			if len(dr.Detections) != 1 || dr.Detections[0].ClassName != id {
				errs <- errors.New("wrong result routed")
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

func TestBatchesNWithinWindow(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 4, runDur: 20 * time.Millisecond}
	s := New(be, Config{Workers: 8, MaxWait: 30 * time.Millisecond})
	defer s.Close()

	const n = 8
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			_, _ = runCtx(s, "x")
		}(i)
	}
	// 先让一批攒满 4 并进入 RunBatch，再发第二批，确保看到至少一次 len==4。
	time.Sleep(15 * time.Millisecond)
	wg.Wait()

	sizes := be.sizes()
	var sawFull bool
	for _, sz := range sizes {
		if sz == 4 {
			sawFull = true
		}
		if sz > 4 {
			t.Fatalf("batch exceeded MaxBatch: %d", sz)
		}
	}
	if !sawFull {
		t.Fatalf("expected at least one full batch of 4, got %v", sizes)
	}
}

func TestMaxWaitFlushesPartial(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 8}
	s := New(be, Config{Workers: 2, MaxWait: 20 * time.Millisecond})
	defer s.Close()

	start := time.Now()
	res, err := runCtx(s, "only")
	if err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d < 15*time.Millisecond {
		t.Fatalf("partial batch should wait MaxWait, returned in %v", d)
	}
	if res.(*engine.DetectionResult).Detections[0].ClassName != "only" {
		t.Fatal("wrong result")
	}
}

func TestNoWaitWhenMaxBatchOne(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 1, runDur: 5 * time.Millisecond}
	s := New(be, Config{Workers: 2, MaxWait: 50 * time.Millisecond})
	defer s.Close()

	start := time.Now()
	if _, err := runCtx(s, "x"); err != nil {
		t.Fatal(err)
	}
	// MaxBatch==1 必须跳过等待定时器，否则单请求平白多 50ms。
	if d := time.Since(start) - 5*time.Millisecond; d > 20*time.Millisecond {
		t.Fatalf("batch=1 should not incur MaxWait, took extra %v", d)
	}
}

func TestQueueFullReturnsErrBusy(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 1, prepareDur: 30 * time.Millisecond}
	// Workers=1：唯一 worker 卡在 Prepare；QueueDepth=1：队列再容 1 个。
	s := New(be, Config{Workers: 1, QueueDepth: 1})
	defer s.Close()

	// 占住 worker。
	go func() { _, _ = runCtx(s, "occupy") }()
	time.Sleep(10 * time.Millisecond)
	// 填满队列。
	go func() { _, _ = runCtx(s, "queued") }()
	time.Sleep(10 * time.Millisecond)
	// 第三次应立即 ErrBusy。
	if _, err := runCtx(s, "overflow"); !errors.Is(err, engine.ErrBusy) {
		t.Fatalf("want ErrBusy, got %v", err)
	}
}

func TestNonBatchFallback(t *testing.T) {
	plain := &plainEngine{runDur: 10 * time.Millisecond}
	s := New(plain, Config{Workers: 2, QueueDepth: 4})
	defer s.Close()

	res, err := s.Run(context.Background(), req("z"))
	if err != nil {
		t.Fatal(err)
	}
	if res.(*engine.DetectionResult).Detections[0].ClassName != "z" {
		t.Fatal("wrong result via fallback")
	}
	if got := plain.calls.Load(); got != 1 {
		t.Fatalf("want 1 inner Run, got %d", got)
	}
}

func TestNonBatchQueueFull(t *testing.T) {
	plain := &plainEngine{runDur: 40 * time.Millisecond}
	s := New(plain, Config{Workers: 1, QueueDepth: 1})
	defer s.Close()

	go func() { _, _ = runCtx(s, "a") }()
	time.Sleep(10 * time.Millisecond)
	go func() { _, _ = runCtx(s, "b") }()
	time.Sleep(10 * time.Millisecond)
	if _, err := runCtx(s, "c"); !errors.Is(err, engine.ErrBusy) {
		t.Fatalf("want ErrBusy on fallback, got %v", err)
	}
}

func TestCancelQueued(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 1, prepareDur: 50 * time.Millisecond}
	s := New(be, Config{Workers: 1, QueueDepth: 4})
	defer s.Close()

	// 占住唯一 worker。
	go func() { _, _ = runCtx(s, "busy") }()
	time.Sleep(10 * time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := s.Run(ctx, req("canceled"))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestCancelDuringInferReturnsCtxErr(t *testing.T) {
	gate := make(chan struct{})
	be := &fakeBatch{name: "fake", maxBatch: 1, runGate: gate}
	s := New(be, Config{Workers: 1})
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := s.Run(ctx, req("x"))
		done <- err
	}()
	time.Sleep(20 * time.Millisecond) // 确保已进入 RunBatch（仍阻塞在 gate）
	cancel()
	// 客户端取消应立即返回 Canceled，不等底层推理结束。
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("want Canceled, got %v", err)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Run did not return promptly on cancel")
	}
	close(gate) // 放行仍在跑的推理；batcher 跑完后释放 Prepared。
	// 等调度器干净关闭（worker/batcher 退出），证明没有 goroutine 泄漏。
	closed := make(chan struct{})
	go func() { s.Close(); close(closed) }()
	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("Close hung — goroutine leak during in-flight cancel")
	}
	if rc := be.releaseCalls.Load(); rc != 1 {
		t.Fatalf("want 1 release, got %d", rc)
	}
}

func TestRunBatchErrorFailsAll(t *testing.T) {
	sentinel := errors.New("boom")
	be := &fakeBatch{name: "fake", maxBatch: 4, failRun: sentinel}
	s := New(be, Config{Workers: 4, MaxWait: 30 * time.Millisecond})
	defer s.Close()

	const n = 4
	errs := make(chan error, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := runCtx(s, "x")
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if !errors.Is(err, sentinel) {
			t.Fatalf("want sentinel, got %v", err)
		}
	}
}

func TestPanicInRunBatchSurvived(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 2, panicRun: true, runDur: 5 * time.Millisecond}
	s := New(be, Config{Workers: 2, MaxWait: 10 * time.Millisecond})
	defer s.Close()

	_, err := runCtx(s, "x")
	if err == nil {
		t.Fatal("want error from recovered panic")
	}
	// 调度器应仍能服务后续请求（panic 没崩进程，batcher 仍活着）。
	be.panicRun = false
	if _, err := runCtx(s, "y"); err != nil {
		t.Fatalf("scheduler did not survive panic: %v", err)
	}
}

func TestReleaseAlwaysCalled(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 1, runDur: 2 * time.Millisecond}
	s := New(be, Config{Workers: 2})

	const n = 20
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = runCtx(s, "x") }()
	}
	wg.Wait()
	s.Close()

	if got := be.prepareCalls.Load(); got != int64(n) {
		t.Fatalf("want %d prepares, got %d", n, got)
	}
	if got := be.releaseCalls.Load(); got != int64(n) {
		t.Fatalf("want %d releases, got %d (leak!)", n, got)
	}
}

func TestCloseIdempotentAndStopsGoroutines(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 1, runDur: 2 * time.Millisecond}
	s := New(be, Config{Workers: 4})

	// 跑一点负载。
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, _ = runCtx(s, "x") }()
	}
	wg.Wait()

	s.Close()
	s.Close() // 幂等

	// 关闭后新请求立即 ErrClosed。
	if _, err := runCtx(s, "late"); !errors.Is(err, engine.ErrClosed) {
		t.Fatalf("want ErrClosed after Close, got %v", err)
	}
}

func TestNilImageRejected(t *testing.T) {
	be := &fakeBatch{name: "fake", maxBatch: 1}
	s := New(be, Config{Workers: 1})
	defer s.Close()
	if _, err := s.Run(context.Background(), &engine.Request{}); err == nil {
		t.Fatal("want error for nil image")
	}
}

func TestStatsRejectedBusy(t *testing.T) {
	// worker 卡在 Prepare（不消费 jobs），queue=1：第 3 个请求立即 ErrBusy。
	be := &fakeBatch{name: "fake", maxBatch: 1, prepareDur: 30 * time.Millisecond}
	s := New(be, Config{Workers: 1, QueueDepth: 1})
	defer s.Close()

	go func() { _, _ = runCtx(s, "occupy") }()
	time.Sleep(10 * time.Millisecond)
	go func() { _, _ = runCtx(s, "queued") }()
	time.Sleep(10 * time.Millisecond)
	if _, err := runCtx(s, "overflow"); !errors.Is(err, engine.ErrBusy) {
		t.Fatalf("want ErrBusy, got %v", err)
	}
	st := s.Snapshot()
	if st.RejectedBusy != 1 {
		t.Fatalf("want 1 rejected, got %d", st.RejectedBusy)
	}
	if st.Accepted != 2 {
		t.Fatalf("want 2 accepted, got %d", st.Accepted)
	}
}

func TestStatsInFlightAndBatches(t *testing.T) {
	// 用 gate + onRun 屏障让请求真正进入 RunBatch，验证在飞/批次数/成功计数。
	gate := make(chan struct{})
	started := make(chan struct{}, 1)
	be := &fakeBatch{name: "fake", maxBatch: 1, runGate: gate, onRun: func() {
		select {
		case started <- struct{}{}:
		default:
		}
	}}
	s := New(be, Config{Workers: 1, QueueDepth: 8})
	defer s.Close()

	if st := s.Snapshot(); st.Workers != 1 || st.QueueCap != 8 || st.MaxBatch != 1 {
		t.Fatalf("unexpected initial stats: %+v", st)
	}

	done := make(chan error, 1)
	go func() { _, err := runCtx(s, "a"); done <- err }()
	<-started // a 已进入 RunBatch 并阻塞在 gate
	waitFor(t, time.Second, func() bool { return s.Snapshot().InFlight == 1 })
	if st := s.Snapshot(); st.Accepted != 1 || st.InFlight != 1 || st.Batches != 1 {
		t.Fatalf("in-flight snapshot wrong: %+v", st)
	}

	close(gate)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("request did not return after gate opened")
	}
	waitFor(t, time.Second, func() bool { return s.Snapshot().InFlight == 0 })
	st := s.Snapshot()
	if st.Succeeded != 1 || st.Failed != 0 || st.Batches != 1 {
		t.Fatalf("final snapshot wrong: %+v", st)
	}
	if st.RejectedBusy != 0 || st.QueueLen != 0 {
		t.Fatalf("unexpected queue/reject: %+v", st)
	}
}

func waitFor(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("condition not met within timeout")
}

// plainEngine 是不实现 BatchEngine 的最小 Engine，用于回退路径测试。
type plainEngine struct {
	name   string
	runDur time.Duration
	calls  atomic.Int64
}

func (p *plainEngine) Name() string      { return "plain" }
func (p *plainEngine) Task() engine.Task { return engine.TaskDetection }
func (p *plainEngine) Framework() string { return "plain" }
func (p *plainEngine) Close() error      { return nil }
func (p *plainEngine) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	p.calls.Add(1)
	if p.runDur > 0 {
		select {
		case <-time.After(p.runDur):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return &engine.DetectionResult{Detections: []engine.Detection{{ClassName: req.Text}}}, nil
}
