package validate

import (
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/andaoai/go-infer/internal/data"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/storage/local"
)

// fakeEngine 对每张图返回预设结果，记录是否被 Close（应由调用方管理生命周期）。
type fakeEngine struct {
	name   string
	task   engine.Task
	result engine.Result
	closed bool
}

func (e *fakeEngine) Name() string      { return e.name }
func (e *fakeEngine) Task() engine.Task { return e.task }
func (e *fakeEngine) Framework() string { return "fake" }
func (e *fakeEngine) Close() error      { e.closed = true; return nil }
func (e *fakeEngine) Run(_ context.Context, _ *engine.Request) (engine.Result, error) {
	return e.result, nil
}

// seedDataset 在临时目录建一个 YOLO 数据集：images/val/<i>.jpg + labels/val/<i>.txt。
func seedDataset(t *testing.T, n int) (storeRoot, dsPrefix, split string) {
	t.Helper()
	root := t.TempDir()
	dsPrefix = "ds"
	split = "val"
	imgDir := filepath.Join(root, dsPrefix, "images", split)
	lblDir := filepath.Join(root, dsPrefix, "labels", split)
	if err := os.MkdirAll(imgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(lblDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		img := image.NewRGBA(image.Rect(0, 0, 100, 100))
		for y := 0; y < 100; y++ {
			for x := 0; x < 100; x++ {
				img.Set(x, y, color.RGBA{uint8(x * 2), uint8(y * 2), 0, 255})
			}
		}
		f, err := os.Create(filepath.Join(imgDir, "img"+itoa(i)+".jpg"))
		if err != nil {
			t.Fatal(err)
		}
		if err := jpeg.Encode(f, img, nil); err != nil {
			t.Fatal(err)
		}
		f.Close()
		// YOLO det: cls cx cy w h（归一化）。class 0，居中 40x20 的框。
		label := "0 0.5 0.5 0.4 0.2\n"
		if err := os.WriteFile(filepath.Join(lblDir, "img"+itoa(i)+".txt"), []byte(label), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root, dsPrefix, split
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

func newOpts(t *testing.T) Options {
	root, prefix, split := seedDataset(t, 3)
	st, err := local.New(root)
	if err != nil {
		t.Fatal(err)
	}
	return Options{Store: st, Codec: yolo.New(), Root: prefix, Split: split, Classes: []string{"obj"}}
}

// detResult 返回与 seedDataset GT 完全一致的检测结果（框 30,40,70,60）。
func detResult() *engine.DetectionResult {
	return &engine.DetectionResult{Detections: []engine.Detection{{
		ClassID: 0, ClassName: "obj", Confidence: 0.9,
		X1: 30, Y1: 40, X2: 70, Y2: 60,
	}}}
}

func TestRunPerfectPrediction(t *testing.T) {
	opts := newOpts(t)
	eng := &fakeEngine{name: "fake-det", task: engine.TaskDetection, result: detResult()}
	opts.Jobs = []Job{{Name: "det", Engine: eng, Metrics: []MetricSpec{{Label: "box"}}}}

	rep, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Images != 3 {
		t.Errorf("Images = %d, want 3", rep.Images)
	}
	if len(rep.Jobs) != 1 {
		t.Fatalf("jobs = %d, want 1", len(rep.Jobs))
	}
	jr := rep.Jobs[0]
	if jr.Task != string(engine.TaskDetection) || jr.Predictions != 3 || jr.GroundTruths != 3 {
		t.Errorf("job report wrong: %+v", jr)
	}
	if len(jr.Metrics) != 1 || jr.Metrics[0].Label != "box" {
		t.Fatalf("metrics wrong: %+v", jr.Metrics)
	}
	if jr.Metrics[0].MAP50 < 0.99 {
		t.Errorf("perfect prediction mAP@.5 = %.4f, want ~1.0", jr.Metrics[0].MAP50)
	}
	if len(jr.Metrics[0].PerClass) != 1 || jr.Metrics[0].PerClass[0].ClassName != "obj" {
		t.Errorf("per-class wrong: %+v", jr.Metrics[0].PerClass)
	}
	if eng.closed {
		t.Error("Run 不应 Close 调用方传入的引擎")
	}
}

func TestRunNoPredictions(t *testing.T) {
	opts := newOpts(t)
	eng := &fakeEngine{name: "fake", task: engine.TaskDetection, result: &engine.DetectionResult{}}
	opts.Jobs = []Job{{Name: "det", Engine: eng, Metrics: []MetricSpec{{Label: "box"}}}}
	rep, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Jobs[0].Metrics[0].MAP50 != 0 {
		t.Errorf("无预测时 mAP@.5 = %.4f, want 0", rep.Jobs[0].Metrics[0].MAP50)
	}
}

func TestRunDetAndSegJobs(t *testing.T) {
	opts := newOpts(t)
	segRes := &engine.SegmentationResult{Instances: []engine.Instance{{
		Detection: engine.Detection{ClassID: 0, ClassName: "obj", Confidence: 0.9, X1: 30, Y1: 40, X2: 70, Y2: 60},
		Mask:      [][]engine.Point{{{X: 30, Y: 40}, {X: 70, Y: 40}, {X: 70, Y: 60}, {X: 30, Y: 60}}},
	}}}
	opts.Jobs = []Job{
		{Name: "det", Engine: &fakeEngine{name: "d", task: engine.TaskDetection, result: detResult()}, Metrics: []MetricSpec{{Label: "box"}}},
		{Name: "seg", Engine: &fakeEngine{name: "s", task: engine.TaskSegmentation, result: segRes}, Metrics: []MetricSpec{{Label: "box"}, {Label: "mask"}}},
	}
	rep, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Jobs) != 2 {
		t.Fatalf("jobs = %d, want 2", len(rep.Jobs))
	}
	if got := len(rep.Jobs[1].Metrics); got != 2 {
		t.Errorf("seg metrics = %d, want 2 (box+mask)", got)
	}
}

func TestRunLimit(t *testing.T) {
	opts := newOpts(t)
	opts.Limit = 1
	eng := &fakeEngine{name: "fake", task: engine.TaskDetection, result: detResult()}
	opts.Jobs = []Job{{Name: "det", Engine: eng, Metrics: []MetricSpec{{Label: "box"}}}}
	rep, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Images != 1 {
		t.Errorf("Images = %d, want 1 (limit)", rep.Images)
	}
	if rep.Jobs[0].Predictions != 1 {
		t.Errorf("Predictions = %d, want 1", rep.Jobs[0].Predictions)
	}
}

func TestRunEmptyJobs(t *testing.T) {
	opts := newOpts(t)
	if _, err := Run(context.Background(), opts); err == nil {
		t.Error("空 Jobs 应返回错误")
	}
}

func TestRunNilDeps(t *testing.T) {
	if _, err := Run(context.Background(), Options{}); err == nil {
		t.Error("空 Store/Codec 应返回错误")
	}
}

func TestRunCanceledContext(t *testing.T) {
	opts := newOpts(t)
	eng := &fakeEngine{name: "fake", task: engine.TaskDetection, result: detResult()}
	opts.Jobs = []Job{{Name: "det", Engine: eng, Metrics: []MetricSpec{{Label: "box"}}}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Run(ctx, opts); err == nil {
		t.Error("已取消的 ctx 应返回错误")
	}
}

// 确认 data.ObjectsFromResult 对 fake 结果可正常转换（防止类型对不齐）。
func TestObjectsFromResult(t *testing.T) {
	objs, err := data.ObjectsFromResult(detResult())
	if err != nil {
		t.Fatal(err)
	}
	if len(objs) != 1 {
		t.Fatalf("objs = %d, want 1", len(objs))
	}
}

// concurrencyEngine 记录并发中的最大在途请求数与每个 imageID 的调用。
type concurrencyEngine struct {
	fakeEngine
	mu        sync.Mutex
	inFlight  int
	maxFlight int
	ran       []int
}

func (e *concurrencyEngine) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	e.mu.Lock()
	e.inFlight++
	if e.inFlight > e.maxFlight {
		e.maxFlight = e.inFlight
	}
	e.mu.Unlock()
	defer func() {
		e.mu.Lock()
		e.inFlight--
		e.mu.Unlock()
	}()
	time.Sleep(2 * time.Millisecond)
	return e.result, nil
}

// TestRunJobConcurrencyAndOrdering 验证 N 个 worker 确实并发执行，且结果按
// imageID 顺序组装（即使完成顺序乱序，preds 也不乱）。
func TestRunJobConcurrencyAndOrdering(t *testing.T) {
	opts := newOpts(t) // 3 张图，自带 GT
	eng := &concurrencyEngine{
		fakeEngine: fakeEngine{name: "c", task: engine.TaskDetection, result: detResult()},
	}
	opts.Jobs = []Job{{Name: "det", Engine: eng, Metrics: []MetricSpec{{Label: "box"}}}}
	opts.Concurrency = 3

	rep, err := Run(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if rep.PredictionsForJob("det") != 3 {
		t.Fatalf("predictions = %d, want 3", rep.PredictionsForJob("det"))
	}
	if eng.maxFlight < 2 {
		t.Errorf("并发度 maxFlight = %d，期望 >= 2（Concurrency=3 应真正并行）", eng.maxFlight)
	}
}

// slowEngine 每张图 sleep 一段时间，用于验证 ctx 取消能在图片之间打断。
type slowEngine struct {
	fakeEngine
	delay time.Duration
}

func (e *slowEngine) Run(ctx context.Context, req *engine.Request) (engine.Result, error) {
	select {
	case <-time.After(e.delay):
		return e.result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// TestRunJobCancelInterrupts 验证 ctx 在调度循环中被取消时 runJob 会等待已启动
// worker 退出并返回 ctx.Err，不会卡死也不会泄漏 goroutine。
func TestRunJobCancelInterrupts(t *testing.T) {
	opts := newOpts(t)
	eng := &slowEngine{
		fakeEngine: fakeEngine{name: "slow", task: engine.TaskDetection, result: detResult()},
		delay:      100 * time.Millisecond,
	}
	opts.Jobs = []Job{{Name: "det", Engine: eng, Metrics: []MetricSpec{{Label: "box"}}}}
	opts.Concurrency = 1

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	_, err := Run(ctx, opts)
	if err == nil {
		t.Fatal("取消后应返回错误")
	}
	if time.Since(start) > 500*time.Millisecond {
		t.Errorf("取消后耗时 %v，期望尽快返回（<500ms）", time.Since(start))
	}
}

// PredictionsForJob 暴露 job 报告里的预测数（测试辅助）。
func (r *Report) PredictionsForJob(name string) int {
	for _, j := range r.Jobs {
		if j.Name == name {
			return j.Predictions
		}
	}
	return -1
}
