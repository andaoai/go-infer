// Package validate 是引擎/存储/格式无关的数据集 mAP 校验库。
//
// 它把"加载数据集 → 逐张推理 → 算 mAP"的循环从 cmd/validate 和 HTTP 层中
// 抽出来，调用方只需构造好引擎（engine.Engine）、存储（storage.Storage）和
// 标签编解码（format.Codec）并通过 Options 注入。validate 包不依赖任何具体
// 引擎实现（detect/seg）或本地存储，新增后端/格式时本包无需改动。
package validate

import (
	"bytes"
	"context"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/andaoai/go-infer/internal/data"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/format"
	"github.com/andaoai/go-infer/internal/metric"
	"github.com/andaoai/go-infer/internal/storage"
)

// Options 控制一次数据集校验。
type Options struct {
	Store   storage.Storage // 必填，数据集字节存储
	Codec   format.Codec    // 必填，标签编解码（首个实现为 yolo）
	Root    string          // 数据集在 Store 内的前缀（相对 store root）
	Split   string          // 数据划分，如 "train2017"
	Limit   int             // 只校验前 N 张，0=全量
	Classes []string        // 类别名，用于报告显示
	// Concurrency 是单 Job 内并发推理的 goroutine 数。引擎是否真正合批
	// 取决于其实现（调用方通常已用 sched 包装，dynamic batching 在调度层
	// 完成）。0 取 min(runtime.NumCPU(),4)，避免 seg 原型缓冲在小机器上爆内存。
	Concurrency int
	Jobs        []Job // 要跑的引擎列表（检测/分割...），不能为空
}

// Job 是一个待校验的引擎及要计算的指标维度。
type Job struct {
	Name    string // 报告显示名，如 "检测模型"
	Engine  engine.Engine
	Metrics []MetricSpec
}

// MetricSpec 描述一个指标维度：框 IoU 或掩膜 IoU。
type MetricSpec struct {
	Label   string // 报告标签，如 "box" / "mask"
	UseMask bool   // false=框 IoU，true=掩膜 IoU
}

// Report 是一次校验的完整结果。
type Report struct {
	Images  int         `json:"images"`
	Classes int         `json:"classes"`
	TookMs  int64       `json:"took_ms"`
	Jobs    []JobReport `json:"jobs"`
}

// JobReport 是单个引擎的校验结果。
type JobReport struct {
	Name         string         `json:"name"`
	Task         string         `json:"task"`
	Predictions  int            `json:"predictions"`
	GroundTruths int            `json:"ground_truths"`
	TookMs       int64          `json:"took_ms"`
	Metrics      []MetricResult `json:"metrics"`
}

// MetricResult 是一个指标维度的 mAP 与每类 AP。
type MetricResult struct {
	Label    string    `json:"label"`
	MAP50    float64   `json:"map50"`
	MAP5095  float64   `json:"map5095"`
	PerClass []ClassAP `json:"per_class"`
}

// ClassAP 是单个类别的 AP@.5。
type ClassAP struct {
	ClassID   int     `json:"class_id"`
	ClassName string  `json:"class_name"`
	AP50      float64 `json:"ap50"`
}

// Run 加载数据集、依次跑每个 Job、计算指标，返回报告。
// ctx 取消在图片之间检查；已开始的单次引擎 Run 受底层 C 调用限制不可中断。
func Run(ctx context.Context, opts Options) (*Report, error) {
	if opts.Store == nil || opts.Codec == nil {
		return nil, fmt.Errorf("Store 和 Codec 不能为空")
	}
	if len(opts.Jobs) == 0 {
		return nil, fmt.Errorf("至少需要一个校验 Job")
	}

	samples, gts, err := loadDataset(ctx, opts)
	if err != nil {
		return nil, err
	}

	report := &Report{Images: len(samples), Classes: len(opts.Classes)}
	start := time.Now()
	for _, job := range opts.Jobs {
		jr, err := runJob(ctx, job, samples, gts, opts.Classes, opts.Concurrency)
		if err != nil {
			return nil, fmt.Errorf("job %s: %w", job.Name, err)
		}
		report.Jobs = append(report.Jobs, jr)
	}
	report.TookMs = time.Since(start).Milliseconds()
	return report, nil
}

// loadDataset 枚举图片、解码图片与标签，组 samples 与 ground truths。
func loadDataset(ctx context.Context, opts Options) ([]data.Sample, []metric.GroundTruth, error) {
	refs, err := opts.Codec.ListImages(ctx, opts.Store, opts.Root, opts.Split)
	if err != nil {
		return nil, nil, fmt.Errorf("枚举数据集: %w", err)
	}
	if opts.Limit > 0 && opts.Limit < len(refs) {
		refs = refs[:opts.Limit]
	}
	if len(refs) == 0 {
		return nil, nil, fmt.Errorf("数据集 %s/%s 没有图片", opts.Root, opts.Split)
	}

	samples := make([]data.Sample, len(refs))
	gts := make([]metric.GroundTruth, 0, len(refs)*4)
	for i, ref := range refs {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		img, w, h, err := loadImage(ctx, opts.Store, ref.ImageKey)
		if err != nil {
			return nil, nil, err
		}
		var objs []data.Object
		if rc, err := opts.Store.Get(ctx, ref.LabelKey); err == nil {
			buf := new(bytes.Buffer)
			_, _ = io.Copy(buf, rc)
			rc.Close()
			if objs, err = opts.Codec.Decode(buf.Bytes(), w, h); err != nil {
				return nil, nil, fmt.Errorf("解码标签 %s: %w", ref.LabelKey, err)
			}
		}
		samples[i] = data.Sample{Key: ref.ImageKey, Image: img, W: w, H: h, Objects: objs}
		for _, o := range objs {
			gts = append(gts, metric.GroundTruth{ImageID: i, Object: o, W: w, H: h})
		}
	}
	return samples, gts, nil
}

func loadImage(ctx context.Context, st storage.Storage, key string) (image.Image, int, int, error) {
	rc, err := st.Get(ctx, key)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("读取图片 %s: %w", key, err)
	}
	defer rc.Close()
	img, _, err := image.Decode(rc)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("解码图片 %s: %w", key, err)
	}
	b := img.Bounds()
	return img, b.Dx(), b.Dy(), nil
}

func runJob(ctx context.Context, job Job, samples []data.Sample, gts []metric.GroundTruth, classes []string, concurrency int) (JobReport, error) {
	jr := JobReport{Name: job.Name, Task: string(job.Engine.Task()), GroundTruths: len(gts)}
	if len(job.Metrics) == 0 {
		job.Metrics = []MetricSpec{{Label: "box", UseMask: false}}
	}

	workers := concurrency
	if workers <= 0 {
		workers = runtime.NumCPU()
		if workers > 4 {
			workers = 4
		}
	}
	if workers > len(samples) {
		workers = len(samples)
	}
	if workers < 1 {
		workers = 1
	}

	// 每张图推理出的对象按 imageID 写入预留切片；各 goroutine 写不同下标，
	// 无需加锁。append 到 preds 在 WaitGroup 汇合后单线程进行。
	objsByImg := make([][]data.Object, len(samples))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	var firstErr error
	var errMu sync.Mutex

	jobStart := time.Now()
	for i := range samples {
		if err := ctx.Err(); err != nil {
			// 仍等待已启动的 worker 结束，避免泄漏 goroutine。
			wg.Wait()
			return jr, err
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int) {
			defer wg.Done()
			defer func() { <-sem }()
			res, err := job.Engine.Run(ctx, &engine.Request{Image: samples[idx].Image})
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("第 %d 张推理失败: %w", idx, err)
				}
				errMu.Unlock()
				return
			}
			objs, err := data.ObjectsFromResult(res)
			if err != nil {
				errMu.Lock()
				if firstErr == nil {
					firstErr = fmt.Errorf("第 %d 张结果转换失败: %w", idx, err)
				}
				errMu.Unlock()
				return
			}
			objsByImg[idx] = objs
		}(i)
	}
	wg.Wait()
	if firstErr != nil {
		return jr, firstErr
	}

	preds := make([]metric.Prediction, 0, len(samples)*4)
	for i, objs := range objsByImg {
		for _, o := range objs {
			preds = append(preds, metric.Prediction{ImageID: i, Object: o})
		}
	}
	jr.Predictions = len(preds)
	jr.TookMs = time.Since(jobStart).Milliseconds()

	// 单次 Evaluate 同时得到 AP@.5、mAP@.5 与 mAP@.50:.95（掩膜只栅格化一次），
	// 取代原来的 metric.AP + metric.MAPOverThresholds 双调用。
	for _, spec := range job.Metrics {
		er := metric.Evaluate(preds, gts, spec.UseMask)
		jr.Metrics = append(jr.Metrics, MetricResult{
			Label:    spec.Label,
			MAP50:    er.MAP50,
			MAP5095:  er.MAP5095,
			PerClass: classAPs(er.PerClass, gts, classes),
		})
	}
	return jr, nil
}

// classAPs 只输出在 GT 中出现过的类，按 AP 降序。
func classAPs(perClass map[int]float64, gts []metric.GroundTruth, classes []string) []ClassAP {
	type row struct {
		id   int
		name string
		ap   float64
	}
	seen := map[int]bool{}
	var rows []row
	for _, g := range gts {
		id := g.Object.ClassID
		if seen[id] {
			continue
		}
		seen[id] = true
		name := fmt.Sprintf("class_%d", id)
		if id >= 0 && id < len(classes) {
			name = classes[id]
		}
		rows = append(rows, row{id: id, name: name, ap: perClass[id]})
	}
	sort.Slice(rows, func(a, b int) bool { return rows[a].ap > rows[b].ap })
	out := make([]ClassAP, 0, len(rows))
	for _, r := range rows {
		out = append(out, ClassAP{ClassID: r.id, ClassName: r.name, AP50: r.ap})
	}
	return out
}
