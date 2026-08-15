// Package capture 在服务推理链路上按策略把"原图 + 预测伪标签"落盘成数据集。
//
// 采集池按 <engine>/<YYYYMMDD>/images|labels 分组，按天分目录、按模型隔离。
// 写入是异步的（不阻塞推理响应），并受总字节配额管控：超出配额时按文件 mtime
// 从最旧开始成对删除 image+label。落盘格式由注入的 format.Codec 决定（YOLO
// 为首例），字节落到注入的 storage.Storage（本地 FS 为首例），二者均可替换。
package capture

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/andaoai/go-infer/internal/data"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/format"
	"github.com/andaoai/go-infer/internal/fsx"
	"github.com/andaoai/go-infer/internal/storage"
)

// Sample 是一次待采集的推理结果（原始图片字节 + 引擎结果）。
type Sample struct {
	Engine string
	Image  []byte // 原始上传字节，按原编码落盘
	W, H   int
	ImgExt string // ".jpg"/".png"，空则按 ".jpg"
	Result engine.Result
}

// Config 控制采集策略与存储。
type Config struct {
	// Store 是采集池字节存储（必填）。
	Store storage.Storage
	// Codec 决定标签编码与图片/标签 key 推导（必填）。
	Codec format.Codec
	// PoolRoot 是采集池在 Store 下的前缀，如 "pool"。
	PoolRoot string
	// Rate 是"普通样本"的采样概率（0~1）。无检测/低置信度样本不受此限，必存。
	Rate float64
	// LowConf 是低置信阈值：最高置信度低于它（且有检测）视为不确定样本，必存。
	LowConf float32
	// QuotaBytes 是采集池总字节上限；<=0 表示不限制。
	QuotaBytes int64
	// Buffer 是异步写入队列长度；0 用默认 256。
	Buffer int
	// Now 用于注入时间（测试）；nil 用 time.Now。
	Now func() time.Time
	// Rng 用于采样随机数与文件名后缀（测试）；nil 用全局 rand。
	Rng *rand.Rand
}

// Recorder 是异步采集器。零值不可用，用 New 构造。
type Recorder struct {
	cfg  Config
	ch   chan Sample
	wg   sync.WaitGroup
	stop chan struct{}

	mu               sync.Mutex
	size             int64  // 当前采集池总字节（启动时扫描，运行期增量维护）
	writesSinceEvict uint64 // 距上次淘汰以来的成功写入次数（节流全量 List+排序）
}

// New 创建采集器并启动后台 worker。Store/Codec 为空返回 nil（表示不采集）。
func New(cfg Config) (*Recorder, error) {
	if cfg.Store == nil || cfg.Codec == nil {
		return nil, nil
	}
	if cfg.Rate < 0 {
		cfg.Rate = 0
	}
	if cfg.Rate > 1 {
		cfg.Rate = 1
	}
	if cfg.LowConf < 0 {
		cfg.LowConf = 0
	}
	if cfg.Buffer <= 0 {
		cfg.Buffer = 256
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	ctx := context.Background()
	size, err := scanSize(ctx, cfg.Store, cfg.PoolRoot)
	if err != nil {
		return nil, fmt.Errorf("扫描采集池 %s: %w", cfg.PoolRoot, err)
	}
	r := &Recorder{
		cfg:  cfg,
		ch:   make(chan Sample, cfg.Buffer),
		stop: make(chan struct{}),
		size: size,
	}
	r.wg.Add(1)
	go r.run()
	return r, nil
}

// Record 异步入队一个样本。队列满时丢弃最旧待写以避免阻塞推理调用方。
func (r *Recorder) Record(s Sample) {
	if r == nil {
		return
	}
	if !r.shouldCapture(s) {
		return
	}
	select {
	case r.ch <- s:
	default:
		select {
		case <-r.ch:
		default:
		}
		select {
		case r.ch <- s:
		default:
		}
	}
}

// Close 停止接收并等待已入队样本落盘。
func (r *Recorder) Close() error {
	if r == nil {
		return nil
	}
	close(r.stop)
	r.wg.Wait()
	return nil
}

func (r *Recorder) run() {
	defer r.wg.Done()
	for {
		select {
		case <-r.stop:
			for {
				select {
				case s := <-r.ch:
					r.write(s)
				default:
					return
				}
			}
		case s := <-r.ch:
			r.write(s)
		}
	}
}

// shouldCapture 实现"默认按 Rate 采样 + 必存无检测 + 必存低置信度"。
func (r *Recorder) shouldCapture(s Sample) bool {
	maxConf := maxConfidence(s.Result)
	switch {
	case maxConf < 0:
		return true
	case maxConf < r.cfg.LowConf:
		return true
	default:
		if r.cfg.Rate >= 1 {
			return true
		}
		if r.cfg.Rate <= 0 {
			return false
		}
		if r.cfg.Rng != nil {
			return r.cfg.Rng.Float64() < r.cfg.Rate
		}
		return rand.Float64() < r.cfg.Rate
	}
}

// maxConfidence 返回结果中最高置信度；无检测返回 -1。
func maxConfidence(r engine.Result) float32 {
	if r == nil {
		return -1
	}
	var best float32 = -1
	switch v := r.(type) {
	case *engine.DetectionResult:
		for _, d := range v.Detections {
			if d.Confidence > best {
				best = d.Confidence
			}
		}
	case *engine.SegmentationResult:
		for _, ins := range v.Instances {
			if ins.Confidence > best {
				best = ins.Confidence
			}
		}
	}
	return best
}

func (r *Recorder) write(s Sample) {
	if s.W <= 0 || s.H <= 0 || len(s.Image) == 0 || s.Engine == "" || s.Result == nil {
		return
	}
	ctx := context.Background()
	objs, err := data.ObjectsFromResult(s.Result)
	if err != nil {
		log.Printf("capture: 结果转换失败 (%s): %v", s.Engine, err)
		return
	}
	labelBytes, err := r.cfg.Codec.Encode(objs, s.W, s.H)
	if err != nil {
		log.Printf("capture: 标签编码失败 (%s): %v", s.Engine, err)
		return
	}

	now := r.cfg.Now()
	day := now.Format("20060102")
	stamp := now.Format("150405")
	suffix := r.randHex(4)
	stem := fmt.Sprintf("%s_%s", stamp, suffix)
	ext := strings.ToLower(s.ImgExt)
	if !r.cfg.Codec.IsImageKey("x" + ext) {
		ext = ".jpg"
	}
	imgKey := fsx.Join(r.cfg.PoolRoot, sanitize(s.Engine), day, "images", stem+ext)
	lblKey := r.cfg.Codec.LabelKey(imgKey)

	// 先写标签，再写图片；图片失败则回滚标签，避免孤儿标签。
	if err := r.cfg.Store.Put(ctx, lblKey, bytes.NewReader(labelBytes)); err != nil {
		log.Printf("capture: 写标签失败 %s: %v", lblKey, err)
		return
	}
	if err := r.cfg.Store.Put(ctx, imgKey, bytes.NewReader(s.Image)); err != nil {
		log.Printf("capture: 写图片失败 %s: %v", imgKey, err)
		_ = r.cfg.Store.Remove(ctx, lblKey)
		return
	}

	r.addSize(int64(len(s.Image)) + int64(len(labelBytes)))
	r.enforceQuota()
}

// enforceQuota 当采集池超过配额时，按 mtime 从最旧文件开始删除直到回到上限。
func (r *Recorder) enforceQuota() {
	if r.cfg.QuotaBytes <= 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.size <= r.cfg.QuotaBytes {
		return
	}
	if r.writesSinceEvict < 64 {
		return
	}
	ctx := context.Background()
	infos, err := r.cfg.Store.List(ctx, r.cfg.PoolRoot)
	if err != nil {
		log.Printf("capture: 枚举采集池失败: %v", err)
		return
	}
	files := make([]storage.Info, 0, len(infos))
	for _, in := range infos {
		if !in.IsDir {
			files = append(files, in)
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].ModTime.Before(files[j].ModTime) })
	for _, f := range files {
		if r.size <= r.cfg.QuotaBytes {
			break
		}
		if err := r.cfg.Store.Remove(ctx, f.Key); err != nil {
			log.Printf("capture: 删除失败 %s: %v", f.Key, err)
			continue
		}
		r.size -= f.Size
		// 图片连带删同名标签，标签连带删同名图片。
		var peer string
		if r.cfg.Codec.IsImageKey(f.Key) {
			peer = r.cfg.Codec.LabelKey(f.Key)
		} else {
			peer = "" // 非图片不反推（格式自定），由图片删除时连带处理
		}
		if peer != "" {
			if pi, err := r.cfg.Store.Stat(ctx, peer); err == nil {
				if r.cfg.Store.Remove(ctx, peer) == nil {
					r.size -= pi.Size
				}
			}
		}
	}
	r.writesSinceEvict = 0
}

// scanSize 统计存储中采集池现有总字节。
func scanSize(ctx context.Context, st storage.Storage, root string) (int64, error) {
	infos, err := st.List(ctx, root)
	if err != nil {
		return 0, err
	}
	var total int64
	for _, in := range infos {
		if !in.IsDir {
			total += in.Size
		}
	}
	return total, nil
}

func (r *Recorder) addSize(n int64) {
	r.mu.Lock()
	r.size += n
	r.writesSinceEvict++
	r.mu.Unlock()
}

func (r *Recorder) randHex(n int) string {
	const hex = "0123456789abcdef"
	b := make([]byte, n)
	if r.cfg.Rng != nil {
		for i := range b {
			b[i] = hex[r.cfg.Rng.Intn(16)]
		}
	} else {
		for i := range b {
			b[i] = hex[rand.Intn(16)]
		}
	}
	return string(b)
}

// sanitize 把引擎名里的路径分隔符/特殊字符替换掉，避免越出采集根目录。
func sanitize(name string) string {
	name = strings.TrimSpace(name)
	name = strings.ReplaceAll(name, "..", "_")
	name = strings.ReplaceAll(name, "/", "_")
	name = strings.ReplaceAll(name, "\\", "_")
	if name == "" {
		name = "unknown"
	}
	return name
}
