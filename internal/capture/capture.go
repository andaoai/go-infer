// Package capture 在服务推理链路上按策略把"原图 + 预测伪标签"落盘成 YOLO 数据集。
//
// 采集池按 <engine>/<YYYYMMDD>/{images,labels} 分组，按天分目录、按模型隔离。
// 写入是异步的（不阻塞推理响应），并受总字节配额管控：超出配额时按文件 mtime
// 从最旧开始成对删除 image+label。标签直接用 YOLO 格式，采集池就是一个可被
// X-AnyLabeling/Label Studio 打开、修正后回流训练的数据集。
package capture

import (
	"fmt"
	"io/fs"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/andaoai/go-infer/internal/dataset"
	"github.com/andaoai/go-infer/internal/engine"
)

// Sample 是一次待采集的推理结果（原始图片字节 + 预测）。
type Sample struct {
	Engine     string
	Task       engine.Task
	Image      []byte // 原始上传字节，按原编码落盘
	W, H       int
	ImgExt     string // ".jpg"/".png"，空则按 ".jpg"
	Detections []engine.Detection
	Instances  []engine.Instance
}

// Config 控制采集策略与存储。
type Config struct {
	// Dir 是采集池根目录，如 dataset/pool。
	Dir string
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
	cfg    Config
	ch     chan Sample
	wg     sync.WaitGroup
	stop   chan struct{}
	mu     sync.Mutex
	size   int64 // 当前采集池总字节（启动时扫描，运行期增量维护）
	inited bool
}

// New 创建采集器并启动后台 worker。Dir 为空返回 nil（表示不采集）。
func New(cfg Config) (*Recorder, error) {
	if cfg.Dir == "" {
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
	if err := os.MkdirAll(cfg.Dir, 0o755); err != nil {
		return nil, fmt.Errorf("创建采集目录 %s: %w", cfg.Dir, err)
	}
	r := &Recorder{
		cfg:  cfg,
		ch:   make(chan Sample, cfg.Buffer),
		stop: make(chan struct{}),
	}
	r.size = r.scanSize()
	r.inited = true
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
		// 队列满：丢一个最旧的，给新样本腾位（优先保留近期数据）。
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
			// 排空队列中已入队的样本再退出。
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
	maxConf := r.maxConfidence(s)
	switch {
	case maxConf < 0:
		// 无任何检测：hard negative，必存。
		return true
	case maxConf < r.cfg.LowConf:
		// 有检测但最高置信度很低：不确定样本，必存。
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

// maxConfidence 返回样本中最高置信度；无检测返回 -1。
func (r *Recorder) maxConfidence(s Sample) float32 {
	var best float32 = -1
	for _, d := range s.Detections {
		if d.Confidence > best {
			best = d.Confidence
		}
	}
	for _, ins := range s.Instances {
		if ins.Confidence > best {
			best = ins.Confidence
		}
	}
	return best
}

func (r *Recorder) write(s Sample) {
	if s.W <= 0 || s.H <= 0 || len(s.Image) == 0 || s.Engine == "" {
		return
	}
	now := r.cfg.Now()
	day := now.Format("20060102")
	stamp := now.Format("150405")
	suffix := r.randHex(4)
	stem := fmt.Sprintf("%s_%s", stamp, suffix)
	ext := strings.ToLower(s.ImgExt)
	if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
		ext = ".jpg"
	}

	base := filepath.Join(r.cfg.Dir, sanitize(s.Engine), day)
	imgPath := filepath.Join(base, "images", stem+ext)
	lblPath := filepath.Join(base, "labels", stem+".txt")

	var labelErr error
	switch s.Task {
	case engine.TaskSegmentation:
		labelErr = dataset.WriteSegLabel(lblPath, s.W, s.H, s.Instances)
	default:
		labelErr = dataset.WriteBoxLabel(lblPath, s.W, s.H, s.Detections)
	}
	if labelErr != nil {
		log.Printf("capture: 写标签失败 %s: %v", lblPath, labelErr)
		return
	}
	if err := dataset.SaveImage(imgPath, s.Image); err != nil {
		log.Printf("capture: 写图片失败 %s: %v", imgPath, err)
		// 标签已写但图片失败，回滚标签，避免出现无图孤儿标签。
		_ = os.Remove(lblPath)
		return
	}

	// 图片 + 标签都计入配额，保证淘汰准确。
	added := int64(len(s.Image))
	if fi, err := os.Stat(lblPath); err == nil {
		added += fi.Size()
	}
	r.addSize(added)
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
	files := r.listFiles()
	sort.Slice(files, func(i, j int) bool { return files[i].mod.Before(files[j].mod) })
	for _, f := range files {
		if r.size <= r.cfg.QuotaBytes {
			break
		}
		sz := f.size
		if err := os.Remove(f.path); err != nil {
			if !os.IsNotExist(err) {
				log.Printf("capture: 删除失败 %s: %v", f.path, err)
			}
			continue
		}
		r.size -= sz
		// 若是图片，顺带删除同名标签；若是标签，顺带删除同名图片。
		peer := peerPath(f.path)
		if fi, err := os.Stat(peer); err == nil {
			if rmErr := os.Remove(peer); rmErr == nil {
				r.size -= fi.Size()
			}
		}
	}
	r.cleanEmptyDirs()
}

type fileEntry struct {
	path string
	size int64
	mod  time.Time
}

// listFiles 列出采集池下所有普通文件。
func (r *Recorder) listFiles() []fileEntry {
	var out []fileEntry
	_ = filepath.WalkDir(r.cfg.Dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return nil
		}
		out = append(out, fileEntry{path: p, size: fi.Size(), mod: fi.ModTime()})
		return nil
	})
	return out
}

// scanSize 启动时统计采集池现有总字节。
func (r *Recorder) scanSize() int64 {
	var total int64
	_ = filepath.WalkDir(r.cfg.Dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		fi, err := d.Info()
		if err == nil {
			total += fi.Size()
		}
		return nil
	})
	return total
}

func (r *Recorder) addSize(n int64) {
	r.mu.Lock()
	r.size += n
	r.mu.Unlock()
}

// cleanEmptyDirs 删除池内空的 images/labels/天/模型目录（配额淘汰后收尾）。
func (r *Recorder) cleanEmptyDirs() {
	_ = filepath.WalkDir(r.cfg.Dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() || p == r.cfg.Dir {
			return nil
		}
		entries, e := os.ReadDir(p)
		if e == nil && len(entries) == 0 {
			_ = os.Remove(p)
		}
		return nil
	})
}

// peerPath 返回与某文件同名、位于兄弟 images/labels 目录的对应文件路径。
func peerPath(path string) string {
	dir, file := filepath.Split(path)
	parent := filepath.Dir(filepath.Clean(dir))
	base := filepath.Base(filepath.Clean(dir))
	stem := strings.TrimSuffix(file, filepath.Ext(file))
	var peerDir string
	switch base {
	case "images":
		peerDir = filepath.Join(parent, "labels")
	case "labels":
		peerDir = filepath.Join(parent, "images")
	default:
		return ""
	}
	// 在 peer 目录里找任意匹配 stem 的扩展名。
	matches, err := filepath.Glob(filepath.Join(peerDir, stem+".*"))
	if err != nil || len(matches) == 0 {
		return ""
	}
	return matches[0]
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
	name = strings.ReplaceAll(name, string(os.PathSeparator), "_")
	name = strings.ReplaceAll(name, "/", "_")
	name = strings.ReplaceAll(name, "\\", "_")
	if name == "" {
		name = "unknown"
	}
	return name
}
