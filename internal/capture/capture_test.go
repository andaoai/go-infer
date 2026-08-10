package capture

import (
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
)

func fixedClock() func() time.Time {
	t := time.Date(2026, 8, 10, 14, 30, 52, 0, time.Local)
	return func() time.Time { return t }
}

func newTestRecorder(t *testing.T, cfg Config) *Recorder {
	t.Helper()
	if cfg.Dir == "" {
		cfg.Dir = t.TempDir()
	}
	if cfg.Now == nil {
		cfg.Now = fixedClock()
	}
	if cfg.Rng == nil {
		cfg.Rng = rand.New(rand.NewSource(42))
	}
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return r
}

func detSample(conf float32) Sample {
	return Sample{
		Engine: "yolov8n", Task: engine.TaskDetection,
		Image: []byte("FAKEJPEG"), W: 100, H: 100, ImgExt: ".jpg",
		Detections: []engine.Detection{{ClassID: 0, Confidence: conf, X1: 0, Y1: 0, X2: 10, Y2: 10}},
	}
}

func TestShouldCaptureRules(t *testing.T) {
	r := newTestRecorder(t, Config{Rate: 0.1, LowConf: 0.25})

	// 无检测 -> 必存
	if !r.shouldCapture(Sample{Engine: "e", W: 1, H: 1, Image: []byte{1}}) {
		t.Error("无检测样本应必存")
	}
	// 低置信度（低于阈值）-> 必存
	if !r.shouldCapture(detSample(0.1)) {
		t.Error("低置信度样本应必存")
	}
	// Rate=1 时普通样本必存
	r.cfg.Rate = 1
	if !r.shouldCapture(detSample(0.9)) {
		t.Error("Rate=1 时普通样本应必存")
	}
	// Rate=0 且高置信度 -> 不存
	r.cfg.Rate = 0
	if r.shouldCapture(detSample(0.9)) {
		t.Error("Rate=0 且高置信度时不应采集")
	}
}

func TestWriteLayoutAndFiles(t *testing.T) {
	dir := t.TempDir()
	r := newTestRecorder(t, Config{Dir: dir, Rate: 1})
	r.Record(detSample(0.9))
	r.Close()

	wantImg := filepath.Join(dir, "yolov8n", "20260810", "images")
	wantLbl := filepath.Join(dir, "yolov8n", "20260810", "labels")
	imgs, _ := os.ReadDir(wantImg)
	lbls, _ := os.ReadDir(wantLbl)
	if len(imgs) != 1 || !strings.HasSuffix(imgs[0].Name(), ".jpg") {
		t.Fatalf("图片落盘异常: %v", imgs)
	}
	if len(lbls) != 1 || !strings.HasSuffix(lbls[0].Name(), ".txt") {
		t.Fatalf("标签落盘异常: %v", lbls)
	}
	// 文件名格式 HHMMSS_<hex>
	name := imgs[0].Name()
	if !strings.HasPrefix(name, "143052_") {
		t.Errorf("文件名前缀应为时分秒，got %s", name)
	}
}

func TestQuotaEviction(t *testing.T) {
	dir := t.TempDir()
	// 每个样本约 8 字节图 + 36 字节标签 ≈ 44 字节；配额 90 应只保留约 2 个。
	r := newTestRecorder(t, Config{Dir: dir, Rate: 1, QuotaBytes: 90})
	for i := 0; i < 6; i++ {
		r.Record(detSample(0.9))
	}
	r.Close()

	var imgs int
	_ = filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && strings.HasSuffix(p, ".jpg") {
			imgs++
		}
		return nil
	})
	if imgs > 2 {
		t.Errorf("配额未生效，剩余图片 %d 张（应 <=2）", imgs)
	}
}

func TestSanitizeEngineName(t *testing.T) {
	if got := sanitize("../evil"); got == "../evil" || strings.Contains(got, "..") {
		t.Errorf("sanitize 未拦截路径穿越: %q", got)
	}
	if got := sanitize("a/b\\c"); strings.ContainsAny(got, "/\\") {
		t.Errorf("sanitize 未去除分隔符: %q", got)
	}
}

func TestNilRecorderSafe(t *testing.T) {
	var r *Recorder
	r.Record(detSample(0.9)) // 不应 panic
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
}
