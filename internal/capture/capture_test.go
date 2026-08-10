package capture

import (
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/storage/local"
)

func fixedClock() func() time.Time {
	t := time.Date(2026, 8, 10, 14, 30, 52, 0, time.Local)
	return func() time.Time { return t }
}

func newTestRecorder(t *testing.T, cfg Config) (*Recorder, string) {
	t.Helper()
	root := t.TempDir()
	st, err := local.New(root)
	if err != nil {
		t.Fatalf("local storage: %v", err)
	}
	if cfg.Store == nil {
		cfg.Store = st
	}
	if cfg.Codec == nil {
		cfg.Codec = yolo.New()
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
	return r, root
}

func detSample(conf float32) Sample {
	return Sample{
		Engine: "yolov8n",
		Image:  []byte("FAKEJPEG"), W: 100, H: 100, ImgExt: ".jpg",
		Result: &engine.DetectionResult{
			Detections: []engine.Detection{
				{ClassID: 0, Confidence: conf, X1: 0, Y1: 0, X2: 10, Y2: 10},
			},
		},
	}
}

func noDetSample() Sample {
	return Sample{
		Engine: "yolov8n",
		Image:  []byte("FAKEJPEG"), W: 100, H: 100, ImgExt: ".jpg",
		Result: &engine.DetectionResult{},
	}
}

func TestShouldCaptureRules(t *testing.T) {
	r, _ := newTestRecorder(t, Config{Rate: 0.1, LowConf: 0.25})

	// 无检测 -> 必存
	if !r.shouldCapture(noDetSample()) {
		t.Error("无检测样本应必存")
	}
	// 低置信度 -> 必存
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
	r, root := newTestRecorder(t, Config{Rate: 1})
	r.Record(detSample(0.9))
	r.Close()

	wantImg := filepath.Join(root, "yolov8n", "20260810", "images")
	wantLbl := filepath.Join(root, "yolov8n", "20260810", "labels")
	imgs, _ := filepath.Glob(filepath.Join(wantImg, "*.jpg"))
	lbls, _ := filepath.Glob(filepath.Join(wantLbl, "*.txt"))
	if len(imgs) != 1 {
		t.Fatalf("图片落盘异常: %v", imgs)
	}
	if len(lbls) != 1 {
		t.Fatalf("标签落盘异常: %v", lbls)
	}
	name := filepath.Base(imgs[0])
	if !strings.HasPrefix(name, "143052_") {
		t.Errorf("文件名前缀应为时分秒，got %s", name)
	}
	// 标签内容合法 YOLO 框
	b := readFile(t, lbls[0])
	if !strings.HasPrefix(b, "0 ") {
		t.Errorf("标签应为 cls cx cy w h 行，got %q", b)
	}
}

func TestQuotaEviction(t *testing.T) {
	r, root := newTestRecorder(t, Config{Rate: 1, QuotaBytes: 90})
	for i := 0; i < 6; i++ {
		r.Record(detSample(0.9))
	}
	r.Close()

	imgs, _ := filepath.Glob(filepath.Join(root, "yolov8n", "20260810", "images", "*.jpg"))
	if len(imgs) > 2 {
		t.Errorf("配额未生效，剩余图片 %d 张（应 <=2）", len(imgs))
	}
}

func TestHardNegativeWritesEmptyLabel(t *testing.T) {
	r, root := newTestRecorder(t, Config{Rate: 1})
	r.Record(noDetSample())
	r.Close()

	lbls, _ := filepath.Glob(filepath.Join(root, "yolov8n", "20260810", "labels", "*.txt"))
	if len(lbls) != 1 {
		t.Fatalf("hard negative 应落空标签，got %v", lbls)
	}
	if b := readFile(t, lbls[0]); b != "" {
		t.Errorf("空检测应为空标签，got %q", b)
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
	r.Record(detSample(0.9))
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
}

func readFile(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
