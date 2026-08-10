package promote

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// seedPool 构造一个采集池：
//
//	<pool>/<eng>/<date>/images/<stem>.jpg
//	<pool>/<eng>/<date>/labels/<stem>.txt （有标签时）
func seedPool(t *testing.T, pool, eng, date, stem string, withLabel bool) {
	t.Helper()
	imgDir := filepath.Join(pool, eng, date, "images")
	lblDir := filepath.Join(pool, eng, date, "labels")
	if err := os.MkdirAll(imgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(imgDir, stem+".jpg"), []byte("img"), 0o644); err != nil {
		t.Fatal(err)
	}
	if withLabel {
		if err := os.MkdirAll(lblDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(lblDir, stem+".txt"), []byte("0 0.5 0.5 0.2 0.2\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func fixedNow() time.Time { return time.Date(2026, 8, 10, 15, 0, 0, 0, time.Local) }

func TestPromoteMovesAndWritesYAML(t *testing.T) {
	pool := t.TempDir()
	root := t.TempDir()
	seedPool(t, pool, "yolov8n", "20260810", "143052_ab", true)
	seedPool(t, pool, "yolov8n", "20260810", "143100_cd", false) // hard negative，无标签

	res, err := Run(Options{
		PoolDir: pool, DatasetRoot: root, Move: true,
		Classes: []string{"person"}, Now: fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 2 {
		t.Fatalf("want 2 promoted, got %d", res.Total)
	}

	// 目标位置应有两张图、两个标签（无标签样本生成空标签）。
	imgDir := filepath.Join(root, "yolov8n", "images", "train")
	lblDir := filepath.Join(root, "yolov8n", "labels", "train")
	if imgs, _ := os.ReadDir(imgDir); len(imgs) != 2 {
		t.Errorf("目标图片数 = %d, want 2", len(imgs))
	}
	if lbls, _ := os.ReadDir(lblDir); len(lbls) != 2 {
		t.Errorf("目标标签数 = %d, want 2", len(lbls))
	}
	// 移动后源目录应被清空。
	if _, err := os.Stat(filepath.Join(pool, "yolov8n", "20260810")); !os.IsNotExist(err) {
		t.Error("移动后源日期目录应被删除")
	}

	yaml, err := os.ReadFile(filepath.Join(root, "yolov8n", "data.yaml"))
	if err != nil {
		t.Fatalf("data.yaml 未生成: %v", err)
	}
	if !strings.Contains(string(yaml), "nc: 1") || !strings.Contains(string(yaml), "person") {
		t.Errorf("data.yaml 内容异常:\n%s", yaml)
	}
	if _, err := os.Stat(res.Manifest); err != nil {
		t.Errorf("版本清单未生成: %v", err)
	}
}

func TestPromoteCopyKeepsPool(t *testing.T) {
	pool := t.TempDir()
	root := t.TempDir()
	seedPool(t, pool, "yolov8n-seg", "20260810", "a", true)

	res, err := Run(Options{
		PoolDir: pool, DatasetRoot: root,
		Classes: []string{"obj"}, Move: false, Now: fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 1 {
		t.Fatalf("want 1, got %d", res.Total)
	}
	// 复制模式下源文件仍在。
	if _, err := os.Stat(filepath.Join(pool, "yolov8n-seg", "20260810", "images", "a.jpg")); err != nil {
		t.Errorf("复制模式不应删除源文件: %v", err)
	}
}

func TestPromoteFilterByEngineAndDate(t *testing.T) {
	pool := t.TempDir()
	root := t.TempDir()
	seedPool(t, pool, "yolov8n", "20260810", "a", true)
	seedPool(t, pool, "yolov8n-seg", "20260810", "b", true)
	seedPool(t, pool, "yolov8n", "20260811", "c", true)

	res, err := Run(Options{
		PoolDir: pool, DatasetRoot: root,
		Engine: "yolov8n", Date: "20260810",
		Classes: []string{"x"}, Now: fixedNow,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 1 {
		t.Fatalf("过滤后应只提升 1 个，got %d (%+v)", res.Total, res.Promoted)
	}
}
