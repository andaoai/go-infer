package dataset

import (
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"
)

// 构造一个临时 YOLO 数据集：1 张图 + det 标签和 seg 标签各验证解析。
func writeTmpDataset(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, sub := range []string{"images/val", "labels/val"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	// 写一张 100x200 的图。
	img := image.NewRGBA(image.Rect(0, 0, 100, 200))
	for y := 0; y < 200; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 128, A: 255})
		}
	}
	f, err := os.Create(filepath.Join(dir, "images/val/sample.jpg"))
	if err != nil {
		t.Fatal(err)
	}
	if err := jpeg.Encode(f, img, nil); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return dir
}

func TestLoadDetectionLabel(t *testing.T) {
	dir := writeTmpDataset(t)
	// cx=0.5 cy=0.5 w=0.2 h=0.4 → 100x200 图上 [40,60,60,140]
	if err := os.WriteFile(filepath.Join(dir, "labels/val/sample.txt"),
		[]byte("0 0.5 0.5 0.2 0.4\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	samples, err := Load(dir, "val")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 1 {
		t.Fatalf("want 1 sample, got %d", len(samples))
	}
	s := samples[0]
	if s.W != 100 || s.H != 200 {
		t.Fatalf("size = %dx%d, want 100x200", s.W, s.H)
	}
	if len(s.Objects) != 1 {
		t.Fatalf("want 1 object, got %d", len(s.Objects))
	}
	g := s.Objects[0]
	if g.ClassID != 0 {
		t.Errorf("class = %d, want 0", g.ClassID)
	}
	if !near(g.X1, 40) || !near(g.Y1, 60) || !near(g.X2, 60) || !near(g.Y2, 140) {
		t.Errorf("box = [%.1f,%.1f,%.1f,%.1f], want [40,60,60,140]", g.X1, g.Y1, g.X2, g.Y2)
	}
	if len(g.Polygon) != 0 {
		t.Errorf("det label should have no polygon, got %d pts", len(g.Polygon))
	}
}

func TestLoadSegmentationLabel(t *testing.T) {
	dir := writeTmpDataset(t)
	// 一个三角多边形：(0.2,0.1) (0.8,0.1) (0.5,0.9) → 像素
	// x: 20,80,50  y: 20,20,180
	if err := os.WriteFile(filepath.Join(dir, "labels/val/sample.txt"),
		[]byte("3 0.2 0.1 0.8 0.1 0.5 0.9\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	samples, err := Load(dir, "val")
	if err != nil {
		t.Fatal(err)
	}
	g := samples[0].Objects[0]
	if g.ClassID != 3 {
		t.Errorf("class = %d, want 3", g.ClassID)
	}
	// bbox 由点列 min/max 得出
	if !near(g.X1, 20) || !near(g.Y1, 20) || !near(g.X2, 80) || !near(g.Y2, 180) {
		t.Errorf("box = [%.1f,%.1f,%.1f,%.1f], want [20,20,80,180]", g.X1, g.Y1, g.X2, g.Y2)
	}
	if len(g.Polygon) != 3 {
		t.Fatalf("polygon pts = %d, want 3", len(g.Polygon))
	}
	if g.Polygon[0] != (image.Point{X: 20, Y: 20}) {
		t.Errorf("first point = %v, want (20,20)", g.Polygon[0])
	}
}

func TestMissingLabelIsNegativeSample(t *testing.T) {
	dir := writeTmpDataset(t)
	samples, err := Load(dir, "val")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples[0].Objects) != 0 {
		t.Errorf("missing label should yield 0 objects, got %d", len(samples[0].Objects))
	}
}

func near(a, b float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 0.5
}
