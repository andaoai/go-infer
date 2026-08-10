package dataset

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/andaoai/go-infer/internal/engine"
)

func TestWriteAndReadBoxLabel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "labels", "a.txt")
	dets := []engine.Detection{
		{ClassID: 0, X1: 40, Y1: 60, X2: 60, Y2: 140}, // 100x200 图：cx=50 cy=100 w=20 h=80
	}
	if err := WriteBoxLabel(path, 100, 200, dets); err != nil {
		t.Fatal(err)
	}
	objs, err := loadLabels(path, 100, 200)
	if err != nil {
		t.Fatal(err)
	}
	if len(objs) != 1 {
		t.Fatalf("want 1 obj, got %d", len(objs))
	}
	o := objs[0]
	if o.ClassID != 0 {
		t.Errorf("class = %d", o.ClassID)
	}
	for _, got := range []float32{o.X1, o.Y1, o.X2, o.Y2} {
		if got != got { // NaN check
			t.Fatal("NaN in coords")
		}
	}
	if !near(o.X1, 40) || !near(o.Y1, 60) || !near(o.X2, 60) || !near(o.Y2, 140) {
		t.Errorf("bbox roundtrip got [%.1f %.1f %.1f %.1f]", o.X1, o.Y1, o.X2, o.Y2)
	}
}

func TestWriteAndReadSegLabel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "labels", "a.txt")
	insts := []engine.Instance{
		{Detection: engine.Detection{ClassID: 2}, Mask: [][]engine.Point{
			{{X: 20, Y: 20}, {X: 80, Y: 20}, {X: 80, Y: 180}, {X: 20, Y: 180}},
		}},
	}
	if err := WriteSegLabel(path, 100, 200, insts); err != nil {
		t.Fatal(err)
	}
	objs, err := loadLabels(path, 100, 200)
	if err != nil {
		t.Fatal(err)
	}
	if len(objs) != 1 || objs[0].ClassID != 2 {
		t.Fatalf("want 1 obj class 2, got %+v", objs)
	}
	if len(objs[0].Polygon) != 4 {
		t.Fatalf("want 4 polygon pts, got %d", len(objs[0].Polygon))
	}
	if objs[0].X1 != 20 || objs[0].Y1 != 20 || objs[0].X2 != 80 || objs[0].Y2 != 180 {
		t.Errorf("bbox from polygon = [%.0f %.0f %.0f %.0f]", objs[0].X1, objs[0].Y1, objs[0].X2, objs[0].Y2)
	}
}

func TestSaveImagePreservesBytes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "i.jpg")
	want := []byte{0xFF, 0xD8, 0xFF, 1, 2, 3}
	if err := SaveImage(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Errorf("bytes differ: got %v want %v", got, want)
	}
}
