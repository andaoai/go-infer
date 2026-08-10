package yolo

import (
	"context"
	"strings"
	"testing"

	"github.com/andaoai/go-infer/internal/data"
	"github.com/andaoai/go-infer/internal/storage/local"
)

func TestEncodeDecodeBox(t *testing.T) {
	c := New()
	objs := []data.Object{
		{ClassID: 0, BBox: data.BBox{X1: 40, Y1: 60, X2: 60, Y2: 140}}, // 100x200
	}
	b, err := c.Encode(objs, 100, 200)
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.Decode(b, 100, 200)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1, got %d", len(got))
	}
	o := got[0]
	if o.ClassID != 0 || !near(o.BBox.X1, 40) || !near(o.BBox.Y1, 60) || !near(o.BBox.X2, 60) || !near(o.BBox.Y2, 140) {
		t.Errorf("bbox roundtrip = %+v", o.BBox)
	}
}

func TestEncodeDecodeSeg(t *testing.T) {
	c := New()
	objs := []data.Object{
		{ClassID: 2, BBox: data.BBox{X1: 20, Y1: 20, X2: 80, Y2: 180}, Rings: [][]data.Point{
			{{X: 20, Y: 20}, {X: 80, Y: 20}, {X: 80, Y: 180}, {X: 20, Y: 180}},
		}},
	}
	b, err := c.Encode(objs, 100, 200)
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.Decode(b, 100, 200)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ClassID != 2 {
		t.Fatalf("want 1 obj class 2, got %+v", got)
	}
	if len(got[0].Rings) != 1 || len(got[0].Rings[0]) != 4 {
		t.Fatalf("want 4-pt ring, got %+v", got[0].Rings)
	}
	bb := got[0].BBox
	if bb.X1 != 20 || bb.Y1 != 20 || bb.X2 != 80 || bb.Y2 != 180 {
		t.Errorf("bbox from polygon = %+v", bb)
	}
}

func TestDecodeEmptyIsNegative(t *testing.T) {
	c := New()
	got, err := c.Decode(nil, 100, 100)
	if err != nil || len(got) != 0 {
		t.Fatalf("empty label should be negative sample, got %+v err=%v", got, err)
	}
}

func TestLabelKey(t *testing.T) {
	c := New()
	cases := map[string]string{
		"root/images/train/a.jpg": "root/labels/train/a.txt",
		"root/images/val/x_y.png": "root/labels/val/x_y.txt",
	}
	for in, want := range cases {
		if got := c.LabelKey(in); got != want {
			t.Errorf("LabelKey(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestListImages(t *testing.T) {
	ctx := context.Background()
	st, err := local.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	_ = st.Put(ctx, "ds/images/train/a.jpg", strings.NewReader("img"))
	_ = st.Put(ctx, "ds/images/train/b.jpg", strings.NewReader("img"))
	_ = st.Put(ctx, "ds/labels/train/a.txt", strings.NewReader("0 0.5 0.5 0.2 0.2"))
	// b 无标签（负样本）

	c := New()
	refs, err := c.ListImages(ctx, st, "ds", "train")
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 2 {
		t.Fatalf("want 2 refs, got %d", len(refs))
	}
	if refs[0].ImageKey != "ds/images/train/a.jpg" || refs[0].LabelKey != "ds/labels/train/a.txt" {
		t.Errorf("ref0 = %+v", refs[0])
	}
}

func TestWriteDatasetConfig(t *testing.T) {
	ctx := context.Background()
	st, _ := local.New(t.TempDir())
	c := New()
	if err := c.WriteDatasetConfig(ctx, st, "ds/yolov8n", "/abs/ds/yolov8n", []string{"person", "car"}); err != nil {
		t.Fatal(err)
	}
	rc, err := st.Get(ctx, "ds/yolov8n/data.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer rc.Close()
	buf := make([]byte, 1024)
	n, _ := rc.Read(buf)
	out := string(buf[:n])
	if !strings.Contains(out, "nc: 2") || !strings.Contains(out, "person") || !strings.Contains(out, "images/train") ||
		!strings.Contains(out, "path: /abs/ds/yolov8n") {
		t.Errorf("data.yaml 内容异常:\n%s", out)
	}
}

func near(a, b float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 0.5
}
