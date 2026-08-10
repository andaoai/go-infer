package detect

import (
	"testing"

	"github.com/andaoai/go-infer/internal/engine"
	ort "github.com/yalue/onnxruntime_go"
)

func TestNMS(t *testing.T) {
	boxes := []engine.Box{
		{ClassID: 0, Confidence: 0.9, X1: 0, Y1: 0, X2: 10, Y2: 10},
		{ClassID: 0, Confidence: 0.8, X1: 1, Y1: 1, X2: 11, Y2: 11},
		{ClassID: 1, Confidence: 0.7, X1: 0, Y1: 0, X2: 10, Y2: 10},
	}
	keep := nms(boxes, 0.5)
	if len(keep) != 2 {
		t.Fatalf("expected 2 kept, got %d", len(keep))
	}
	if keep[0].Confidence != 0.9 {
		t.Fatalf("expected highest conf first, got %f", keep[0].Confidence)
	}
}

func TestIoU(t *testing.T) {
	a := engine.Box{X1: 0, Y1: 0, X2: 10, Y2: 10}
	b := engine.Box{X1: 5, Y1: 5, X2: 15, Y2: 15}
	got := iou(a, b)
	if got < 0.142 || got > 0.144 {
		t.Fatalf("iou got %f, want ~0.1428", got)
	}
}

func TestParseOutputShape(t *testing.T) {
	// 官方 YOLOv8 导出 [1,4+nc,anchors], nc=80 → 84
	anchors, attrs, tr, err := parseOutputShape(ort.NewShape(1, 84, 8400), 80)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if anchors != 8400 || attrs != 84 || tr {
		t.Fatalf("got %d/%d/%v, want 8400/84/false", anchors, attrs, tr)
	}
	// 转置 [1,anchors,attrs]
	anchors, attrs, tr, err = parseOutputShape(ort.NewShape(1, 8400, 84), 80)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !tr || anchors != 8400 || attrs != 84 {
		t.Fatalf("got %d/%d/%v, want 8400/84/true", anchors, attrs, tr)
	}
	// 类别不匹配
	if _, _, _, err := parseOutputShape(ort.NewShape(1, 12, 8400), 80); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}
