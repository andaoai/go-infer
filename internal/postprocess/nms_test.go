package postprocess

import (
	"testing"

	"github.com/andaoai/go-infer/internal/engine"
)

func TestNMS(t *testing.T) {
	boxes := []engine.Box{
		{ClassID: 0, Confidence: 0.9, X1: 0, Y1: 0, X2: 10, Y2: 10},
		{ClassID: 0, Confidence: 0.8, X1: 1, Y1: 1, X2: 11, Y2: 11},
		{ClassID: 1, Confidence: 0.7, X1: 0, Y1: 0, X2: 10, Y2: 10},
	}
	keep := NMS(boxes, 0.5)
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
	got := IoU(a, b)
	if got < 0.142 || got > 0.144 {
		t.Fatalf("iou got %f, want ~0.1428", got)
	}
}

func TestNMSNoMutation(t *testing.T) {
	// NMS 不应重排输入切片（detect/seg 还会按原始锚点顺序访问其他关联数据）。
	boxes := []engine.Box{
		{ClassID: 0, Confidence: 0.5, X1: 0, Y1: 0, X2: 2, Y2: 2},
		{ClassID: 0, Confidence: 0.9, X1: 0, Y1: 0, X2: 10, Y2: 10},
	}
	_ = NMS(boxes, 0.5)
	if boxes[0].Confidence != 0.5 || boxes[1].Confidence != 0.9 {
		t.Fatalf("NMS mutated input order: %+v", boxes)
	}
}
