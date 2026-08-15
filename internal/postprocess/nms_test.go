package postprocess

import (
	"math/rand"
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

// TestNMSIndexedMatchesNMS 用确定性随机框验证 NMSIndexed 与 NMS 在 engine.Box 上
// 产生完全相同的保留集合（按下标映射回原框）。
func TestNMSIndexedMatchesNMS(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	for iter := 0; iter < 200; iter++ {
		n := 1 + rng.Intn(20)
		boxes := make([]engine.Box, n)
		for i := range boxes {
			x := float32(rng.Float64()) * 100
			y := float32(rng.Float64()) * 100
			w := float32(rng.Float64())*40 + 5
			h := float32(rng.Float64())*40 + 5
			boxes[i] = engine.Box{
				ClassID:    rng.Intn(3),
				Confidence: float32(rng.Float64()),
				X1:         x, Y1: y, X2: x + w, Y2: y + h,
			}
		}
		thresh := float32(0.3 + rng.Float64()*0.4)
		kept := NMS(boxes, thresh)
		idx := NMSIndexed(n,
			func(i int) float32 { return boxes[i].Confidence },
			func(i, j int) bool { return boxes[i].ClassID == boxes[j].ClassID },
			func(i, j int) float32 { return IoU(boxes[i], boxes[j]) },
			thresh,
		)
		if len(idx) != len(kept) {
			t.Fatalf("iter %d: kept %d boxes, indexed kept %d indices", iter, len(kept), len(idx))
		}
		for k, bi := range idx {
			if boxes[bi] != kept[k] {
				t.Fatalf("iter %d pos %d mismatch: indexed=%+v nms=%+v", iter, k, boxes[bi], kept[k])
			}
		}
	}
}
