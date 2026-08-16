package metric

import (
	"image"
	"math/rand"
	"testing"

	"github.com/andaoai/go-infer/internal/data"
)

func TestBoxIoU(t *testing.T) {
	a := data.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}
	if got := BoxIoU(a, a); got < 0.999 {
		t.Errorf("self IoU = %.3f, want 1.0", got)
	}
	b := data.BBox{X1: 5, Y1: 0, X2: 15, Y2: 10}
	if got := BoxIoU(a, b); !about(got, 1.0/3.0, 1e-6) {
		t.Errorf("half-overlap IoU = %.3f, want %.3f", got, 1.0/3.0)
	}
	c := data.BBox{X1: 100, Y1: 100, X2: 110, Y2: 110}
	if got := BoxIoU(a, c); got != 0 {
		t.Errorf("disjoint IoU = %.3f, want 0", got)
	}
}

func TestRasterizeRectangleAndIoU(t *testing.T) {
	sq := func(x0, y0, x1, y1 int) []image.Point {
		return []image.Point{
			image.Pt(x0, y0), image.Pt(x1, y0),
			image.Pt(x1, y1), image.Pt(x0, y1),
		}
	}
	a := sq(0, 0, 10, 10)
	m := Rasterize(a, 20, 20)
	n := 0
	for _, v := range m {
		if v {
			n++
		}
	}
	if n != 100 {
		t.Errorf("10x10 rect rasterized area = %d, want 100", n)
	}
	if got := MaskIoU(a, sq(5, 0, 15, 10), 20, 20); !about(got, 1.0/3.0, 0.02) {
		t.Errorf("mask IoU = %.3f, want ~%.3f", got, 1.0/3.0)
	}
	if got := MaskIoU(a, sq(50, 50, 60, 60), 70, 70); got != 0 {
		t.Errorf("disjoint mask IoU = %.3f, want 0", got)
	}
}

func TestAPPerfectPrediction(t *testing.T) {
	gts := []GroundTruth{
		{ImageID: 0, Object: data.Object{ClassID: 0, BBox: data.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}}, W: 100, H: 100},
		{ImageID: 1, Object: data.Object{ClassID: 0, BBox: data.BBox{X1: 20, Y1: 20, X2: 30, Y2: 30}}, W: 100, H: 100},
	}
	preds := []Prediction{
		{ImageID: 0, Object: data.Object{ClassID: 0, Confidence: 0.9, BBox: data.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}}},
		{ImageID: 1, Object: data.Object{ClassID: 0, Confidence: 0.8, BBox: data.BBox{X1: 20, Y1: 20, X2: 30, Y2: 30}}},
	}
	_, mAP := AP(preds, gts, 0.5, false)
	if !about(mAP, 1.0, 1e-6) {
		t.Errorf("perfect mAP = %.3f, want 1.0", mAP)
	}
}

func TestAPAllWrong(t *testing.T) {
	gts := []GroundTruth{
		{ImageID: 0, Object: data.Object{ClassID: 0, BBox: data.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}}, W: 100, H: 100},
	}
	preds := []Prediction{
		{ImageID: 0, Object: data.Object{ClassID: 0, Confidence: 0.9, BBox: data.BBox{X1: 50, Y1: 50, X2: 60, Y2: 60}}},
	}
	_, mAP := AP(preds, gts, 0.5, false)
	if mAP != 0 {
		t.Errorf("all-wrong mAP = %.3f, want 0", mAP)
	}
}

func TestAPFalsePositive(t *testing.T) {
	gts := []GroundTruth{
		{ImageID: 0, Object: data.Object{ClassID: 0, BBox: data.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}}, W: 100, H: 100},
	}
	preds := []Prediction{
		{ImageID: 0, Object: data.Object{ClassID: 0, Confidence: 0.9, BBox: data.BBox{X1: 50, Y1: 50, X2: 60, Y2: 60}}},
		{ImageID: 0, Object: data.Object{ClassID: 0, Confidence: 0.8, BBox: data.BBox{X1: 0, Y1: 0, X2: 10, Y2: 10}}},
	}
	_, mAP := AP(preds, gts, 0.5, false)
	if mAP <= 0 || mAP >= 1 {
		t.Errorf("high-FP mAP = %.3f, want in (0,1)", mAP)
	}
}

func about(a, b, eps float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < eps
}

// TestEvaluateMatchesAPAndMAPOverThresholds 验证统一 Evaluate 与历史的
// AP + MAPOverThresholds 在 box 和 mask 两种匹配下逐位一致。
func TestEvaluateMatchesAPAndMAPOverThresholds(t *testing.T) {
	rng := rand.New(rand.NewSource(2024))
	mkBox := func(x, y, s int) data.BBox {
		return data.BBox{X1: float32(x), Y1: float32(y), X2: float32(x + s), Y2: float32(y + s)}
	}
	mkRing := func(x, y, s int) [][]data.Point {
		return [][]data.Point{{
			{X: float32(x), Y: float32(y)},
			{X: float32(x + s), Y: float32(y)},
			{X: float32(x + s), Y: float32(y + s)},
			{X: float32(x), Y: float32(y + s)},
		}}
	}
	for iter := 0; iter < 100; iter++ {
		imgs := 2 + rng.Intn(4)
		var gts []GroundTruth
		var preds []Prediction
		for img := 0; img < imgs; img++ {
			nGT := rng.Intn(5)
			for i := 0; i < nGT; i++ {
				cls := rng.Intn(3)
				x, y, s := rng.Intn(80), rng.Intn(80), 10+rng.Intn(20)
				gts = append(gts, GroundTruth{
					ImageID: img, W: 100, H: 100,
					Object: data.Object{ClassID: cls, BBox: mkBox(x, y, s), Rings: mkRing(x, y, s)},
				})
			}
			nP := rng.Intn(6)
			for i := 0; i < nP; i++ {
				cls := rng.Intn(3)
				x, y, s := rng.Intn(80), rng.Intn(80), 10+rng.Intn(20)
				jx, jy := x, y
				if rng.Intn(2) == 0 {
					jx += rng.Intn(6) - 3
				}
				preds = append(preds, Prediction{
					ImageID: img,
					Object: data.Object{
						ClassID: cls, Confidence: float32(rng.Float64()),
						BBox: mkBox(jx, jy, s), Rings: mkRing(jx, jy, s),
					},
				})
			}
		}
		for _, useMask := range []bool{false, true} {
			er := Evaluate(preds, gts, useMask)
			apClasses, ap50 := AP(preds, gts, 0.50, useMask)
			map50Leg, map5095Leg := MAPOverThresholds(preds, gts, useMask)
			if !about(er.MAP50, ap50, 1e-9) {
				t.Fatalf("iter %d mask=%v: MAP50 %.6f != AP.mAP %.6f", iter, useMask, er.MAP50, ap50)
			}
			if !about(er.MAP50, map50Leg, 1e-9) || !about(er.MAP5095, map5095Leg, 1e-9) {
				t.Fatalf("iter %d mask=%v: Evaluate(%g,%g) != MAPOverThresholds(%g,%g)",
					iter, useMask, er.MAP50, er.MAP5095, map50Leg, map5095Leg)
			}
			for c, ap := range apClasses {
				if !about(er.PerClass[c], ap, 1e-9) {
					t.Fatalf("iter %d mask=%v class %d: PerClass %.6f != AP %.6f",
						iter, useMask, c, er.PerClass[c], ap)
				}
			}
		}
	}
}
