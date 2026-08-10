package metric

import (
	"image"
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
