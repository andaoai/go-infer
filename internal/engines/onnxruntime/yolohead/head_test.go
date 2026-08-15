package yolohead

import (
	"math"
	"testing"
)

// 构造转置布局 [B,anchors,attrs] 的一批输出。
func makeTransposed(slots, anchors, attrs int, anchorData [][]float32) []float32 {
	buf := make([]float32, slots*anchors*attrs)
	for s := 0; s < slots; s++ {
		for a := 0; a < anchors; a++ {
			copy(buf[s*anchors*attrs+a*attrs:], anchorData[s*anchors+a])
		}
	}
	return buf
}

// 把同样的 anchor 数据填成非转置 [B,attrs,anchors]。
func makeNonTransposed(slots, anchors, attrs int, anchorData [][]float32) []float32 {
	buf := make([]float32, slots*anchors*attrs)
	for s := 0; s < slots; s++ {
		base := s * anchors * attrs
		for a := 0; a < anchors; a++ {
			for k := 0; k < attrs; k++ {
				buf[base+k*anchors+a] = anchorData[s*anchors+a][k]
			}
		}
	}
	return buf
}

type hit struct {
	a, cls              int
	cx, cy, w, h, score float32
	coeffs              []float32
}

func collect(t *testing.T, buf []float32, slot, stride, anchors, attrs, nc int, transposed bool, conf float32) []hit {
	t.Helper()
	var got []hit
	Walk(buf, slot, stride, anchors, attrs, nc, transposed, conf,
		func(a, cls int, cx, cy, w, h, score float32, coeffs []float32) {
			cp := make([]float32, len(coeffs))
			copy(cp, coeffs)
			got = append(got, hit{a, cls, cx, cy, w, h, score, cp})
		})
	return got
}

func approxEqual(a, b []hit) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		x, y := a[i], b[i]
		if x.a != y.a || x.cls != y.cls {
			return false
		}
		for _, pair := range [][2]float32{{x.cx, y.cx}, {x.cy, y.cy}, {x.w, y.w}, {x.h, y.h}, {x.score, y.score}} {
			if math.Abs(float64(pair[0]-pair[1])) > 1e-6 {
				return false
			}
		}
		if len(x.coeffs) != len(y.coeffs) {
			return false
		}
		for k := range x.coeffs {
			if math.Abs(float64(x.coeffs[k]-y.coeffs[k])) > 1e-6 {
				return false
			}
		}
	}
	return true
}

func TestWalkDetectBothLayouts(t *testing.T) {
	const slots, anchors, attrs, nc = 2, 3, 5, 1
	// 每 anchor: cx, cy, w, h, cls0
	data := [][]float32{
		{1, 2, 3, 4, 0.9},      // slot0 a0
		{5, 6, 7, 8, 0.1},      // slot0 a1 (below conf)
		{9, 10, 11, 12, 0.7},   // slot0 a2
		{13, 14, 15, 16, 0.5},  // slot1 a0
		{17, 18, 19, 20, 0.01}, // slot1 a1
		{21, 22, 23, 24, 0.95}, // slot1 a2
	}
	trans := makeTransposed(slots, anchors, attrs, data)
	non := makeNonTransposed(slots, anchors, attrs, data)
	stride := anchors * attrs

	// slot0 应回调 a0(cls0,.9) 和 a2(cls0,.7)
	want0 := []hit{
		{0, 0, 1, 2, 3, 4, 0.9, nil},
		{2, 0, 9, 10, 11, 12, 0.7, nil},
	}
	g0t := collect(t, trans, 0, stride, anchors, attrs, nc, true, 0.25)
	g0n := collect(t, non, 0, stride, anchors, attrs, nc, false, 0.25)
	if !approxEqual(want0, g0t) {
		t.Fatalf("transposed slot0 mismatch:\n got %+v\nwant %+v", g0t, want0)
	}
	if !approxEqual(want0, g0n) {
		t.Fatalf("non-transposed slot0 mismatch:\n got %+v\nwant %+v", g0n, want0)
	}

	// slot1: a0(.5), a2(.95) (a1 被阈值过滤)
	want1 := []hit{
		{0, 0, 13, 14, 15, 16, 0.5, nil},
		{2, 0, 21, 22, 23, 24, 0.95, nil},
	}
	g1t := collect(t, trans, 1, stride, anchors, attrs, nc, true, 0.25)
	g1n := collect(t, non, 1, stride, anchors, attrs, nc, false, 0.25)
	if !approxEqual(want1, g1t) {
		t.Fatalf("transposed slot1 mismatch:\n got %+v\nwant %+v", g1t, want1)
	}
	if !approxEqual(want1, g1n) {
		t.Fatalf("non-transposed slot1 mismatch:\n got %+v\nwant %+v", g1n, want1)
	}
}

func TestWalkSegCoeffsBothLayouts(t *testing.T) {
	const slots, anchors, attrs, nc = 1, 2, 7, 1 // nm = 2
	data := [][]float32{
		{1, 2, 3, 4, 0.9, 0.11, 0.22},
		{5, 6, 7, 8, 0.3, 0.33, 0.44},
	}
	trans := makeTransposed(slots, anchors, attrs, data)
	non := makeNonTransposed(slots, anchors, attrs, data)
	stride := anchors * attrs
	want := []hit{
		{0, 0, 1, 2, 3, 4, 0.9, []float32{0.11, 0.22}},
		{1, 0, 5, 6, 7, 8, 0.3, []float32{0.33, 0.44}},
	}
	gt := collect(t, trans, 0, stride, anchors, attrs, nc, true, 0.25)
	gn := collect(t, non, 0, stride, anchors, attrs, nc, false, 0.25)
	if !approxEqual(want, gt) {
		t.Fatalf("trans coeffs mismatch:\n got %+v\nwant %+v", gt, want)
	}
	if !approxEqual(want, gn) {
		t.Fatalf("non-trans coeffs mismatch:\n got %+v\nwant %+v", gn, want)
	}
}

func TestWalkMultiClassPicksMax(t *testing.T) {
	const anchors, attrs, nc = 1, 6, 2 // cx cy w h cls0 cls1
	data := [][]float32{{10, 20, 30, 40, 0.4, 0.8}}
	trans := makeTransposed(1, anchors, attrs, data)
	non := makeNonTransposed(1, anchors, attrs, data)
	want := []hit{{0, 1, 10, 20, 30, 40, 0.8, nil}}
	if got := collect(t, trans, 0, anchors*attrs, anchors, attrs, nc, true, 0.25); !approxEqual(want, got) {
		t.Fatalf("trans: got %+v want %+v", got, want)
	}
	if got := collect(t, non, 0, anchors*attrs, anchors, attrs, nc, false, 0.25); !approxEqual(want, got) {
		t.Fatalf("non: got %+v want %+v", got, want)
	}
}
