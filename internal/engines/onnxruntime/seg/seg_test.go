package seg

import (
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestParseDetShape(t *testing.T) {
	// 官方排布 [1, 4+nc+nm, anchors]，nc=80, nm=32 → 116
	anchors, attrs, nm, tr, err := parseDetShape(ort.NewShape(1, 116, 8400), 80)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if anchors != 8400 || attrs != 116 || nm != 32 || tr {
		t.Fatalf("got %d/%d/%d/%v, want 8400/116/32/false", anchors, attrs, nm, tr)
	}
	// 转置排布 [1, anchors, 116]
	anchors, attrs, nm, tr, err = parseDetShape(ort.NewShape(1, 8400, 116), 80)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !tr || anchors != 8400 || attrs != 116 || nm != 32 {
		t.Fatalf("got %d/%d/%d/%v, want 8400/116/32/true", anchors, attrs, nm, tr)
	}
	// 类别不匹配
	if _, _, _, _, err := parseDetShape(ort.NewShape(1, 5, 8400), 80); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

func TestParseShapesDynamicBatch(t *testing.T) {
	// 动态 batch 维：[-1,116,8400] 与 [-1,8400,116]
	if a, attr, nm, tr, err := parseDetShape(ort.NewShape(-1, 116, 8400), 80); err != nil || a != 8400 || attr != 116 || nm != 32 || tr {
		t.Fatalf("dynamic official: got %d/%d/%d/%v %v", a, attr, nm, tr, err)
	}
	if a, attr, nm, tr, err := parseDetShape(ort.NewShape(-1, 8400, 116), 80); err != nil || a != 8400 || attr != 116 || nm != 32 || !tr {
		t.Fatalf("dynamic transposed: got %d/%d/%d/%v %v", a, attr, nm, tr, err)
	}
	// 固定 N>1 + 动态 proto
	if a, attr, _, _, err := parseDetShape(ort.NewShape(4, 116, 8400), 80); err != nil || a != 8400 || attr != 116 {
		t.Fatalf("fixed N=4: got %d/%d %v", a, attr, err)
	}
	if mh, mw, err := parseProtoShape(ort.NewShape(-1, 32, 160, 160), 32); err != nil || mh != 160 || mw != 160 {
		t.Fatalf("dynamic proto: got %dx%d %v", mh, mw, err)
	}
	// 非 batch 维动态应报错
	if _, _, _, _, err := parseDetShape(ort.NewShape(1, -1, 8400), 80); err == nil {
		t.Fatal("expected error for dynamic non-batch det dim")
	}
}

func TestDecodeDetectionsSlotIsolation(t *testing.T) {
	// 官方排布 [B,attrs,anchors]，nc=1,nm=2 → attrs=7，anchors=2，两槽。
	// 仅 slot1/anchor1 放一个高分项，slot0 必须为空。
	e := &Engine{
		attrs: 7, anchors: 2, nc: 1, nm: 2, transposed: false,
		detStride: 7 * 2,
		detBuf:    make([]float32, 2*7*2),
		cfg:       Config{ConfThresh: 0.25},
	}
	base := 1*e.detStride + 1
	e.detBuf[base] = 10              // cx
	e.detBuf[base+e.anchors] = 20    // cy
	e.detBuf[base+2*e.anchors] = 4   // w
	e.detBuf[base+3*e.anchors] = 6   // h
	e.detBuf[base+4*e.anchors] = 0.9 // class0
	e.detBuf[base+5*e.anchors] = 0.1 // coeff0
	e.detBuf[base+6*e.anchors] = 0.2 // coeff1

	if c := e.decodeDetections(0, 0.25); len(c) != 0 {
		t.Fatalf("slot 0 should be empty, got %d", len(c))
	}
	c := e.decodeDetections(1, 0.25)
	if len(c) != 1 {
		t.Fatalf("slot 1 want 1 cand, got %d", len(c))
	}
	if c[0].box.ClassID != 0 || c[0].box.Confidence < 0.89 || c[0].box.X1 != 8 || c[0].box.Y1 != 17 {
		t.Fatalf("unexpected box: %+v", c[0].box)
	}
	if len(c[0].coeffs) != 2 || c[0].coeffs[0] != 0.1 || c[0].coeffs[1] != 0.2 {
		t.Fatalf("coeffs not copied out: %v", c[0].coeffs)
	}
}

func TestParseProtoShape(t *testing.T) {
	mh, mw, err := parseProtoShape(ort.NewShape(1, 32, 160, 160), 32)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if mh != 160 || mw != 160 {
		t.Fatalf("got %dx%d, want 160x160", mh, mw)
	}
	if _, _, err := parseProtoShape(ort.NewShape(1, 16, 160, 160), 32); err == nil {
		t.Fatal("expected nm mismatch error")
	}
}

func TestContourSquare(t *testing.T) {
	// 5x5 掩膜，中心 3x3 实心方块
	w, h := 5, 5
	mask := make([]bool, w*h)
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			mask[y*w+x] = true
		}
	}
	polys := contour(mask, w, h)
	if len(polys) != 1 {
		t.Fatalf("expected 1 contour, got %d", len(polys))
	}
	// 外轮廓应经过方块的 8 个角点（追踪方向不限）。
	if len(polys[0]) < 4 {
		t.Fatalf("contour too short: %d points", len(polys[0]))
	}
}

func TestContourEmpty(t *testing.T) {
	mask := make([]bool, 4*4)
	if polys := contour(mask, 4, 4); len(polys) != 0 {
		t.Fatalf("expected 0 contours, got %d", len(polys))
	}
}
