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
