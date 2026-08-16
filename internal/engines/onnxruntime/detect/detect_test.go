package detect

import (
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

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

func TestParseOutputShapeDynamicBatch(t *testing.T) {
	// 动态 batch 维 <=0，官方排布 [-1,84,8400]
	anchors, attrs, tr, err := parseOutputShape(ort.NewShape(-1, 84, 8400), 80)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if anchors != 8400 || attrs != 84 || tr {
		t.Fatalf("got %d/%d/%v", anchors, attrs, tr)
	}
	// 动态 batch + 转置 [-1,8400,84]
	if anchors, attrs, tr, err = parseOutputShape(ort.NewShape(-1, 8400, 84), 80); err != nil || !tr || anchors != 8400 || attrs != 84 {
		t.Fatalf("got %d/%d/%v err=%v, want 8400/84/true", anchors, attrs, tr, err)
	}
	// 固定 N>1 [4,84,8400]
	if anchors, attrs, _, err = parseOutputShape(ort.NewShape(4, 84, 8400), 80); err != nil || anchors != 8400 || attrs != 84 {
		t.Fatalf("got %d/%d err=%v, want 8400/84", anchors, attrs, err)
	}
	// 非 batch 维不能动态
	if _, _, _, err := parseOutputShape(ort.NewShape(1, -1, 8400), 80); err == nil {
		t.Fatal("expected error for dynamic non-batch dim")
	}
}

func TestPostprocessSlotIsolation(t *testing.T) {
	// 合成一个 2 槽输出，官方排布 [B,attrs,anchors]，attrs=4+1=5(nc=1)，anchors=3。
	// 只在 slot 1 的 anchor 2 放一个高分框，断言 postprocess(0) 空、postprocess(1) 命中该框。
	e := &Engine{
		outAttrs: 5, outAnchors: 3, outTransposed: false,
		cfg:       Config{Classes: []string{"obj"}, ConfThresh: 0.25, IoUThresh: 0.45},
		outStride: 5 * 3,
	}
	buf := make([]float32, 2*5*3)
	// slot 1, anchor 2: cx=10 cy=20 w=4 h=6, class0=0.9
	base := 1*e.outStride + 2 // cx 偏移 (base0 + 0*anchors + a)
	buf[base] = 10
	buf[base+e.outAnchors] = 20
	buf[base+2*e.outAnchors] = 4
	buf[base+3*e.outAnchors] = 6
	buf[base+4*e.outAnchors] = 0.9
	v := &detectView{buf: buf, stride: e.outStride}

	if boxes := e.postprocess(v, 0, 0.25); len(boxes) != 0 {
		t.Fatalf("slot 0 should be empty, got %d", len(boxes))
	}
	boxes := e.postprocess(v, 1, 0.25)
	if len(boxes) != 1 {
		t.Fatalf("slot 1 want 1 box, got %d", len(boxes))
	}
	b := boxes[0]
	if b.ClassID != 0 || b.Confidence < 0.89 || b.X1 != 8 || b.Y1 != 17 || b.X2 != 12 || b.Y2 != 23 {
		t.Fatalf("unexpected box: %+v", b)
	}
}
