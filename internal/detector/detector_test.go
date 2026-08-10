package detector

import (
	"image"
	"testing"

	"github.com/yalue/onnxruntime_go"
)

func TestParseOutputShape(t *testing.T) {
	// 官方 YOLOv8 导出: [1, 4+nc, 8400]，nc=80 → 84
	anchors, attrs, tr, err := parseOutputShape(onnxruntime_go.NewShape(1, 84, 8400), 80)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if anchors != 8400 || attrs != 84 || tr {
		t.Fatalf("got anchors=%d attrs=%d transposed=%v, want 8400/84/false", anchors, attrs, tr)
	}

	// 转置排布: [1, 8400, 84]
	anchors, attrs, tr, err = parseOutputShape(onnxruntime_go.NewShape(1, 8400, 84), 80)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if anchors != 8400 || attrs != 84 || !tr {
		t.Fatalf("got anchors=%d attrs=%d transposed=%v, want 8400/84/true", anchors, attrs, tr)
	}

	// 类别数不匹配
	if _, _, _, err := parseOutputShape(onnxruntime_go.NewShape(1, 12, 8400), 80); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

func TestNMS(t *testing.T) {
	// 两个同类高重叠框 + 一个异类框
	dets := []rawDet{
		{classID: 0, conf: 0.9, x1: 0, y1: 0, x2: 10, y2: 10},
		{classID: 0, conf: 0.8, x1: 1, y1: 1, x2: 11, y2: 11}, // 与第一个 IoU 高
		{classID: 1, conf: 0.7, x1: 0, y1: 0, x2: 10, y2: 10}, // 异类，应保留
	}
	keep := nms(dets, 0.5)
	if len(keep) != 2 {
		t.Fatalf("expected 2 kept, got %d", len(keep))
	}
	if keep[0].conf != 0.9 {
		t.Fatalf("expected highest conf first, got %f", keep[0].conf)
	}
}

func TestIOU(t *testing.T) {
	a := rawDet{x1: 0, y1: 0, x2: 10, y2: 10}
	b := rawDet{x1: 5, y1: 5, x2: 15, y2: 15}
	got := iou(a, b)
	// 交集 25，并集 175 → 1/7
	if got < 0.142 || got > 0.144 {
		t.Fatalf("iou got %f, want ~0.1428", got)
	}
}

func TestLetterboxSquare(t *testing.T) {
	// 200x100 的图 letterbox 到 100x100：等比缩放为 100x50，上下各填 25
	src := image.NewRGBA(image.Rect(0, 0, 200, 100))
	lb, err := letterbox(src, 100, 100)
	if err != nil {
		t.Fatal(err)
	}
	if lb.scale != 0.5 {
		t.Fatalf("scale got %f, want 0.5", lb.scale)
	}
	if lb.padX != 0 || lb.padY != 25 {
		t.Fatalf("pad got (%f,%f), want (0,25)", lb.padX, lb.padY)
	}
}

func TestNCHWNormalization(t *testing.T) {
	// 2x2 纯白图，NCHW 三个通道都应是 1.0
	w, h := 2, 2
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i], src.Pix[i+1], src.Pix[i+2] = 255, 255, 255
		src.Pix[i+3] = 255
	}
	dst := make([]float32, 3*w*h)
	fillNCHW(dst, src, w, h)
	for i, v := range dst {
		if v != 1.0 {
			t.Fatalf("dst[%d]=%f, want 1.0", i, v)
		}
	}
}
