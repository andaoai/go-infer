package detect

import (
	"sort"

	"github.com/andaoai/go-infer/internal/engine"
)

// nms 按置信度降序做按类分组的非极大值抑制。
func nms(boxes []engine.Box, iouThresh float32) []engine.Box {
	sort.SliceStable(boxes, func(i, j int) bool {
		return boxes[i].Confidence > boxes[j].Confidence
	})
	keep := make([]engine.Box, 0, len(boxes))
	suppressed := make([]bool, len(boxes))
	for i := 0; i < len(boxes); i++ {
		if suppressed[i] {
			continue
		}
		keep = append(keep, boxes[i])
		for j := i + 1; j < len(boxes); j++ {
			if suppressed[j] || boxes[j].ClassID != boxes[i].ClassID {
				continue
			}
			if iou(boxes[i], boxes[j]) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func iou(a, b engine.Box) float32 {
	ix1 := maxF(a.X1, b.X1)
	iy1 := maxF(a.Y1, b.Y1)
	ix2 := minF(a.X2, b.X2)
	iy2 := minF(a.Y2, b.Y2)
	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	areaA := (a.X2 - a.X1) * (a.Y2 - a.Y1)
	areaB := (b.X2 - b.X1) * (b.Y2 - b.Y1)
	return inter / (areaA + areaB - inter + 1e-7)
}

func maxF(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
func minF(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
