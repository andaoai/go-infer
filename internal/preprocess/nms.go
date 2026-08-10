package preprocess

import "sort"

// NMSBox 是 NMS 操作的最小输入：轴对齐框 + 类别 + 置信度。
type NMSBox struct {
	ClassID    int
	Confidence float32
	X1, Y1     float32
	X2, Y2     float32
}

// NMS 按置信度降序做按类分组的非极大值抑制。
func NMS(boxes []NMSBox, iouThresh float32) []NMSBox {
	sort.SliceStable(boxes, func(i, j int) bool {
		return boxes[i].Confidence > boxes[j].Confidence
	})
	keep := make([]NMSBox, 0, len(boxes))
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
			if IoU(boxes[i], boxes[j]) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

// IoU 计算两个轴对齐框的交并比。
func IoU(a, b NMSBox) float32 {
	ix1 := MaxF(a.X1, b.X1)
	iy1 := MaxF(a.Y1, b.Y1)
	ix2 := MinF(a.X2, b.X2)
	iy2 := MinF(a.Y2, b.Y2)
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
