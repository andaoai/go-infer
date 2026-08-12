// Package postprocess 提供与具体推理框架无关的检测后处理算法：
// IoU、按类非极大值抑制（NMS）。输入统一使用 engine.Box（模型空间坐标），
// 供各 ONNX/其他后端引擎复用，不再需要在框类型之间做转换。
package postprocess

import (
	"slices"

	"github.com/andaoai/go-infer/internal/engine"
)

// IoU 计算两个轴对齐框的交并比。
func IoU(a, b engine.Box) float32 {
	ix1 := maxf(a.X1, b.X1)
	iy1 := maxf(a.Y1, b.Y1)
	ix2 := minf(a.X2, b.X2)
	iy2 := minf(a.Y2, b.Y2)
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

// NMS 按置信度降序做按类分组的非极大值抑制：同类且 IoU 超过 iouThresh 的低分框被抑制。
// 直接在 engine.Box 上操作，返回的框为输入切片中被保留的那些（不复制底层数组之外的数据）。
func NMS(boxes []engine.Box, iouThresh float32) []engine.Box {
	order := make([]int, len(boxes))
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(i, j int) int {
		// 降序：高分在前。
		switch {
		case boxes[i].Confidence > boxes[j].Confidence:
			return -1
		case boxes[i].Confidence < boxes[j].Confidence:
			return 1
		default:
			return 0
		}
	})
	suppressed := make([]bool, len(boxes))
	keep := make([]engine.Box, 0, len(boxes))
	for i := 0; i < len(order); i++ {
		if suppressed[i] {
			continue
		}
		bi := boxes[order[i]]
		keep = append(keep, bi)
		for j := i + 1; j < len(order); j++ {
			if suppressed[j] {
				continue
			}
			bj := boxes[order[j]]
			if bj.ClassID != bi.ClassID {
				continue
			}
			if IoU(bi, bj) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
