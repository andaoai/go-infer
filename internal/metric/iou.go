// Package metric 在数据集上计算检测/分割的精度指标（IoU、AP、mAP）。
//
// 设计为引擎与格式无关：消费规范 data.Object，不依赖任何具体推理后端或标签格式。
// 框/掩膜几何全部用标准库实现，不引入 OpenCV。
package metric

import (
	"image"
	"sort"

	"github.com/andaoai/go-infer/internal/data"
)

// BoxIoU 返回两个轴对齐框的交并比。
func BoxIoU(a, b data.BBox) float64 {
	ix1 := maxf(a.X1, b.X1)
	iy1 := maxf(a.Y1, b.Y1)
	ix2 := minf(a.X2, b.X2)
	iy2 := minf(a.Y2, b.Y2)
	iw := ix2 - ix1
	ih := iy2 - iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := float64(iw * ih)
	areaA := float64((a.X2 - a.X1) * (a.Y2 - a.Y1))
	areaB := float64((b.X2 - b.X1) * (b.Y2 - b.Y1))
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// Rasterize 把多边形扫描线填充成布尔掩膜（true=前景）。
func Rasterize(poly []image.Point, w, h int) []bool {
	mask := make([]bool, w*h)
	if len(poly) < 3 || w <= 0 || h <= 0 {
		return mask
	}
	type edge struct {
		yMin, yMax int
		x0         float32
		dx         float32
	}
	var edges []edge
	n := len(poly)
	for i := 0; i < n; i++ {
		ax, ay := poly[i].X, poly[i].Y
		bx, by := poly[(i+1)%n].X, poly[(i+1)%n].Y
		if ay == by {
			continue
		}
		if ay > by {
			ax, ay, bx, by = bx, by, ax, ay
		}
		dx := float32(bx-ax) / float32(by-ay)
		edges = append(edges, edge{yMin: ay, yMax: by, x0: float32(ax), dx: dx})
	}
	for y := 0; y < h; y++ {
		var xs []float32
		for _, e := range edges {
			if y >= e.yMin && y < e.yMax {
				xs = append(xs, e.x0+float32(y-e.yMin)*e.dx)
			}
		}
		sort.Slice(xs, func(a, b int) bool { return xs[a] < xs[b] })
		for k := 0; k+1 < len(xs); k += 2 {
			l := clampInt(int(xs[k]+0.5), 0, w)
			r := clampInt(int(xs[k+1]+0.5), 0, w)
			for x := l; x < r; x++ {
				mask[y*w+x] = true
			}
		}
	}
	return mask
}

// MaskIoU 把两个多边形栅格化到 w×h 后计算交并比。
func MaskIoU(a, b []image.Point, w, h int) float64 {
	return maskIoU(Rasterize(a, w, h), Rasterize(b, w, h))
}

func maskIoU(ma, mb []bool) float64 {
	var inter, union int
	for i := range ma {
		if ma[i] && mb[i] {
			inter++
		}
		if ma[i] || mb[i] {
			union++
		}
	}
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
