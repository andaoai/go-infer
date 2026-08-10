package metric

import (
	"image"
	"math/bits"

	"github.com/andaoai/go-infer/internal/data"
)

// maskMaxSide 是掩膜栅格化的目标长边像素数。多边形先等比缩放到该分辨率，
// 再转成位集，既保证 IoU 精度，又把每张掩膜的内存与运算量控制在常数级
// （COCO 评测在 C 里用全分辨率，但对 Go 回归/冒烟校验，降采样到 256
// 长边引入的 IoU 误差 <1%，而速度提升一个数量级）。
const maskMaxSide = 256

// maskBits 把一组外轮廓（规范 data.Point）等比缩放到长边 maskMaxSide 后栅格化为位集。
// 返回位集、网格宽高；空轮廓返回 nil。
func maskBits(rings [][]data.Point, w, h int) ([]uint64, int, int) {
	if w <= 0 || h <= 0 || len(rings) == 0 {
		return nil, 0, 0
	}
	scale := float64(maskMaxSide) / float64(maxi(w, h))
	gw := maxi(1, int(float64(w)*scale+0.5))
	gh := maxi(1, int(float64(h)*scale+0.5))
	stride := (gw + 63) / 64
	bitset := make([]uint64, stride*gh)

	for _, ring := range rings {
		if len(ring) < 3 {
			continue
		}
		scaled := make([]image.Point, len(ring))
		for i, p := range ring {
			scaled[i] = image.Pt(
				clampInt(int(float64(p.X)*scale+0.5), 0, gw),
				clampInt(int(float64(p.Y)*scale+0.5), 0, gh),
			)
		}
		fillBitset(bitset, scaled, gw, gh, stride)
	}
	return bitset, gw, gh
}

// fillBitset 用扫描线算法把多边形填进位集。
func fillBitset(bitset []uint64, poly []image.Point, w, h, stride int) {
	type edge struct {
		yMin, yMax int
		x0, dx     float64
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
		edges = append(edges, edge{yMin: ay, yMax: by, x0: float64(ax), dx: float64(bx-ax) / float64(by-ay)})
	}
	for y := 0; y < h; y++ {
		var xs []float64
		for _, e := range edges {
			if y >= e.yMin && y < e.yMax {
				xs = append(xs, e.x0+float64(y-e.yMin)*e.dx)
			}
		}
		for i := 1; i < len(xs); i++ {
			for j := i; j > 0 && xs[j-1] > xs[j]; j-- {
				xs[j-1], xs[j] = xs[j], xs[j-1]
			}
		}
		rowBase := y * stride
		for k := 0; k+1 < len(xs); k += 2 {
			l := clampInt(int(xs[k]+0.5), 0, w)
			r := clampInt(int(xs[k+1]+0.5), 0, w)
			for x := l; x < r; x++ {
				bitset[rowBase+x>>6] |= 1 << (uint(x) & 63)
			}
		}
	}
}

// bitsetIoU 计算两个同尺寸位集的交并比。
func bitsetIoU(a, b []uint64) float64 {
	var inter, union int
	for i := range a {
		av, bv := a[i], b[i]
		inter += bits.OnesCount64(av & bv)
		union += bits.OnesCount64(av | bv)
	}
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

func maxi(a, b int) int {
	if a > b {
		return a
	}
	return b
}
