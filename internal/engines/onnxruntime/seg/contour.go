package seg

// point 是掩膜上的一个整数坐标点（局部坐标系，原点在框左上角）。
type point struct{ x, y int }

// contour 从二值掩膜（true=前景）提取每个连通域的外轮廓，返回多条闭合多边形。
//
// 等价于 OpenCV 的 findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)：
//  1. 8-连通连通域标记，把碎裂的前景像素分组成若干实例
//  2. 每个连通域取最上最左的边界点做 Moore 邻域外轮廓追踪
//  3. RDP 算法压缩共线点
//
// 掩膜坐标原点在左上角，x 向右、y 向下。
func contour(mask []bool, w, h int) [][]point {
	if w <= 0 || h <= 0 || len(mask) != w*h {
		return nil
	}

	labels := make([]int, len(mask))
	type component struct {
		pixels []int
		label  int
	}
	var components []component
	nextLabel := 1

	// 8-连通洪水填充标记。
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			if !mask[idx] || labels[idx] != 0 {
				continue
			}
			pixels := floodFill(mask, labels, w, h, x, y, nextLabel)
			components = append(components, component{pixels, nextLabel})
			nextLabel++
		}
	}

	var polys [][]point
	for _, c := range components {
		if poly := traceOuter(mask, labels, w, h, c.pixels, c.label); len(poly) >= 3 {
			polys = append(polys, simplify(poly))
		}
	}
	return polys
}

// floodFill 从 (sx,sy) 做 8-连通 BFS，标记 label 并返回该连通域所有像素下标。
func floodFill(mask []bool, labels []int, w, h, sx, sy, label int) []int {
	stack := []int{sy*w + sx}
	labels[stack[0]] = label
	var pixels []int
	for len(stack) > 0 {
		idx := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		pixels = append(pixels, idx)
		x, y := idx%w, idx/w
		for dy := -1; dy <= 1; dy++ {
			ny := y + dy
			if ny < 0 || ny >= h {
				continue
			}
			for dx := -1; dx <= 1; dx++ {
				if dx == 0 && dy == 0 {
					continue
				}
				nx := x + dx
				if nx < 0 || nx >= w {
					continue
				}
				ni := ny*w + nx
				if mask[ni] && labels[ni] == 0 {
					labels[ni] = label
					stack = append(stack, ni)
				}
			}
		}
	}
	return pixels
}

// 8-邻域，顺序：从正上方开始顺时针。
var nbr8 = [8][2]int{
	{0, -1}, {1, -1}, {1, 0}, {1, 1},
	{0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
}

// traceOuter 用 Moore 邻域追踪单个连通域（标签 compLabel）的外轮廓。
//
// 取该连通域最上最左的边界像素为起点，沿邻域顺时针绕行一周回到起点。
// 只把属于本连通域的像素当作"前景"，避免穿到其他实例（即使它们也在掩膜中）。
func traceOuter(mask []bool, labels []int, w, h int, pixels []int, compLabel int) []point {
	// 找最上最左像素（y 最小，同 y 取 x 最小）——它一定是外边界点。
	sx, sy := w, h
	for _, idx := range pixels {
		x, y := idx%w, idx/w
		if y < sy || (y == sy && x < sx) {
			sx, sy = x, y
		}
	}

	var poly []point
	x, y := sx, sy
	// backDir 是"我们从哪个方向进入当前像素"的邻域索引；下一个候选从
	// (backDir+1)%8 开始顺时针搜，保证始终贴外边界走。
	backDir := 6 // 起点：假定从左邻域(-1,0)进入，先向上方找

	for {
		poly = append(poly, point{x, y})
		found := false
		for k := 0; k < 8; k++ {
			d := (backDir + k) % 8
			nx, ny := x+nbr8[d][0], y+nbr8[d][1]
			if nx < 0 || ny < 0 || nx >= w || ny >= h {
				continue
			}
			if labels[ny*w+nx] == compLabel {
				// 进入 (nx,ny) 的方向是 d 的反向；下一次从其顺时针下一个搜起。
				backDir = (d + 5) % 8 // 反方向再 -1（顺时针优先）
				x, y = nx, ny
				found = true
				break
			}
		}
		if !found {
			break // 孤立像素
		}
		if x == sx && y == sy {
			break
		}
		if len(poly) > w*h {
			break // 防御性上限
		}
	}
	return poly
}

// simplify 用 Ramer-Douglas-Peucker 算法剔除共线点，压缩点数。
func simplify(p []point) []point {
	if len(p) <= 4 {
		return p
	}
	const eps2 = 1.0 // 像素平方距离：偏离连线超过 1px 才保留
	keep := make([]bool, len(p))
	keep[0], keep[len(p)-1] = true, true
	rdp(p, keep, 0, len(p)-1, eps2)
	out := make([]point, 0, len(p)/2)
	for i, k := range keep {
		if k {
			out = append(out, p[i])
		}
	}
	return out
}

func rdp(p []point, keep []bool, lo, hi int, eps2 float32) {
	if hi-lo < 2 {
		return
	}
	dmax, idx := float32(0), 0
	for i := lo + 1; i < hi; i++ {
		if d := perpDist2(p[lo], p[hi], p[i]); d > dmax {
			dmax, idx = d, i
		}
	}
	if dmax > eps2 {
		keep[idx] = true
		rdp(p, keep, lo, idx, eps2)
		rdp(p, keep, idx, hi, eps2)
	}
}

// perpDist2 返回点 c 到直线 ab 的垂直距离平方。
func perpDist2(a, b, c point) float32 {
	dx, dy := float32(b.x-a.x), float32(b.y-a.y)
	if dx == 0 && dy == 0 {
		return float32((c.x-a.x)*(c.x-a.x) + (c.y-a.y)*(c.y-a.y))
	}
	t := (float32(c.x-a.x)*dx + float32(c.y-a.y)*dy) / (dx*dx + dy*dy)
	px := float32(a.x) + t*dx
	py := float32(a.y) + t*dy
	ex := float32(c.x) - px
	ey := float32(c.y) - py
	return ex*ex + ey*ey
}
