package seg

// contour 从二值掩膜（true=前景）中追踪外轮廓，返回多条闭合多边形。
//
// 采用简化的 Moore 邻域追踪：扫描到一个未访问的前景边界像素后，沿其
// 8-邻域顺时针绕行回到起点；被追踪过的像素标记 visited，避免重复。
// 掩膜坐标原点在左上角，x 向右、y 向下。
func contour(mask []bool, w, h int) [][]point {
	if w <= 0 || h <= 0 || len(mask) != w*h {
		return nil
	}
	visited := make([]bool, len(mask))
	var polys [][]point

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			if !mask[idx] || visited[idx] {
				continue
			}
			// 只把"与背景或图像边缘相邻"的像素当作轮廓起点，减少内部扫描。
			if !isBoundary(mask, w, h, x, y) {
				continue
			}
			poly := traceOne(mask, visited, w, h, x, y)
			if len(poly) >= 3 {
				polys = append(polys, simplify(poly))
			}
		}
	}
	return polys
}

type point struct{ x, y int }

// 8-邻域，顺时针顺序，起点为正上方。
var nbr8 = [8][2]int{
	{0, -1}, {1, -1}, {1, 0}, {1, 1},
	{0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
}

func isBoundary(mask []bool, w, h, x, y int) bool {
	for _, n := range nbr8 {
		nx, ny := x+n[0], y+n[1]
		if nx < 0 || ny < 0 || nx >= w || ny >= h || !mask[ny*w+nx] {
			return true
		}
	}
	return false
}

// traceOne 从 (sx,sy) 出发沿边界绕行一周，标记 visited。
func traceOne(mask []bool, visited []bool, w, h, sx, sy int) []point {
	var poly []point
	x, y := sx, sy
	startDir := 0 // 进入方向；下一个候选从其反方向开始
	for {
		idx := y*w + x
		if !visited[idx] {
			visited[idx] = true
			poly = append(poly, point{x, y})
		}
		found := false
		for k := 0; k < 8; k++ {
			d := (startDir + k) % 8
			nx, ny := x+nbr8[d][0], y+nbr8[d][1]
			if nx < 0 || ny < 0 || nx >= w || ny >= h {
				continue
			}
			if mask[ny*w+nx] {
				// 记录进入方向：下一步从当前方向的"后方"搜起。
				startDir = (d + 4 + 1) % 8
				x, y = nx, ny
				found = true
				break
			}
		}
		if !found {
			break // 孤立像素，已加入
		}
		if x == sx && y == sy {
			break
		}
		// 防御性上限，避免异常掩膜造成死循环。
		if len(poly) > w*h {
			break
		}
	}
	return poly
}

// simplify 用 Ramer-Douglas-Peucker 算法剔除共线点，压缩点数
// （逐像素的轮廓会产生很大 JSON/渲染开销）。
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
