// 可视化绘制：检测框/实例分割多边形/标签/扫描线填充/Bresenham 边线/alpha 混合。
// 从 server.go 抽出，保持 HTTP 层只关心路由与 handler。
package api

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"

	"github.com/andaoai/go-infer/internal/engine"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// drawPalette 是检测/分割可视化用的固定 6 色调色板（按 ClassID 取模）。
var drawPalette = []color.RGBA{
	{0, 255, 0, 255}, {255, 0, 0, 255}, {0, 255, 255, 255},
	{255, 255, 0, 255}, {255, 0, 255, 255}, {0, 128, 255, 255},
}

func drawBoxes(src image.Image, dets []engine.Detection) image.Image {
	b := src.Bounds()
	canvas, ok := src.(*image.RGBA)
	if !ok {
		canvas = image.NewRGBA(b)
		draw.Draw(canvas, b, src, b.Min, draw.Src)
	}
	out := image.NewRGBA(b)
	draw.Draw(out, b, canvas, b.Min, draw.Src)

	for _, d := range dets {
		c := drawPalette[d.ClassID%len(drawPalette)]
		x1, y1, x2, y2 := int(d.X1), int(d.Y1), int(d.X2), int(d.Y2)
		drawRect(out, x1, y1, x2, y2, c)
		label := fmt.Sprintf("%s %.2f", d.ClassName, d.Confidence)
		drawLabel(out, x1, y1, label, c)
	}
	return out
}

func drawRect(img *image.RGBA, x1, y1, x2, y2 int, c color.Color) {
	for x := x1; x <= x2; x++ {
		img.Set(x, y1, c)
		img.Set(x, y2, c)
	}
	for y := y1; y <= y2; y++ {
		img.Set(x1, y, c)
		img.Set(x2, y, c)
	}
}

func drawLabel(img *image.RGBA, x, y int, text string, c color.Color) {
	const (
		fontW = 7
		fontH = 13
		pad   = 2
	)
	bgW := len(text)*fontW + pad*2
	bgH := fontH + pad*2
	bgY := y - bgH
	if bgY < 0 {
		bgY = y
	}
	bounds := img.Bounds()
	for dy := 0; dy < bgH; dy++ {
		for dx := 0; dx < bgW; dx++ {
			px, py := x+dx, bgY+dy
			if px >= bounds.Min.X && px < bounds.Max.X && py >= bounds.Min.Y && py < bounds.Max.Y {
				img.Set(px, py, c)
			}
		}
	}
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.Black),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(x+pad, bgY+pad+fontH-1),
	}
	d.DrawString(text)
}

func drawInstances(src image.Image, insts []engine.Instance) image.Image {
	b := src.Bounds()
	canvas, ok := src.(*image.RGBA)
	if !ok {
		canvas = image.NewRGBA(b)
		draw.Draw(canvas, b, src, b.Min, draw.Src)
	}
	out := image.NewRGBA(b)
	draw.Draw(out, b, canvas, b.Min, draw.Src)

	for _, ins := range insts {
		c := drawPalette[ins.ClassID%len(drawPalette)]
		for _, poly := range ins.Mask {
			fillPolygon(out, poly, color.RGBA{c.R, c.G, c.B, 90})
			drawPolygon(out, poly, c)
		}
		x1, y1, x2, y2 := int(ins.X1), int(ins.Y1), int(ins.X2), int(ins.Y2)
		drawRect(out, x1, y1, x2, y2, c)
		label := fmt.Sprintf("%s %.2f", ins.ClassName, ins.Confidence)
		drawLabel(out, x1, y1, label, c)
	}
	return out
}

// fillPolygon 用扫描线算法填充多边形（半透明覆盖）。
func fillPolygon(img *image.RGBA, poly []engine.Point, c color.Color) {
	if len(poly) < 3 {
		return
	}
	bounds := img.Bounds()
	minY, maxY := int(poly[0].Y), int(poly[0].Y)
	for _, p := range poly {
		y := int(p.Y)
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}
	}
	if minY < bounds.Min.Y {
		minY = bounds.Min.Y
	}
	if maxY >= bounds.Max.Y {
		maxY = bounds.Max.Y - 1
	}
	cr, cg, cb, ca := c.RGBA()
	fillA := uint8(ca >> 8)
	for y := minY; y <= maxY; y++ {
		// 求扫描线与各边的交点 x。
		var xs []float32
		for i := 0; i < len(poly); i++ {
			a := poly[i]
			b := poly[(i+1)%len(poly)]
			ay, by := int(a.Y), int(b.Y)
			if ay == by {
				continue
			}
			if y >= min(ay, by) && y < max(ay, by) {
				t := float32(y-ay) / float32(by-ay)
				xs = append(xs, a.X+t*(b.X-a.X))
			}
		}
		// 交点排序后两两配对填充。
		for i := 1; i < len(xs); i++ {
			for j := i; j > 0 && xs[j-1] > xs[j]; j-- {
				xs[j-1], xs[j] = xs[j], xs[j-1]
			}
		}
		for i := 0; i+1 < len(xs); i += 2 {
			x1 := int(xs[i])
			x2 := int(xs[i+1])
			if x1 < bounds.Min.X {
				x1 = bounds.Min.X
			}
			if x2 >= bounds.Max.X {
				x2 = bounds.Max.X - 1
			}
			for x := x1; x <= x2; x++ {
				alphaBlend(img, x, y, uint8(cr>>8), uint8(cg>>8), uint8(cb>>8), fillA)
			}
		}
	}
}

// drawPolygon 连接多边形顶点画边线。
func drawPolygon(img *image.RGBA, poly []engine.Point, c color.Color) {
	for i := 0; i < len(poly); i++ {
		a := poly[i]
		b := poly[(i+1)%len(poly)]
		drawLine(img, int(a.X), int(a.Y), int(b.X), int(b.Y), c)
	}
}

// drawLine 用 Bresenham 算法画线。
func drawLine(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx := 1
	if x0 >= x1 {
		sx = -1
	}
	sy := 1
	if y0 >= y1 {
		sy = -1
	}
	err := dx + dy
	bounds := img.Bounds()
	for {
		if x0 >= bounds.Min.X && x0 < bounds.Max.X && y0 >= bounds.Min.Y && y0 < bounds.Max.Y {
			img.Set(x0, y0, c)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

// alphaBlend 将 (r,g,b,a) 以 source-over 混合到 img 的 (x,y)。
func alphaBlend(img *image.RGBA, x, y int, r, g, b, a uint8) {
	i := img.PixOffset(x, y)
	if i < 0 {
		return
	}
	dst := img.Pix[i : i+4 : i+4]
	sa := uint32(a)
	da := uint32(dst[3])
	oa := sa + (da*(255-sa))/255 // 输出 alpha
	if oa == 0 {
		return
	}
	blend := func(s, d uint32) uint8 {
		// (s*sa/255 + d*da/255*(1-sa/255)) / (oa/255)
		return uint8((s*sa*255 + d*da*(255-sa)) / (oa * 255))
	}
	dst[0] = blend(uint32(r), uint32(dst[0]))
	dst[1] = blend(uint32(g), uint32(dst[1]))
	dst[2] = blend(uint32(b), uint32(dst[2]))
	dst[3] = uint8(oa)
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
