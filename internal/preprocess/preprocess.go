// Package preprocess 提供各视觉引擎可复用的图像预处理与后处理工具：
// letterbox 等比缩放、双线性插值、NCHW 归一化、NMS/IoU。
package preprocess

import (
	"image"
	"image/draw"
)

// Letterboxed 是 letterbox 的产物：含画布与坐标还原参数。
type Letterboxed struct {
	RGBA  *image.RGBA
	Scale float32 // 原图 -> 画布的等比缩放系数
	PadX  float32 // 居中左填充
	PadY  float32 // 居中上填充
}

// Letterbox 将原图等比缩放并居中填充到目标尺寸，填充色 114（YOLO 惯例）。
func Letterbox(src image.Image, dstW, dstH int) *Letterboxed {
	b := src.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	scale := MinF(float32(dstW)/float32(srcW), float32(dstH)/float32(srcH))
	newW := Max2(1, int(float32(srcW)*scale+0.5))
	newH := Max2(1, int(float32(srcH)*scale+0.5))
	padX := (float32(dstW) - float32(newW)) / 2
	padY := (float32(dstH) - float32(newH)) / 2

	canvas := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	for i := range canvas.Pix {
		canvas.Pix[i] = 114
	}
	scaled := ResizeBilinear(src, newW, newH)
	draw.Draw(canvas,
		image.Rect(int(padX), int(padY), int(padX)+newW, int(padY)+newH),
		scaled, scaled.Bounds().Min, draw.Over)

	return &Letterboxed{RGBA: canvas, Scale: scale, PadX: padX, PadY: padY}
}

// FillNCHW 将 RGBA 写入 NCHW float32 缓冲，RGB 顺序，归一化到 0~1。
func FillNCHW(dst []float32, src *image.RGBA, w, h int) {
	plane := w * h
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (y*w + x) * 4
			idx := y*w + x
			dst[idx] = float32(src.Pix[i]) / 255.0
			dst[plane+idx] = float32(src.Pix[i+1]) / 255.0
			dst[2*plane+idx] = float32(src.Pix[i+2]) / 255.0
		}
	}
}

// ResizeBilinear 双线性插值缩放。
func ResizeBilinear(src image.Image, dstW, dstH int) *image.RGBA {
	srcB := src.Bounds()
	srcW, srcH := srcB.Dx(), srcB.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	if srcW == 0 || srcH == 0 {
		return dst
	}
	scaleX := float32(srcW) / float32(dstW)
	scaleY := float32(srcH) / float32(dstH)
	for y := 0; y < dstH; y++ {
		sy := float32(y)*scaleY + float32(srcB.Min.Y)
		y0 := int(sy)
		dy := sy - float32(y0)
		if y0 >= srcB.Max.Y-1 {
			y0 = srcB.Max.Y - 2
			dy = 1
		}
		for x := 0; x < dstW; x++ {
			sx := float32(x)*scaleX + float32(srcB.Min.X)
			x0 := int(sx)
			dx := sx - float32(x0)
			if x0 >= srcB.Max.X-1 {
				x0 = srcB.Max.X - 2
				dx = 1
			}
			c00 := rgbAt(src, x0, y0)
			c10 := rgbAt(src, x0+1, y0)
			c01 := rgbAt(src, x0, y0+1)
			c11 := rgbAt(src, x0+1, y0+1)

			w00 := (1 - dx) * (1 - dy)
			w10 := dx * (1 - dy)
			w01 := (1 - dx) * dy
			w11 := dx * dy

			off := (y*dstW + x) * 4
			dst.Pix[off] = uint8(float32(c00.r)*w00 + float32(c10.r)*w10 + float32(c01.r)*w01 + float32(c11.r)*w11)
			dst.Pix[off+1] = uint8(float32(c00.g)*w00 + float32(c10.g)*w10 + float32(c01.g)*w01 + float32(c11.g)*w11)
			dst.Pix[off+2] = uint8(float32(c00.b)*w00 + float32(c10.b)*w10 + float32(c01.b)*w01 + float32(c11.b)*w11)
			dst.Pix[off+3] = 255
		}
	}
	return dst
}

type rgb struct{ r, g, b uint8 }

func rgbAt(img image.Image, x, y int) rgb {
	if r, ok := img.(*image.RGBA); ok {
		i := (y-r.Rect.Min.Y)*r.Stride + (x-r.Rect.Min.X)*4
		return rgb{r.Pix[i], r.Pix[i+1], r.Pix[i+2]}
	}
	c := img.At(x, y)
	rr, gg, bb, _ := c.RGBA()
	return rgb{uint8(rr >> 8), uint8(gg >> 8), uint8(bb >> 8)}
}

// Clamp 将 v 限制在 [lo, hi]。
func Clamp(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func MinF(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
func MaxF(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
func Max2(a, b int) int {
	if a > b {
		return a
	}
	return b
}
