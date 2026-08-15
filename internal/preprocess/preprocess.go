// Package preprocess 提供各视觉引擎可复用的图像预处理与后处理工具：
// letterbox 等比缩放、双线性插值、NCHW 归一化、NMS/IoU。
package preprocess

import (
	"image"
	"image/color"
)

// Letterboxed 是 letterbox 的产物：含画布与坐标还原参数。
type Letterboxed struct {
	RGBA  *image.RGBA
	Scale float32 // 原图 -> 画布的等比缩放系数
	PadX  float32 // 居中左填充
	PadY  float32 // 居中上填充
}

// Letterbox 将原图等比缩放并居中填充到目标尺寸，填充色 114（YOLO 惯例）。
//
// 直接把双线性缩放结果写入 letterbox canvas 的 pad 偏移处，不再分配独立
// scaled 图像并用 draw.Draw 二次拷贝；背景填充按 4 字节批量写入。
func Letterbox(src image.Image, dstW, dstH int) *Letterboxed {
	b := src.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	scale := MinF(float32(dstW)/float32(srcW), float32(dstH)/float32(srcH))
	newW := Max2(1, int(float32(srcW)*scale+0.5))
	newH := Max2(1, int(float32(srcH)*scale+0.5))
	padX := (float32(dstW) - float32(newW)) / 2
	padY := (float32(dstH) - float32(newH)) / 2

	canvas := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	fill114(canvas.Pix)
	resizeBilinearInto(canvas, int(padX), int(padY), src, newW, newH)

	return &Letterboxed{RGBA: canvas, Scale: scale, PadX: padX, PadY: padY}
}

// fill114 把 RGBA 像素缓冲填充为 114/114/114/255（YOLO letterbox 填充色），
// 按 4 字节一组写入，避免逐字节循环。
func fill114(pix []uint8) {
	for i := 0; i+4 <= len(pix); i += 4 {
		pix[i] = 114
		pix[i+1] = 114
		pix[i+2] = 114
		pix[i+3] = 255
	}
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
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	if srcB.Dx() == 0 || srcB.Dy() == 0 {
		return dst
	}
	resizeBilinearInto(dst, 0, 0, src, dstW, dstH)
	return dst
}

// resizeBilinearInto 把 src 双线性缩放到 w×h，结果直接写入 dst 中以
// (dstX,dstY) 为左上角的区域。w/h 是缩放后的尺寸（非整张 dst 尺寸）。
// 采样坐标计算与 ResizeBilinear 完全一致，保证两条路径输出像素对齐。
func resizeBilinearInto(dst *image.RGBA, dstX, dstY int, src image.Image, w, h int) {
	srcB := src.Bounds()
	srcW, srcH := srcB.Dx(), srcB.Dy()
	if srcW == 0 || srcH == 0 {
		return
	}
	scaleX := float32(srcW) / float32(w)
	scaleY := float32(srcH) / float32(h)
	dstStride := dst.Stride
	for y := 0; y < h; y++ {
		sy := float32(y)*scaleY + float32(srcB.Min.Y)
		y0 := int(sy)
		dy := sy - float32(y0)
		if y0 >= srcB.Max.Y-1 {
			y0 = srcB.Max.Y - 2
			dy = 1
		}
		dstRowOff := (dstY+y)*dstStride + dstX*4
		for x := 0; x < w; x++ {
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

			off := dstRowOff + x*4
			dst.Pix[off] = uint8(float32(c00.r)*w00 + float32(c10.r)*w10 + float32(c01.r)*w01 + float32(c11.r)*w11)
			dst.Pix[off+1] = uint8(float32(c00.g)*w00 + float32(c10.g)*w10 + float32(c01.g)*w01 + float32(c11.g)*w11)
			dst.Pix[off+2] = uint8(float32(c00.b)*w00 + float32(c10.b)*w10 + float32(c01.b)*w01 + float32(c11.b)*w11)
			dst.Pix[off+3] = 255
		}
	}
}

type rgb struct{ r, g, b uint8 }

// rgbAt 读取 (x,y) 的 RGB。对 *image.RGBA 与 JPEG 解码出的 *image.YCbCr
// 走快路径，其余走通用 image.Image.At 接口。YCbCr 路径用标准库
// color.YCbCrToRGB（full-range），与通用路径 c.RGBA() 逐像素一致。
func rgbAt(img image.Image, x, y int) rgb {
	switch r := img.(type) {
	case *image.RGBA:
		i := (y-r.Rect.Min.Y)*r.Stride + (x-r.Rect.Min.X)*4
		return rgb{r.Pix[i], r.Pix[i+1], r.Pix[i+2]}
	case *image.YCbCr:
		yy := r.YOffset(x, y)
		ci := r.COffset(x, y)
		rr, gg, bb := color.YCbCrToRGB(r.Y[yy], r.Cb[ci], r.Cr[ci])
		return rgb{rr, gg, bb}
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
