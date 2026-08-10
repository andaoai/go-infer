package detector

import (
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
)

type letterboxed struct {
	rgba  *image.RGBA
	scale float32 // 原图 -> letterbox 的等比缩放系数
	padX  float32 // 左右填充（居中时的左填充）
	padY  float32 // 上下填充（居中时的上填充）
}

// letterbox 将原图等比缩放并居中填充到目标尺寸，填充色 114（YOLO 惯例）。
func letterbox(src image.Image, dstW, dstH int) (*letterboxed, error) {
	b := src.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	scale := minF(float32(dstW)/float32(srcW), float32(dstH)/float32(srcH))
	newW := int(float32(srcW)*scale + 0.5)
	newH := int(float32(srcH)*scale + 0.5)
	if newW < 1 {
		newW = 1
	}
	if newH < 1 {
		newH = 1
	}
	padX := (float32(dstW) - float32(newW)) / 2
	padY := (float32(dstH) - float32(newH)) / 2

	canvas := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	// 填充灰底 114
	for i := range canvas.Pix {
		canvas.Pix[i] = 114
	}
	// 缩放绘制
	scaled := resizeBilinear(src, newW, newH)
	draw.Draw(canvas, image.Rect(int(padX), int(padY), int(padX)+newW, int(padY)+newH),
		scaled, scaled.Bounds().Min, draw.Over)

	return &letterboxed{rgba: canvas, scale: scale, padX: padX, padY: padY}, nil
}

// fillNCHW 将 RGBA 转为 NCHW float32，RGB 顺序，归一化到 0~1。
func fillNCHW(dst []float32, src *image.RGBA, w, h int) {
	plane := w * h
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := (y*w + x) * 4
			r := src.Pix[i]
			g := src.Pix[i+1]
			b := src.Pix[i+2]
			idx := y*w + x
			dst[idx] = float32(r) / 255.0
			dst[plane+idx] = float32(g) / 255.0
			dst[2*plane+idx] = float32(b) / 255.0
		}
	}
}

// resizeBilinear 双线性插值缩放。
func resizeBilinear(src image.Image, dstW, dstH int) *image.RGBA {
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
			c00 := rGBAAt(src, x0, y0)
			c10 := rGBAAt(src, x0+1, y0)
			c01 := rGBAAt(src, x0, y0+1)
			c11 := rGBAAt(src, x0+1, y0+1)

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

func rGBAAt(img image.Image, x, y int) rgb {
	if r, ok := img.(*image.RGBA); ok {
		i := (y-r.Rect.Min.Y)*r.Stride + (x-r.Rect.Min.X)*4
		return rgb{r.Pix[i], r.Pix[i+1], r.Pix[i+2]}
	}
	c := img.At(x, y)
	rr, gg, bb, _ := c.RGBA()
	return rgb{uint8(rr >> 8), uint8(gg >> 8), uint8(bb >> 8)}
}
