package preprocess

import (
	"bytes"
	"image"
	"image/draw"
	"image/jpeg"
	"math/rand"
	"testing"
)

func TestLetterboxSquare(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 200, 100))
	lb := Letterbox(src, 100, 100)
	if lb.Scale != 0.5 {
		t.Fatalf("scale got %f, want 0.5", lb.Scale)
	}
	if lb.PadX != 0 || lb.PadY != 25 {
		t.Fatalf("pad got (%f,%f), want (0,25)", lb.PadX, lb.PadY)
	}
}

func TestNCHWNormalization(t *testing.T) {
	w, h := 2, 2
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(src.Pix); i += 4 {
		src.Pix[i], src.Pix[i+1], src.Pix[i+2] = 255, 255, 255
		src.Pix[i+3] = 255
	}
	dst := make([]float32, 3*w*h)
	FillNCHW(dst, src, w, h)
	for i, v := range dst {
		if v != 1.0 {
			t.Fatalf("dst[%d]=%f, want 1.0", i, v)
		}
	}
}

func TestClamp(t *testing.T) {
	if Clamp(5, 0, 10) != 5 {
		t.Fatal("clamp interior failed")
	}
	if Clamp(-1, 0, 10) != 0 {
		t.Fatal("clamp lo failed")
	}
	if Clamp(11, 0, 10) != 10 {
		t.Fatal("clamp hi failed")
	}
}

// TestRGBAtYCbCrMatchesGeneric 验证 JPEG 解码出的 *image.YCbCr 快路径与
// 通用 image.Image.At().RGBA() 逐像素一致（golden 像素对齐）。
func TestRGBAtYCbCrMatchesGeneric(t *testing.T) {
	// 用 image/jpeg 编码再解码，得到真实的 4:2:0 YCbCr（带色度抽样）。
	src := image.NewRGBA(image.Rect(0, 0, 17, 13))
	rng := rand.New(rand.NewSource(7))
	for i := range src.Pix {
		src.Pix[i] = uint8(rng.Intn(256))
	}
	for i := 3; i < len(src.Pix); i += 4 {
		src.Pix[i] = 255
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatal(err)
	}
	img, err := jpeg.Decode(&buf)
	if err != nil {
		t.Fatal(err)
	}
	ycc, ok := img.(*image.YCbCr)
	if !ok {
		t.Skipf("jpeg.Decode 未返回 *image.YCbCr（got %T），跳过", img)
	}
	b := ycc.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			fast := rgbAt(ycc, x, y)
			c := ycc.At(x, y) // 走通用 color.Color 路径
			rr, gg, bb, _ := c.RGBA()
			want := rgb{uint8(rr >> 8), uint8(gg >> 8), uint8(bb >> 8)}
			if fast != want {
				t.Fatalf("(%d,%d) fast=%+v generic=%+v", x, y, fast, want)
			}
		}
	}
}

// TestResizeBilinearIntoMatchesReference 验证直接写入 pad 偏移的内联缩放路径
// 与"先缩放整幅 + 用 draw.Draw 贴到画布 pad 处"像素对齐。
func TestResizeBilinearIntoMatchesReference(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 31, 23))
	rng := rand.New(rand.NewSource(99))
	for i := range src.Pix {
		src.Pix[i] = uint8(rng.Intn(256))
	}
	for i := 3; i < len(src.Pix); i += 4 {
		src.Pix[i] = 255
	}
	const dstW, dstH = 96, 96
	const padX, padY, sw, sh = 16, 20, 64, 56

	// 新路径：fill114 + resizeBilinearInto 直接写 pad 区。
	got := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	fill114(got.Pix)
	resizeBilinearInto(got, padX, padY, src, sw, sh)

	// 参考路径：ResizeBilinear 缩出独立图，再用 draw.Draw 贴到画布 pad 区。
	scaled := ResizeBilinear(src, sw, sh)
	want := image.NewRGBA(image.Rect(0, 0, dstW, dstH))
	fill114(want.Pix)
	draw.Draw(want, image.Rect(padX, padY, padX+sw, padY+sh), scaled, image.Point{}, draw.Src)

	if !bytes.Equal(got.Pix, want.Pix) {
		// 定位首个不同像素
		for i := range got.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("pixel offset %d differs: got %d want %d", i, got.Pix[i], want.Pix[i])
			}
		}
		t.Fatal("像素不一致（未定位到具体偏移）")
	}
}

func BenchmarkLetterbox(b *testing.B) {
	src := image.NewRGBA(image.Rect(0, 0, 1920, 1080))
	rng := rand.New(rand.NewSource(1))
	for i := range src.Pix {
		src.Pix[i] = uint8(rng.Intn(256))
	}
	for i := 3; i < len(src.Pix); i += 4 {
		src.Pix[i] = 255
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Letterbox(src, 640, 640)
	}
}

func BenchmarkFillNCHW(b *testing.B) {
	const w, h = 640, 640
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	dst := make([]float32, 3*w*h)
	rng := rand.New(rand.NewSource(2))
	for i := range src.Pix {
		src.Pix[i] = uint8(rng.Intn(256))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FillNCHW(dst, src, w, h)
	}
}
