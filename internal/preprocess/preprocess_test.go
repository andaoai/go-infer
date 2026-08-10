package preprocess

import (
	"image"
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
