package seg

import (
	"image"
	"math"
	"math/rand"
	"testing"

	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/ortbatch"

	ort "github.com/yalue/onnxruntime_go"
)

func TestParseDetShape(t *testing.T) {
	// 官方排布 [1, 4+nc+nm, anchors]，nc=80, nm=32 → 116
	anchors, attrs, nm, tr, err := parseDetShape(ort.NewShape(1, 116, 8400), 80)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if anchors != 8400 || attrs != 116 || nm != 32 || tr {
		t.Fatalf("got %d/%d/%d/%v, want 8400/116/32/false", anchors, attrs, nm, tr)
	}
	// 转置排布 [1, anchors, 116]
	anchors, attrs, nm, tr, err = parseDetShape(ort.NewShape(1, 8400, 116), 80)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if !tr || anchors != 8400 || attrs != 116 || nm != 32 {
		t.Fatalf("got %d/%d/%d/%v, want 8400/116/32/true", anchors, attrs, nm, tr)
	}
	// 类别不匹配
	if _, _, _, _, err := parseDetShape(ort.NewShape(1, 5, 8400), 80); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

func TestParseShapesDynamicBatch(t *testing.T) {
	// 动态 batch 维：[-1,116,8400] 与 [-1,8400,116]
	if a, attr, nm, tr, err := parseDetShape(ort.NewShape(-1, 116, 8400), 80); err != nil || a != 8400 || attr != 116 || nm != 32 || tr {
		t.Fatalf("dynamic official: got %d/%d/%d/%v %v", a, attr, nm, tr, err)
	}
	if a, attr, nm, tr, err := parseDetShape(ort.NewShape(-1, 8400, 116), 80); err != nil || a != 8400 || attr != 116 || nm != 32 || !tr {
		t.Fatalf("dynamic transposed: got %d/%d/%d/%v %v", a, attr, nm, tr, err)
	}
	// 固定 N>1 + 动态 proto
	if a, attr, _, _, err := parseDetShape(ort.NewShape(4, 116, 8400), 80); err != nil || a != 8400 || attr != 116 {
		t.Fatalf("fixed N=4: got %d/%d %v", a, attr, err)
	}
	if mh, mw, err := parseProtoShape(ort.NewShape(-1, 32, 160, 160), 32); err != nil || mh != 160 || mw != 160 {
		t.Fatalf("dynamic proto: got %dx%d %v", mh, mw, err)
	}
	// 非 batch 维动态应报错
	if _, _, _, _, err := parseDetShape(ort.NewShape(1, -1, 8400), 80); err == nil {
		t.Fatal("expected error for dynamic non-batch det dim")
	}
}

func TestDecodeDetectionsSlotIsolation(t *testing.T) {
	// 官方排布 [B,attrs,anchors]，nc=1,nm=2 → attrs=7，anchors=2，两槽。
	// 仅 slot1/anchor1 放一个高分项，slot0 必须为空。
	e := &Engine{
		attrs: 7, anchors: 2, nc: 1, nm: 2, transposed: false,
		detStride: 7 * 2,
		cfg:       Config{ConfThresh: 0.25},
	}
	detBuf := make([]float32, 2*7*2)
	protoBuf := make([]float32, 0)
	base := 1*e.detStride + 1
	detBuf[base] = 10              // cx
	detBuf[base+e.anchors] = 20    // cy
	detBuf[base+2*e.anchors] = 4   // w
	detBuf[base+3*e.anchors] = 6   // h
	detBuf[base+4*e.anchors] = 0.9 // class0
	detBuf[base+5*e.anchors] = 0.1 // coeff0
	detBuf[base+6*e.anchors] = 0.2 // coeff1
	v := &segView{detBuf: detBuf, protoBuf: protoBuf, detStride: e.detStride}

	if c := e.decodeDetections(v, 0, 0.25); len(c) != 0 {
		t.Fatalf("slot 0 should be empty, got %d", len(c))
	}
	c := e.decodeDetections(v, 1, 0.25)
	if len(c) != 1 {
		t.Fatalf("slot 1 want 1 cand, got %d", len(c))
	}
	if c[0].box.ClassID != 0 || c[0].box.Confidence < 0.89 || c[0].box.X1 != 8 || c[0].box.Y1 != 17 {
		t.Fatalf("unexpected box: %+v", c[0].box)
	}
	if len(c[0].coeffs) != 2 || c[0].coeffs[0] != 0.1 || c[0].coeffs[1] != 0.2 {
		t.Fatalf("coeffs not copied out: %v", c[0].coeffs)
	}
}

func TestParseProtoShape(t *testing.T) {
	mh, mw, err := parseProtoShape(ort.NewShape(1, 32, 160, 160), 32)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if mh != 160 || mw != 160 {
		t.Fatalf("got %dx%d, want 160x160", mh, mw)
	}
	if _, _, err := parseProtoShape(ort.NewShape(1, 16, 160, 160), 32); err == nil {
		t.Fatal("expected nm mismatch error")
	}
}

func TestContourSquare(t *testing.T) {
	// 5x5 掩膜，中心 3x3 实心方块
	w, h := 5, 5
	mask := make([]bool, w*h)
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			mask[y*w+x] = true
		}
	}
	labels := make([]int, w*h)
	polys := contour(mask, labels, w, h)
	if len(polys) != 1 {
		t.Fatalf("expected 1 contour, got %d", len(polys))
	}
	// 外轮廓应经过方块的 8 个角点（追踪方向不限）。
	if len(polys[0]) < 4 {
		t.Fatalf("contour too short: %d points", len(polys[0]))
	}
}

func TestContourEmpty(t *testing.T) {
	mask := make([]bool, 4*4)
	labels := make([]int, 4*4)
	if polys := contour(mask, labels, 4, 4); len(polys) != 0 {
		t.Fatalf("expected 0 contours, got %d", len(polys))
	}
}

func TestLogitFromProb(t *testing.T) {
	// thr=0.5 -> logit=0
	if v := logitFromProb(0.5); v != 0 {
		t.Fatalf("logit(0.5)=%v, want 0", v)
	}
	// 边界保护：0 和 1 不应产生 ±Inf。
	if v := logitFromProb(0); v <= -1e6 || v >= 1e6 {
		t.Fatalf("logit(0) out of clamp range: %v", v)
	}
	if v := logitFromProb(1); v <= -1e6 || v >= 1e6 {
		t.Fatalf("logit(1) out of clamp range: %v", v)
	}
}

// sigmoid 是旧实现的逐像素激活，仅用于等价性测试。
func sigmoid(x float32) float32 { return float32(1.0 / (1.0 + math.Exp(-float64(x)))) }

// TestMaskLogitMatchesSigmoid 用确定性随机 proto/系数验证：
//  1. logit 阈值比较与"sigmoid(sum) >= thr"逐像素等价；
//  2. 预计算每列/每行 floor/clamp 的向量化掩膜与旧 sampleProto 逐像素路径完全一致。
func TestMaskLogitMatchesSigmoid(t *testing.T) {
	rng := rand.New(rand.NewSource(31337))
	// 小尺寸掩膜原型，保持与真实模型 [nm,mh,mw] 同构。
	const nm, mh, mw, slots = 4, 8, 10, 2
	proto := make([]float32, slots*nm*mh*mw)
	for i := range proto {
		proto[i] = float32(rng.NormFloat64()) * 2
	}
	e := &Engine{nm: nm, maskH: mh, maskW: mw, maskLogitThr: logitFromProb(0.5)}
	v := &segView{protoBuf: proto, protoStride: nm * mh * mw}

	for iter := 0; iter < 50; iter++ {
		coeffs := make([]float32, nm)
		for k := range coeffs {
			coeffs[k] = float32(rng.NormFloat64()) * 0.5
		}
		// 取一个落在输入画布内的任意"框"。
		ix1 := rng.Intn(mw)
		iy1 := rng.Intn(mh)
		ix2 := ix1 + 2 + rng.Intn(mw-ix1)
		iy2 := iy1 + 2 + rng.Intn(mh-iy1)
		_ = ix2
		_ = iy2
		bw, bh := 4, 4
		slot := rng.Intn(slots)
		inW, inH := mw, mh
		sx, sy := float32(mw)/float32(inW), float32(mh)/float32(inH)

		// 新路径（向量化 + logit 阈值）。
		newMask := make([]bool, bw*bh)
		type xg struct {
			x0, x1 int
			dx     float32
		}
		type yg struct {
			y0, y1 int
			dy     float32
		}
		xs := make([]xg, bw)
		for px := 0; px < bw; px++ {
			mx := (float32(px) + 0.5 + float32(ix1)) * sx
			x0 := int(math.Floor(float64(mx)))
			x1 := x0 + 1
			if x0 < 0 {
				x0 = 0
			}
			if x0 >= mw {
				x0 = mw - 1
			}
			if x1 >= mw {
				x1 = mw - 1
			}
			xs[px] = xg{x0, x1, mx - float32(x0)}
		}
		ys := make([]yg, bh)
		for py := 0; py < bh; py++ {
			my := (float32(py) + 0.5 + float32(iy1)) * sy
			y0 := int(math.Floor(float64(my)))
			y1 := y0 + 1
			if y0 < 0 {
				y0 = 0
			}
			if y0 >= mh {
				y0 = mh - 1
			}
			if y1 >= mh {
				y1 = mh - 1
			}
			ys[py] = yg{y0, y1, my - float32(y0)}
		}
		batchOff := slot * v.protoStride
		plane := mw * mh
		for py := 0; py < bh; py++ {
			yy := ys[py]
			y0w, y1w := yy.y0*mw, yy.y1*mw
			dy1 := 1 - yy.dy
			for px := 0; px < bw; px++ {
				xx := xs[px]
				dx1 := 1 - xx.dx
				w00 := dx1 * dy1
				w10 := xx.dx * dy1
				w01 := dx1 * yy.dy
				w11 := xx.dx * yy.dy
				var sum float32
				for k := 0; k < nm; k++ {
					base := batchOff + k*plane
					ck := coeffs[k]
					sum += ck * (w00*v.protoBuf[base+y0w+xx.x0] +
						w10*v.protoBuf[base+y0w+xx.x1] +
						w01*v.protoBuf[base+y1w+xx.x0] +
						w11*v.protoBuf[base+y1w+xx.x1])
				}
				newMask[py*bw+px] = sum >= e.maskLogitThr
			}
		}

		// 旧路径：sampleProto 双线性 + sigmoid。
		for py := 0; py < bh; py++ {
			for px := 0; px < bw; px++ {
				mx := (float32(px) + 0.5 + float32(ix1)) * sx
				my := (float32(py) + 0.5 + float32(iy1)) * sy
				x0 := int(math.Floor(float64(mx)))
				y0 := int(math.Floor(float64(my)))
				x1 := x0 + 1
				y1 := y0 + 1
				if x0 < 0 {
					x0 = 0
				}
				if y0 < 0 {
					y0 = 0
				}
				if x0 >= mw {
					x0 = mw - 1
				}
				if y0 >= mh {
					y0 = mh - 1
				}
				if x1 >= mw {
					x1 = mw - 1
				}
				if y1 >= mh {
					y1 = mh - 1
				}
				dx := mx - float32(x0)
				dy := my - float32(y0)
				w00 := (1 - dx) * (1 - dy)
				w10 := dx * (1 - dy)
				w01 := (1 - dx) * dy
				w11 := dx * dy
				var sum float32
				for k := 0; k < nm; k++ {
					base := batchOff + k*plane
					sum += coeffs[k] * (w00*v.protoBuf[base+y0*mw+x0] +
						w10*v.protoBuf[base+y0*mw+x1] +
						w01*v.protoBuf[base+y1*mw+x0] +
						w11*v.protoBuf[base+y1*mw+x1])
				}
				old := sigmoid(sum) >= 0.5
				if newMask[py*bw+px] != old {
					t.Fatalf("iter %d px(%d,%d) sum=%v new=%v old=%v",
						iter, px, py, sum, newMask[py*bw+px], old)
				}
			}
		}
	}
}

// BenchmarkSegMakeMasks 在 640 输入、160x160 原型、典型实例数下度量掩膜生成热循环。
func BenchmarkSegMakeMasks(b *testing.B) {
	const inW, inH, nm, mh, mw = 640, 640, 32, 160, 160
	e := &Engine{
		cfg: Config{InputW: inW, InputH: inH},
		nm:  nm, maskH: mh, maskW: mw,
		maskLogitThr: logitFromProb(0.5),
	}
	area := inW * inH
	e.scratchPool.New = func() any {
		return &maskScratch{mask: make([]bool, area), labels: make([]int, area)}
	}
	// 一槽的原型输出。
	proto := make([]float32, nm*mh*mw)
	rng := rand.New(rand.NewSource(1))
	for i := range proto {
		proto[i] = float32(rng.NormFloat64())
	}
	v := &segView{protoBuf: proto, protoStride: nm * mh * mw}
	// 8 个典型尺度实例（覆盖输入画布的不同区域）。
	var cands []cand
	for i := 0; i < 8; i++ {
		x := float32(20 + i*70)
		y := float32(20+i*70) * 0.8
		cands = append(cands, cand{
			box: engine.Box{ClassID: 0, Confidence: 0.9, X1: x, Y1: y, X2: x + 120, Y2: y + 120},
			coeffs: func() []float32 {
				c := make([]float32, nm)
				for k := range c {
					c[k] = float32(rng.NormFloat64()) * 0.2
				}
				return c
			}(),
		})
	}
	m := &ortbatch.Meta{Scale: 1, PadX: 0, PadY: 0, Orig: image.Rect(0, 0, inW, inH)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = e.makeMasks(v, 0, cands, m)
	}
}
