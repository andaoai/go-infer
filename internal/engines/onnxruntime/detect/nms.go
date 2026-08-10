package detect

import (
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/preprocess"
)

// iou 是 preprocess.IoU 的 engine.Box 便捷封装。
func iou(a, b engine.Box) float32 {
	return preprocess.IoU(
		preprocess.NMSBox{X1: a.X1, Y1: a.Y1, X2: a.X2, Y2: a.Y2},
		preprocess.NMSBox{X1: b.X1, Y1: b.Y1, X2: b.X2, Y2: b.Y2},
	)
}

// nms 委托给 preprocess.NMS，在 engine.Box 与 preprocess.NMSBox 间转换。
func nms(boxes []engine.Box, iouThresh float32) []engine.Box {
	nb := make([]preprocess.NMSBox, len(boxes))
	for i, b := range boxes {
		nb[i] = preprocess.NMSBox{
			ClassID: b.ClassID, Confidence: b.Confidence,
			X1: b.X1, Y1: b.Y1, X2: b.X2, Y2: b.Y2,
		}
	}
	kept := preprocess.NMS(nb, iouThresh)
	out := make([]engine.Box, len(kept))
	for i, b := range kept {
		out[i] = engine.Box{
			ClassID: b.ClassID, Confidence: b.Confidence,
			X1: b.X1, Y1: b.Y1, X2: b.X2, Y2: b.Y2,
		}
	}
	return out
}
