package data

import (
	"fmt"

	"github.com/andaoai/go-infer/internal/engine"
)

// ObjectsFromResult 把任意引擎结果统一转成 []Object。
//
// 这是全仓唯一一处对具体结果类型（DetectionResult/SegmentationResult/...）
// 做 type switch 的地方；采集、校验、指标都只消费 []Object，从而新任务类型
// 只需在这里加一个 case。未知结果类型返回 error。
func ObjectsFromResult(r engine.Result) ([]Object, error) {
	switch v := r.(type) {
	case *engine.DetectionResult:
		objs := make([]Object, 0, len(v.Detections))
		for _, d := range v.Detections {
			objs = append(objs, Object{
				ClassID:    d.ClassID,
				Confidence: d.Confidence,
				BBox:       BBox{X1: d.X1, Y1: d.Y1, X2: d.X2, Y2: d.Y2},
			})
		}
		return objs, nil
	case *engine.SegmentationResult:
		objs := make([]Object, 0, len(v.Instances))
		for _, ins := range v.Instances {
			rings := make([][]Point, 0, len(ins.Mask))
			for _, ring := range ins.Mask {
				pts := make([]Point, 0, len(ring))
				for _, p := range ring {
					pts = append(pts, Point{X: p.X, Y: p.Y})
				}
				rings = append(rings, pts)
			}
			objs = append(objs, Object{
				ClassID:    ins.ClassID,
				Confidence: ins.Confidence,
				BBox:       BBox{X1: ins.X1, Y1: ins.Y1, X2: ins.X2, Y2: ins.Y2},
				Rings:      rings,
			})
		}
		return objs, nil
	default:
		if r == nil {
			return nil, fmt.Errorf("nil result")
		}
		return nil, fmt.Errorf("unsupported result type %T for task %s", r, r.Task())
	}
}
