package main

import (
	"context"
	"image"
	"os"
	"sync"
	"testing"

	"github.com/andaoai/go-infer/internal/appcfg"
	"github.com/andaoai/go-infer/internal/dataset"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/detect"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/seg"
	"github.com/andaoai/go-infer/internal/eval"
	"github.com/andaoai/go-infer/internal/ortenv"
)

var ortOnce sync.Once

func initORT(t *testing.T) {
	t.Helper()
	if ortenv.FindLib("") == "" {
		t.Skip("未找到 libonnxruntime.so，跳过（执行 make ort 或设 ORT_LIB_PATH）")
	}
	ortOnce.Do(func() {
		if err := ortenv.Init(""); err != nil {
			t.Fatalf("初始化 ONNX Runtime: %v", err)
		}
	})
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func buildGTs(samples []dataset.Sample) []eval.GroundTruth {
	gts := make([]eval.GroundTruth, 0, len(samples)*4)
	for i, s := range samples {
		for _, o := range s.Objects {
			var rings [][]image.Point
			if len(o.Polygon) > 0 {
				rings = [][]image.Point{o.Polygon}
			}
			gts = append(gts, eval.GroundTruth{
				ImageID: i, ClassID: o.ClassID,
				X1: o.X1, Y1: o.Y1, X2: o.X2, Y2: o.Y2,
				Rings: rings, W: s.W, H: s.H,
			})
		}
	}
	return gts
}

// TestValidateOnCoco128Subset 用真实模型在 coco128-seg 的前 16 张上端到端校验，
// 断言指标处于合理下限（防坐标错位/后处理退化的回归）。缺数据/模型时自动 skip。
func TestValidateOnCoco128Subset(t *testing.T) {
	if testing.Short() {
		t.Skip("-short 模式跳过大数据集校验")
	}
	initORT(t)

	dataDir := "../../testdata/coco128-seg"
	if !exists(dataDir + "/images/train2017") {
		t.Skipf("缺少数据集 %s，跳过（下载 coco128-seg.zip 解压到 testdata/）", dataDir)
	}

	classes, err := appcfg.LoadClasses("", "../../models/coco.names", 0)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := dataset.Load(dataDir, "train2017")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) > 16 {
		samples = samples[:16]
	}
	gts := buildGTs(samples)

	if exists("../../models/yolov8n.onnx") {
		eng, err := detect.New(detect.Config{
			Name: "det-test", ModelPath: "../../models/yolov8n.onnx",
			InputW: 640, InputH: 640, Classes: classes,
			ConfThresh: 0.001, IoUThresh: 0.7,
		})
		if err != nil {
			t.Fatalf("加载检测模型: %v", err)
		}
		preds := make([]eval.Prediction, 0, 64)
		for i, s := range samples {
			res, err := eng.Run(context.Background(), &engine.Request{Image: s.Image})
			if err != nil {
				t.Fatal(err)
			}
			for _, d := range res.(*engine.DetectionResult).Detections {
				preds = append(preds, eval.Prediction{
					ImageID: i, ClassID: d.ClassID, Confidence: d.Confidence,
					X1: d.X1, Y1: d.Y1, X2: d.X2, Y2: d.Y2,
				})
			}
		}
		_, map50 := eval.AP(preds, gts, 0.5, false)
		t.Logf("det box mAP@.5 = %.4f", map50)
		if map50 < 0.4 {
			t.Errorf("检测 mAP@.5 = %.3f，低于回归下限 0.4", map50)
		}
		eng.Close()
	} else {
		t.Skip("缺少 models/yolov8n.onnx")
	}

	if exists("../../models/yolov8n-seg.onnx") {
		eng, err := seg.New(seg.Config{
			Name: "seg-test", ModelPath: "../../models/yolov8n-seg.onnx",
			InputW: 640, InputH: 640, Classes: classes,
			ConfThresh: 0.001, IoUThresh: 0.7, MaskThresh: 0.5,
		})
		if err != nil {
			t.Fatalf("加载分割模型: %v", err)
		}
		preds := make([]eval.Prediction, 0, 64)
		for i, s := range samples {
			res, err := eng.Run(context.Background(), &engine.Request{Image: s.Image})
			if err != nil {
				t.Fatal(err)
			}
			for _, ins := range res.(*engine.SegmentationResult).Instances {
				var rings [][]image.Point
				for _, ring := range ins.Mask {
					pts := make([]image.Point, 0, len(ring))
					for _, p := range ring {
						pts = append(pts, image.Pt(int(p.X), int(p.Y)))
					}
					rings = append(rings, pts)
				}
				preds = append(preds, eval.Prediction{
					ImageID: i, ClassID: ins.ClassID, Confidence: ins.Confidence,
					X1: ins.X1, Y1: ins.Y1, X2: ins.X2, Y2: ins.Y2, Rings: rings,
				})
			}
		}
		_, box50 := eval.AP(preds, gts, 0.5, false)
		_, mask50 := eval.AP(preds, gts, 0.5, true)
		t.Logf("seg box mAP@.5 = %.4f  mask mAP@.5 = %.4f", box50, mask50)
		if box50 < 0.35 {
			t.Errorf("分割 box mAP@.5 = %.3f，低于回归下限 0.35", box50)
		}
		if mask50 < 0.3 {
			t.Errorf("分割 mask mAP@.5 = %.3f，低于回归下限 0.3", mask50)
		}
		eng.Close()
	} else {
		t.Skip("缺少 models/yolov8n-seg.onnx")
	}
}
