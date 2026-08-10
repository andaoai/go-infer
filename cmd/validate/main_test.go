package main

import (
	"bytes"
	"context"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/andaoai/go-infer/internal/appcfg"
	"github.com/andaoai/go-infer/internal/data"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/detect"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/seg"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/metric"
	"github.com/andaoai/go-infer/internal/ortenv"
	"github.com/andaoai/go-infer/internal/storage/local"
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

// loadSamples 用 codec + storage 加载前 limit 张样本，组装 GT。
func loadSamples(t *testing.T, dataDir, split string, limit int) ([]data.Sample, []metric.GroundTruth) {
	t.Helper()
	storeRoot := filepath.Dir(filepath.Clean(dataDir))
	dsPrefix := filepath.Base(filepath.Clean(dataDir))
	st, err := local.New(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	codec := yolo.New()
	ctx := context.Background()
	refs, err := codec.ListImages(ctx, st, dsPrefix, split)
	if err != nil {
		t.Fatal(err)
	}
	if limit > 0 && limit < len(refs) {
		refs = refs[:limit]
	}
	samples := make([]data.Sample, len(refs))
	gts := make([]metric.GroundTruth, 0, len(refs)*4)
	for i, ref := range refs {
		rc, err := st.Get(ctx, ref.ImageKey)
		if err != nil {
			t.Fatal(err)
		}
		img, _, err := image.Decode(rc)
		rc.Close()
		if err != nil {
			t.Fatalf("解码图片 %s: %v", ref.ImageKey, err)
		}
		b := img.Bounds()
		w, h := b.Dx(), b.Dy()
		var objs []data.Object
		if lrc, err := st.Get(ctx, ref.LabelKey); err == nil {
			buf := new(bytes.Buffer)
			buf.ReadFrom(lrc)
			lrc.Close()
			if objs, err = codec.Decode(buf.Bytes(), w, h); err != nil {
				t.Fatalf("解码标签 %s: %v", ref.LabelKey, err)
			}
		}
		samples[i] = data.Sample{Key: ref.ImageKey, Image: img, W: w, H: h, Objects: objs}
		for _, o := range objs {
			gts = append(gts, metric.GroundTruth{ImageID: i, Object: o, W: w, H: h})
		}
	}
	return samples, gts
}

func runPreds(t *testing.T, eng engine.Engine, samples []data.Sample) []metric.Prediction {
	t.Helper()
	preds := make([]metric.Prediction, 0, len(samples)*4)
	for i, s := range samples {
		res, err := eng.Run(context.Background(), &engine.Request{Image: s.Image})
		if err != nil {
			t.Fatalf("第 %d 张推理失败: %v", i, err)
		}
		objs, err := data.ObjectsFromResult(res)
		if err != nil {
			t.Fatal(err)
		}
		for _, o := range objs {
			preds = append(preds, metric.Prediction{ImageID: i, Object: o})
		}
	}
	return preds
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
	samples, gts := loadSamples(t, dataDir, "train2017", 16)

	if exists("../../models/yolov8n.onnx") {
		eng, err := detect.New(detect.Config{
			Name: "det-test", ModelPath: "../../models/yolov8n.onnx",
			InputW: 640, InputH: 640, Classes: classes,
			ConfThresh: 0.001, IoUThresh: 0.7,
		})
		if err != nil {
			t.Fatalf("加载检测模型: %v", err)
		}
		preds := runPreds(t, eng, samples)
		_, map50 := metric.AP(preds, gts, 0.5, false)
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
		preds := runPreds(t, eng, samples)
		_, box50 := metric.AP(preds, gts, 0.5, false)
		_, mask50 := metric.AP(preds, gts, 0.5, true)
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
