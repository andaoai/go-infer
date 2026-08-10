// Command validate 在带标注的数据集上校验检测/分割模型，输出 mAP。
//
// 通过注入的 format.Codec 读取标签（首个内置实现为 YOLO），通过注入的
// storage.Storage 读取字节（首个内置实现为本地 FS），用引擎跑推理后把结果
// 统一成 data.Object，交给引擎/格式无关的 metric 包评分。分别计算：
//   - 检测模型：box mAP@.5 / mAP@.50:.95
//   - 分割模型：box mAP@.5 / mAP@.50:.95、mask mAP@.5 / mAP@.50:.95
//
// 模型文件缺失时自动跳过对应引擎，不报错。
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/andaoai/go-infer/internal/appcfg"
	"github.com/andaoai/go-infer/internal/data"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/detect"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/seg"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/metric"
	"github.com/andaoai/go-infer/internal/ortenv"
	"github.com/andaoai/go-infer/internal/storage/local"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	var (
		dataDir  = flag.String("data", "testdata/coco128-seg", "数据集根目录（含 images/<split> 与 labels/<split>）")
		split    = flag.String("split", "train2017", "数据集划分（子目录名）")
		detModel = flag.String("det-model", "models/yolov8n.onnx", "检测 ONNX 模型路径；文件不存在则跳过检测校验")
		segModel = flag.String("seg-model", "models/yolov8n-seg.onnx", "分割 ONNX 模型路径；文件不存在则跳过分割校验")
		classes  = flag.String("classes", "", "类别名，逗号分隔；留空则用 -classes-file")
		clsFile  = flag.String("classes-file", "models/coco.names", "类别名文件，一行一个")
		numClass = flag.Int("nc", 0, "类别数（前两者均未给时用）")
		imgsz    = flag.Int("imgsz", 640, "模型输入尺寸（正方形）")
		conf     = flag.Float64("conf", 0.001, "推理置信度阈值（评测建议低阈值，默认 0.001）")
		iou      = flag.Float64("iou", 0.7, "NMS IoU 阈值（评测建议高于推理默认，默认 0.7）")
		maskThr  = flag.Float64("mask-thr", 0.5, "分割掩膜二值化阈值")
		limit    = flag.Int("limit", 0, "只校验前 N 张图（0=全量）")
		ortLib   = flag.String("ort-lib", "", "libonnxruntime.so 路径；留空自动查找")
	)
	flag.Parse()

	if err := ortenv.Init(*ortLib); err != nil {
		log.Fatalf("初始化 ONNX Runtime: %v", err)
	}
	defer ort.DestroyEnvironment()

	classList, err := appcfg.LoadClasses(*classes, *clsFile, *numClass)
	if err != nil {
		log.Fatalf("读取类别: %v", err)
	}

	// 数据集根所在的目录作为本地存储根，数据集根作为相对前缀。
	storeRoot := filepath.Dir(filepath.Clean(*dataDir))
	dsPrefix := filepath.Base(filepath.Clean(*dataDir))
	st, err := local.New(storeRoot)
	if err != nil {
		log.Fatalf("打开存储 %s: %v", storeRoot, err)
	}
	codec := yolo.New()
	ctx := context.Background()

	refs, err := codec.ListImages(ctx, st, dsPrefix, *split)
	if err != nil {
		log.Fatalf("枚举数据集: %v", err)
	}
	if *limit > 0 && *limit < len(refs) {
		refs = refs[:*limit]
	}
	if len(refs) == 0 {
		log.Fatalf("数据集 %s/%s 没有图片", *dataDir, *split)
	}
	log.Printf("加载 %d 张图（%d 类），开始校验...", len(refs), len(classList))

	// 加载图片与 GT（标签解码成 data.Object）。
	samples := make([]data.Sample, len(refs))
	gts := make([]metric.GroundTruth, 0, len(refs)*4)
	for i, ref := range refs {
		img, w, h := loadImage(ctx, st, ref.ImageKey)
		var objs []data.Object
		if rc, err := st.Get(ctx, ref.LabelKey); err == nil {
			buf := new(bytes.Buffer)
			buf.ReadFrom(rc)
			rc.Close()
			if objs, err = codec.Decode(buf.Bytes(), w, h); err != nil {
				log.Fatalf("解码标签 %s: %v", ref.LabelKey, err)
			}
		}
		samples[i] = data.Sample{Key: ref.ImageKey, Image: img, W: w, H: h, Objects: objs}
		for _, o := range objs {
			gts = append(gts, metric.GroundTruth{ImageID: i, Object: o, W: w, H: h})
		}
	}

	detPreds, detTook, detOK := runDetect(*detModel, *imgsz, classList, *conf, *iou, samples)
	segPreds, segTook, segOK := runSeg(*segModel, *imgsz, classList, *conf, *iou, *maskThr, samples)

	n := len(samples)
	fmt.Println()
	if detOK {
		printReport("检测模型 box", detPreds, gts, false, detTook, n, classList)
	} else {
		fmt.Printf("== 跳过检测模型（%s 不存在）==\n\n", *detModel)
	}
	if segOK {
		printReport("分割模型 box", segPreds, gts, false, segTook, n, classList)
		printReport("分割模型 mask", segPreds, gts, true, segTook, n, classList)
	} else {
		fmt.Printf("== 跳过分割模型（%s 不存在）==\n\n", *segModel)
	}
}

func loadImage(ctx context.Context, st *local.Storage, key string) (image.Image, int, int) {
	rc, err := st.Get(ctx, key)
	if err != nil {
		log.Fatalf("读取图片 %s: %v", key, err)
	}
	defer rc.Close()
	img, _, err := image.Decode(rc)
	if err != nil {
		log.Fatalf("解码图片 %s: %v", key, err)
	}
	b := img.Bounds()
	return img, b.Dx(), b.Dy()
}

func runDetect(modelPath string, imgsz int, classes []string, conf, iou float64, samples []data.Sample) ([]metric.Prediction, time.Duration, bool) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, 0, false
	}
	eng, err := detect.New(detect.Config{
		Name: "det-val", ModelPath: modelPath, InputW: imgsz, InputH: imgsz,
		Classes: classes, ConfThresh: float32(conf), IoUThresh: float32(iou),
	})
	if err != nil {
		log.Fatalf("加载检测模型: %v", err)
	}
	defer eng.Close()

	preds := make([]metric.Prediction, 0, len(samples)*4)
	var total time.Duration
	for i, s := range samples {
		res, err := eng.Run(context.Background(), &engine.Request{Image: s.Image})
		if err != nil {
			log.Fatalf("第 %d 张检测推理失败: %v", i, err)
		}
		dr := res.(*engine.DetectionResult)
		total += dr.Latency()
		objs, err := data.ObjectsFromResult(res)
		if err != nil {
			log.Fatalf("结果转换失败: %v", err)
		}
		for _, o := range objs {
			preds = append(preds, metric.Prediction{ImageID: i, Object: o})
		}
	}
	return preds, total, true
}

func runSeg(modelPath string, imgsz int, classes []string, conf, iou, maskThr float64, samples []data.Sample) ([]metric.Prediction, time.Duration, bool) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, 0, false
	}
	eng, err := seg.New(seg.Config{
		Name: "seg-val", ModelPath: modelPath, InputW: imgsz, InputH: imgsz,
		Classes: classes, ConfThresh: float32(conf), IoUThresh: float32(iou),
		MaskThresh: float32(maskThr),
	})
	if err != nil {
		log.Fatalf("加载分割模型: %v", err)
	}
	defer eng.Close()

	preds := make([]metric.Prediction, 0, len(samples)*4)
	var total time.Duration
	for i, s := range samples {
		res, err := eng.Run(context.Background(), &engine.Request{Image: s.Image})
		if err != nil {
			log.Fatalf("第 %d 张分割推理失败: %v", i, err)
		}
		sr := res.(*engine.SegmentationResult)
		total += sr.Latency()
		objs, err := data.ObjectsFromResult(res)
		if err != nil {
			log.Fatalf("结果转换失败: %v", err)
		}
		for _, o := range objs {
			preds = append(preds, metric.Prediction{ImageID: i, Object: o})
		}
	}
	return preds, total, true
}

func printReport(title string, preds []metric.Prediction, gts []metric.GroundTruth, useMask bool, took time.Duration, nImages int, classes []string) {
	perClass, mAP50 := metric.AP(preds, gts, 0.50, useMask)
	_, mAP5095 := metric.MAPOverThresholds(preds, gts, useMask)

	fmt.Printf("==== %s ====\n", title)
	fmt.Printf("mAP@.5      = %.4f\n", mAP50)
	fmt.Printf("mAP@.5:.95  = %.4f\n", mAP5095)
	avgMs := 0.0
	if nImages > 0 {
		avgMs = float64(took.Microseconds()) / float64(nImages) / 1000.0
	}
	fmt.Printf("推理总耗时 %v（%.1f ms/张），预测 %d 个实例 / GT %d 个\n",
		took, avgMs, len(preds), len(gts))

	type row struct {
		id   int
		name string
		ap   float64
	}
	var rows []row
	for id, ap := range perClass {
		name := fmt.Sprintf("class_%d", id)
		if id >= 0 && id < len(classes) {
			name = classes[id]
		}
		if hasGT(gts, id) {
			rows = append(rows, row{id, name, ap})
		}
	}
	sort.Slice(rows, func(a, b int) bool { return rows[a].ap > rows[b].ap })
	if len(rows) > 0 {
		fmt.Println("  各类 AP@.5:")
		for _, r := range rows {
			fmt.Printf("    %-16s %.4f\n", r.name, r.ap)
		}
	}
	fmt.Println()
}

func hasGT(gts []metric.GroundTruth, class int) bool {
	for _, g := range gts {
		if g.Object.ClassID == class {
			return true
		}
	}
	return false
}
