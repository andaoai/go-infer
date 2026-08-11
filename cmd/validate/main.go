// Command validate 在带标注的数据集上校验检测/分割模型，输出 mAP。
//
// 通过注入的 format.Codec 读取标签（首个内置实现为 YOLO），通过注入的
// storage.Storage 读取字节（首个内置实现为本地 FS），用引擎跑推理后把结果
// 统一成 data.Object，交给引擎/格式无关的 internal/validate 评分。分别计算：
//   - 检测模型：box mAP@.5 / mAP@.50:.95
//   - 分割模型：box mAP@.5 / mAP@.50:.95、mask mAP@.5 / mAP@.50:.95
//
// 模型文件缺失时自动跳过对应引擎，不报错。
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/andaoai/go-infer/internal/appcfg"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/detect"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/seg"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/ortenv"
	"github.com/andaoai/go-infer/internal/storage/local"
	"github.com/andaoai/go-infer/internal/validate"
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

	opts := validate.Options{
		Store:   st,
		Codec:   yolo.New(),
		Root:    dsPrefix,
		Split:   *split,
		Limit:   *limit,
		Classes: classList,
	}

	// 检测模型（文件缺失则跳过）。
	if _, err := os.Stat(*detModel); err == nil {
		eng, err := detect.New(detect.Config{
			Name: "det-val", ModelPath: *detModel, InputW: *imgsz, InputH: *imgsz,
			Classes: classList, ConfThresh: float32(*conf), IoUThresh: float32(*iou),
		})
		if err != nil {
			log.Fatalf("加载检测模型: %v", err)
		}
		defer eng.Close()
		opts.Jobs = append(opts.Jobs, validate.Job{
			Name: "检测模型", Engine: eng, Metrics: []validate.MetricSpec{{Label: "box"}},
		})
	} else {
		fmt.Printf("== 跳过检测模型（%s 不存在）==\n\n", *detModel)
	}

	// 分割模型（文件缺失则跳过）：box + mask 两个指标维度。
	if _, err := os.Stat(*segModel); err == nil {
		eng, err := seg.New(seg.Config{
			Name: "seg-val", ModelPath: *segModel, InputW: *imgsz, InputH: *imgsz,
			Classes: classList, ConfThresh: float32(*conf), IoUThresh: float32(*iou),
			MaskThresh: float32(*maskThr),
		})
		if err != nil {
			log.Fatalf("加载分割模型: %v", err)
		}
		defer eng.Close()
		opts.Jobs = append(opts.Jobs, validate.Job{
			Name: "分割模型", Engine: eng,
			Metrics: []validate.MetricSpec{{Label: "box"}, {Label: "mask", UseMask: true}},
		})
	} else {
		fmt.Printf("== 跳过分割模型（%s 不存在）==\n\n", *segModel)
	}

	if len(opts.Jobs) == 0 {
		log.Fatalf("没有可校验的模型（det/seg 模型文件均不存在）")
	}

	report, err := validate.Run(context.Background(), opts)
	if err != nil {
		log.Fatalf("校验失败: %v", err)
	}

	fmt.Println()
	for _, jr := range report.Jobs {
		avgMs := 0.0
		if report.Images > 0 {
			avgMs = float64(jr.TookMs) / float64(report.Images)
		}
		for _, m := range jr.Metrics {
			title := jr.Name
			if len(jr.Metrics) > 1 {
				title = fmt.Sprintf("%s %s", jr.Name, m.Label)
			}
			printMetric(title, m, jr, avgMs)
		}
	}
}

func printMetric(title string, m validate.MetricResult, jr validate.JobReport, avgMs float64) {
	fmt.Printf("==== %s ====\n", title)
	fmt.Printf("mAP@.5      = %.4f\n", m.MAP50)
	fmt.Printf("mAP@.5:.95  = %.4f\n", m.MAP5095)
	fmt.Printf("推理 %d ms（%.1f ms/张），预测 %d 个实例 / GT %d 个\n",
		jr.TookMs, avgMs, jr.Predictions, jr.GroundTruths)
	if len(m.PerClass) > 0 {
		fmt.Println("  各类 AP@.5:")
		for _, c := range m.PerClass {
			fmt.Printf("    %-16s %.4f\n", c.ClassName, c.AP50)
		}
	}
	fmt.Println()
}
