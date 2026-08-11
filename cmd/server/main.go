// Command go-infer 启动推理 HTTP 服务。
//
// 它是引擎无关的：所有推理逻辑由实现了 engine.Engine 的具体引擎提供，
// 本程序只负责加载 ONNX Runtime、按配置构造并注册引擎、启动 HTTP。
// 当前内置一个 ONNX Runtime YOLO 检测引擎，后续可在此注册更多后端/算法。
package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/andaoai/go-infer/internal/api"
	"github.com/andaoai/go-infer/internal/appcfg"
	"github.com/andaoai/go-infer/internal/capture"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/detect"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/seg"
	"github.com/andaoai/go-infer/internal/format"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/ortenv"
	"github.com/andaoai/go-infer/internal/storage"
	"github.com/andaoai/go-infer/internal/storage/local"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	var (
		name        = flag.String("name", "yolov8n", "检测引擎实例名（用于 ?engine= 选择）")
		modelPath   = flag.String("model", "models/yolov8n.onnx", "检测 ONNX 模型路径")
		segName     = flag.String("seg-name", "yolov8n-seg", "分割引擎实例名")
		segModel    = flag.String("seg-model", "models/yolov8n-seg.onnx", "分割 ONNX 模型路径；文件不存在则跳过该引擎")
		noSeg       = flag.Bool("no-seg", false, "禁用分割引擎（即使模型存在）")
		classes     = flag.String("classes", "", "类别名，逗号分隔；留空则看 -classes-file")
		classesFile = flag.String("classes-file", "models/coco.names", "类别名文件，一行一个；空字符串表示不读")
		numClass    = flag.Int("nc", 0, "类别数（前两者均未提供时用）")
		imgsz       = flag.Int("imgsz", 640, "模型输入尺寸（正方形）")
		conf        = flag.Float64("conf", 0.25, "默认置信度阈值")
		iou         = flag.Float64("iou", 0.45, "NMS IoU 阈值")
		maskThr     = flag.Float64("mask-thr", 0.5, "分割掩膜二值化阈值")
		addr        = flag.String("addr", ":8080", "监听地址")
		ortLib      = flag.String("ort-lib", "", "libonnxruntime.so 路径；留空则自动查找")

		captureDir    = flag.String("capture-dir", "", "推理采集池根目录（本地目录，如 dataset）；留空则不采集")
		capturePool   = flag.String("capture-pool", "", "采集池在 -capture-dir 下的前缀（默认空，直接落在 <dir>/<engine>/<date>）")
		captureRate   = flag.Float64("capture-rate", 0.1, "普通样本采样概率 0~1；无检测/低置信度样本必存")
		captureLow    = flag.Float64("capture-low-conf", 0.25, "最高置信度低于该值的不确定样本必存")
		captureQuota  = flag.String("capture-quota", "5GB", "采集池总容量上限（如 500MB/5GB/0 表示不限）")
		captureBuffer = flag.Int("capture-buffer", 256, "异步采集队列长度")
	)
	flag.Parse()

	if lib := ortenv.FindLib(*ortLib); lib != "" {
		log.Printf("使用 ONNX Runtime: %s", lib)
	}
	if err := ortenv.Init(*ortLib); err != nil {
		log.Fatalf("初始化 ONNX Runtime: %v", err)
	}
	defer ort.DestroyEnvironment()

	classList, err := appcfg.LoadClasses(*classes, *classesFile, *numClass)
	if err != nil {
		log.Fatalf("读取类别: %v", err)
	}

	eng, err := detect.New(detect.Config{
		Name:       *name,
		ModelPath:  *modelPath,
		InputW:     *imgsz,
		InputH:     *imgsz,
		Classes:    classList,
		ConfThresh: float32(*conf),
		IoUThresh:  float32(*iou),
	})
	if err != nil {
		log.Fatalf("创建引擎 %s: %v", *name, err)
	}
	defer eng.Close()

	srv := api.NewServer()
	srv.Register(eng)

	// 分割引擎：模型文件存在才注册，缺失不影响检测服务。
	if !*noSeg {
		if segEng, err := maybeLoadSeg(*segModel, *segName, *imgsz, classList, *conf, *iou, *maskThr); err != nil {
			log.Printf("警告: 分割引擎加载失败，已跳过: %v", err)
		} else if segEng != nil {
			srv.Register(segEng)
			defer segEng.Close()
			log.Printf("已注册分割引擎: %s (%s)", segEng.Name(), *segModel)
		}
	}
	// 在此 srv.Register(...) 更多引擎（TensorRT/NCNN/llama.cpp ...）。

	// 推理采集（原图 + YOLO 伪标签），默认关闭。
	// store/codec 在采集器与采集池浏览器之间共享。
	var capStore storage.Storage
	var capCodec format.Codec
	if *captureDir != "" {
		st, err := local.New(*captureDir)
		if err != nil {
			log.Fatalf("初始化采集存储: %v", err)
		}
		capStore, capCodec = st, yolo.New()
	}
	if rec, err := newCaptureRecorder(capStore, capCodec, *capturePool, *captureRate, float32(*captureLow), *captureQuota, *captureBuffer); err != nil {
		log.Fatalf("初始化采集器: %v", err)
	} else if rec != nil {
		srv.SetRecorder(recAdapter{rec})
		defer rec.Close()
		srv.SetBrowser(&captureBrowser{st: capStore, codec: capCodec, poolRoot: *capturePool})
		log.Printf("推理采集已开启: dir=%s pool=%s rate=%.2f low-conf=%.2f quota=%s",
			*captureDir, *capturePool, *captureRate, *captureLow, *captureQuota)
	}

	log.Printf("go-infer 服务启动于 %s | 默认引擎=%s task=%s framework=%s classes=%d",
		*addr, eng.Name(), eng.Task(), eng.Framework(), len(classList))
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}

// maybeLoadSeg 在模型文件存在时创建分割引擎；文件不存在返回 (nil, nil)。
func maybeLoadSeg(modelPath, name string, imgsz int, classes []string, conf, iou, maskThr float64) (engine.Engine, error) {
	if _, err := os.Stat(modelPath); err != nil {
		if os.IsNotExist(err) {
			log.Printf("未找到分割模型 %s，跳过分割引擎（用 -seg-model 指定或 -no-seg 关闭此提示）", modelPath)
			return nil, nil
		}
		return nil, err
	}
	return seg.New(seg.Config{
		Name:       name,
		ModelPath:  modelPath,
		InputW:     imgsz,
		InputH:     imgsz,
		Classes:    classes,
		ConfThresh: float32(conf),
		IoUThresh:  float32(iou),
		MaskThresh: float32(maskThr),
	})
}

// 保留 engine 引用以便未来在此文件中引用接口类型。
var (
	_ engine.Engine = (*detect.Engine)(nil)
	_ engine.Engine = (*seg.Engine)(nil)
)

// newCaptureRecorder 构造采集器；store 为 nil 时返回 nil 表示不采集。
func newCaptureRecorder(st storage.Storage, codec format.Codec, poolRoot string, rate float64, lowConf float32, quota string, buffer int) (*capture.Recorder, error) {
	if st == nil || codec == nil {
		return nil, nil
	}
	quotaBytes, err := parseSize(quota)
	if err != nil {
		return nil, err
	}
	return capture.New(capture.Config{
		Store:      st,
		Codec:      codec,
		PoolRoot:   poolRoot,
		Rate:       rate,
		LowConf:    lowConf,
		QuotaBytes: quotaBytes,
		Buffer:     buffer,
	})
}

// recAdapter 把 api.CaptureSample 透传成 capture.Sample。
type recAdapter struct{ r *capture.Recorder }

func (a recAdapter) Record(s api.CaptureSample) {
	a.r.Record(capture.Sample{
		Engine: s.Engine,
		Image:  s.Image,
		W:      s.W,
		H:      s.H,
		ImgExt: s.ImgExt,
		Result: s.Result,
	})
}

// parseSize 解析 "500MB"/"5GB"/"1024"/"0" 为字节数。
func parseSize(s string) (int64, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == "0" {
		return 0, nil
	}
	mult := int64(1)
	upper := strings.ToUpper(s)
	switch {
	case strings.HasSuffix(upper, "GB"):
		mult, s = 1<<30, s[:len(s)-2]
	case strings.HasSuffix(upper, "MB"):
		mult, s = 1<<20, s[:len(s)-2]
	case strings.HasSuffix(upper, "KB"):
		mult, s = 1<<10, s[:len(s)-2]
	}
	n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return 0, err
	}
	return n * mult, nil
}
