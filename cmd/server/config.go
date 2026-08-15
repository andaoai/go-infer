package main

import (
	"flag"
	"fmt"
	"runtime"
	"strings"
	"time"

	"github.com/andaoai/go-infer/internal/video"
)

// Config 是 go-infer 服务的全部启动参数，由 parseConfig 从命令行解析。
// 拆分自原 main.go 里 ~40 个 flag，让 main 只负责 ORT 初始化、装配与启动。
type Config struct {
	// 引擎
	Name, Model, SegName, SegModel string
	NoSeg                          bool
	Classes, ClassesFile           string
	NumClass                       int
	ImgSize                        int
	Conf, IoU, MaskThr             float64
	Addr, OrtLib                   string

	// 调度器
	SchedWorkers  int
	SchedQueue    int
	SchedMaxBatch int
	SchedMaxWait  time.Duration

	// 采集
	CaptureDir, CapturePool     string
	CaptureRate, CaptureLowConf float64
	CaptureQuota                string
	CaptureBuffer               int

	// 网页校验
	ValidateDir, ValidateMaxUp string
	ValidateTests              map[string]testsetConfig

	// 视频
	VideoFFmpeg, VideoScratch string
	VideoMaxSessions          int
	VideoMaxUp                string
	VideoSources              []video.Source
}

// parseConfig 解析命令行参数并填充 Config。args 通常为 os.Args[1:]。
func parseConfig(args []string) (*Config, error) {
	fs := flag.NewFlagSet("go-infer", flag.ContinueOnError)
	var (
		name        = fs.String("name", "yolov8n", "检测引擎实例名（用于 ?engine= 选择）")
		modelPath   = fs.String("model", "models/yolov8n.onnx", "检测 ONNX 模型路径")
		segName     = fs.String("seg-name", "yolov8n-seg", "分割引擎实例名")
		segModel    = fs.String("seg-model", "models/yolov8n-seg.onnx", "分割 ONNX 模型路径；文件不存在则跳过该引擎")
		noSeg       = fs.Bool("no-seg", false, "禁用分割引擎（即使模型存在）")
		classes     = fs.String("classes", "", "类别名，逗号分隔；留空则看 -classes-file")
		classesFile = fs.String("classes-file", "models/coco.names", "类别名文件，一行一个；空字符串表示不读")
		numClass    = fs.Int("nc", 0, "类别数（前两者均未提供时用）")
		imgsz       = fs.Int("imgsz", 640, "模型输入尺寸（正方形）")
		conf        = fs.Float64("conf", 0.25, "默认置信度阈值")
		iou         = fs.Float64("iou", 0.45, "NMS IoU 阈值")
		maskThr     = fs.Float64("mask-thr", 0.5, "分割掩膜二值化阈值")
		addr        = fs.String("addr", ":8080", "监听地址")
		ortLib      = fs.String("ort-lib", "", "libonnxruntime.so 路径；留空则自动查找")

		schedWorkers  = fs.Int("sched-workers", 0, "并发预处理/推理 worker 数；0 取 CPU 核数")
		schedQueue    = fs.Int("sched-queue", 64, "请求队列上限；满了立即返回 503")
		schedMaxBatch = fs.Int("sched-max-batch", 0, "dynamic batching 上限；0 用模型自身上限（固定 batch=1 模型忽略）")
		schedMaxWait  = fs.String("sched-max-wait", "5ms", "凑满一批最多等待时长（如 5ms/0）；batch=1 时忽略")

		captureDir    = fs.String("capture-dir", "", "推理采集池根目录（本地目录，如 dataset）；留空则不采集")
		capturePool   = fs.String("capture-pool", "", "采集池在 -capture-dir 下的前缀（默认空，直接落在 <dir>/<engine>/<date>）")
		captureRate   = fs.Float64("capture-rate", 0.1, "普通样本采样概率 0~1；无检测/低置信度样本必存")
		captureLow    = fs.Float64("capture-low-conf", 0.25, "最高置信度低于该值的不确定样本必存")
		captureQuota  = fs.String("capture-quota", "5GB", "采集池总容量上限（如 500MB/5GB/0 表示不限）")
		captureBuffer = fs.Int("capture-buffer", 256, "异步采集队列长度")

		validateDir   = fs.String("validate-dir", "runs/validate", "网页校验的临时文件目录（上传模型/数据集解压）")
		validateMaxUp = fs.String("validate-max-upload", "2GB", "校验上传体积上限（模型+数据集，如 500MB/2GB）")
		validateTests = &testsetFlag{}

		videoFFmpeg  = fs.String("video-ffmpeg", "ffmpeg", "ffmpeg 可执行文件路径；找不到则视频 tab 自动关闭")
		videoScratch = fs.String("video-scratch", "runs/video", "视频上传临时目录")
		videoMaxSess = fs.Int("video-max-sessions", 3, "并发视频推理会话上限")
		videoMaxUp   = fs.String("video-max-upload", "1GB", "上传视频体积上限（如 500MB/1GB）")
		videoSources = &videoSourceFlag{}
	)
	fs.Var(validateTests, "validate-testset", "网页可选默认测试集，格式 name=path:split；可重复指定")
	fs.Var(videoSources, "video-source", "网页视频源，格式 name=url；可重复指定（同内置名则覆盖）")
	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	workers := *schedWorkers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	var maxWait time.Duration
	if strings.TrimSpace(*schedMaxWait) != "" && *schedMaxWait != "0" {
		d, err := time.ParseDuration(*schedMaxWait)
		if err != nil {
			return nil, fmt.Errorf("解析 -sched-max-wait: %w", err)
		}
		maxWait = d
	}
	return &Config{
		Name: *name, Model: *modelPath, SegName: *segName, SegModel: *segModel,
		NoSeg: *noSeg, Classes: *classes, ClassesFile: *classesFile, NumClass: *numClass,
		ImgSize: *imgsz, Conf: *conf, IoU: *iou, MaskThr: *maskThr,
		Addr: *addr, OrtLib: *ortLib,

		SchedWorkers: workers, SchedQueue: *schedQueue,
		SchedMaxBatch: *schedMaxBatch, SchedMaxWait: maxWait,

		CaptureDir: *captureDir, CapturePool: *capturePool,
		CaptureRate: *captureRate, CaptureLowConf: *captureLow,
		CaptureQuota: *captureQuota, CaptureBuffer: *captureBuffer,

		ValidateDir: *validateDir, ValidateMaxUp: *validateMaxUp,
		ValidateTests: validateTests.toMap(),

		VideoFFmpeg: *videoFFmpeg, VideoScratch: *videoScratch,
		VideoMaxSessions: *videoMaxSess, VideoMaxUp: *videoMaxUp,
		VideoSources: videoSources.list(),
	}, nil
}
