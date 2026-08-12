// Package engine 定义推理引擎的统一抽象。
//
// 设计目标：不绑定具体算法（检测/分类/分割/生成）或推理框架
// （ONNX Runtime/TensorRT/NCNN/llama.cpp ...）。每一种"模型 + 框架"
// 的组合实现 Engine 接口，注册到 Registry，由上层的并发调度器
// （worker pool / batching / pipeline）统一驱动。
package engine

import (
	"context"
	"errors"
	"image"
	"time"
)

// 调度层使用的哨兵错误。引擎实现不应主动返回它们；由 sched 在队列满/已关闭时产生。
var (
	// ErrBusy 表示有界请求队列已满，调用方应稍后重试（HTTP 映射为 503）。
	ErrBusy = errors.New("engine busy, queue full")
	// ErrClosed 表示调度器已关闭，不再接受新请求（HTTP 映射为 503）。
	ErrClosed = errors.New("engine closed")
)

// Task 标识引擎处理的任务类型。
type Task string

const (
	TaskDetection      Task = "detection"
	TaskClassification Task = "classification"
	TaskSegmentation   Task = "segmentation"
	TaskPose           Task = "pose"
	TaskGeneration     Task = "generation" // LLM / VLM 流式生成
)

// Detection 是检测类任务的统一结果（坐标映射回原图）。
type Detection struct {
	ClassID    int     `json:"class_id"`
	ClassName  string  `json:"class_name"`
	Confidence float32 `json:"confidence"`
	X1         float32 `json:"x1"`
	Y1         float32 `json:"y1"`
	X2         float32 `json:"x2"`
	Y2         float32 `json:"y2"`
}

// Point 是轮廓/掩膜上的一个点，坐标已映射回原图。
type Point struct {
	X float32 `json:"x"`
	Y float32 `json:"y"`
}

// Instance 是实例分割/旋转框等"带形状"任务的一个结果：
// 框 + 类别 + 置信度 + 外轮廓多边形（闭合点列）。
type Instance struct {
	Detection
	Mask [][]Point `json:"mask,omitempty"` // 外轮廓，可能多个（含洞或断开区域）
}

// SegmentationResult 是实例分割任务的结果载体。
type SegmentationResult struct {
	Elapsed   time.Duration
	Instances []Instance
}

func (r *SegmentationResult) Task() Task             { return TaskSegmentation }
func (r *SegmentationResult) Latency() time.Duration { return r.Elapsed }

// Request 是一次推理请求。不同 Task 使用不同字段：
//   - 视觉任务（检测/分类/分割/姿态）：Image
//   - 生成任务：Text（prompt）
//
// Params 携带按请求覆盖的可调参数（如 conf 阈值），由具体引擎解释。
type Request struct {
	Image  image.Image
	Text   string
	Params map[string]float32
}

// Result 是所有引擎结果的最小公共接口。
type Result interface {
	Task() Task
	Latency() time.Duration
}

// DetectionResult 是检测/分割/姿态等视觉框选任务的通用结果载体。
type DetectionResult struct {
	Elapsed    time.Duration
	Detections []Detection
}

func (r *DetectionResult) Task() Task             { return TaskDetection }
func (r *DetectionResult) Latency() time.Duration { return r.Elapsed }

// Engine 是一个已加载模型的推理引擎实例。
type Engine interface {
	// Name 是引擎实例的唯一标识（如 "yolov8n-onnx"、"t1-tensorrt"）。
	Name() string
	// Task 返回该引擎处理的任务类型。
	Task() Task
	// Framework 返回底层推理框架（onnxruntime/tensorrt/ncnn/...）。
	Framework() string
	// Run 执行一次推理。实现需对底层 session/设备访问做并发安全保护。
	Run(ctx context.Context, req *Request) (Result, error)
	// Close 释放模型与设备资源。
	Close() error
}

// Prepared 是一张图的预处理产物：已归一化的 NCHW 缓冲（借自引擎 sync.Pool）
// 与引擎私有的元信息（letterbox 缩放/填充、原图边界、conf 等）。
//
// 缓冲所有权在 BatchEngine 与调度器之间流转：Prepare 借出，RunBatch 消费后
// 由调度器调用 Release 归还。Meta 只应持有标量/值类型，不得引用 Input 缓冲。
type Prepared struct {
	Input []float32 // 一张图，3*H*W，NCHW RGB/255
	Meta  any       // 引擎私有
}

// BatchEngine 是 Engine 的可选能力：支持"预处理 / 合批推理 / 释放"三段式，
// 供 sched 做流水线并行与 dynamic batching。未实现该接口的引擎由调度器
// 退化为 N worker 直接调用 Run（仍有队列与背压）。
type BatchEngine interface {
	Engine
	// MaxBatch 返回一次 RunBatch 能接受的最大图片数。
	// 由模型输入 batch 维决定：固定 1 → 1；动态维 → 引擎配置上限。
	MaxBatch() int
	// Prepare 在设备锁之外完成一张图的 CPU 预处理，返回借出的 Prepared。
	// 返回错误时不得泄漏借出的缓冲。
	Prepare(ctx context.Context, req *Request) (*Prepared, error)
	// RunBatch 在一次底层推理中处理 n 张已预处理的图，返回等长结果切片。
	// 整批成功或整批失败（底层 session.Run 是 all-or-nothing）；n∈[1,MaxBatch]。
	RunBatch(ctx context.Context, batch []*Prepared) ([]Result, error)
	// Release 归还 Prepared 借用的缓冲。对 nil 安全、可重复调用。
	Release(p *Prepared)
}

// Box 是内部使用的轴对齐框（预处理/后处理阶段，坐标为模型空间）。
type Box struct {
	ClassID    int
	Confidence float32
	X1, Y1     float32
	X2, Y2     float32
}
