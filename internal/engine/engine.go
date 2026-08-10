// Package engine 定义推理引擎的统一抽象。
//
// 设计目标：不绑定具体算法（检测/分类/分割/生成）或推理框架
// （ONNX Runtime/TensorRT/NCNN/llama.cpp ...）。每一种"模型 + 框架"
// 的组合实现 Engine 接口，注册到 Registry，由上层的并发调度器
// （worker pool / batching / pipeline）统一驱动。
package engine

import (
	"context"
	"image"
	"time"
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

// Box 是内部使用的轴对齐框（预处理/后处理阶段，坐标为模型空间）。
type Box struct {
	ClassID    int
	Confidence float32
	X1, Y1     float32
	X2, Y2     float32
}
