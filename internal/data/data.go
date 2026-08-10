// Package data 定义引擎无关的规范数据模型：一次标注/预测在内存中的统一表示。
//
// 各推理后端返回的 engine.Result、各标注格式（YOLO/COCO/...）的标签文件、
// 各指标实现，都围绕 Point/BBox/Object/Sample 转换，避免同一份"框+多边形"
// 在多个包里各有一套类型。本包不依赖任何具体存储或格式。
package data

import "image"

// Point 是轮廓/掩膜上的一个点，像素坐标（float 保精度，不提前取整）。
type Point struct {
	X float32 `json:"x"`
	Y float32 `json:"y"`
}

// BBox 是轴对齐框，像素坐标。
type BBox struct {
	X1 float32 `json:"x1"`
	Y1 float32 `json:"y1"`
	X2 float32 `json:"x2"`
	Y2 float32 `json:"y2"`
}

// Object 是一个标注实例或预测实例：类别 + 置信度 + 框 + 可选外轮廓。
//
// Confidence 对 GT 为 0；BBox 对检测框任务直接使用，对分割任务取多边形外接框；
// Rings 为分割外轮廓（多条取并集），检测任务为空。
type Object struct {
	ClassID    int       `json:"class_id"`
	Confidence float32   `json:"confidence,omitempty"`
	BBox       BBox      `json:"bbox"`
	Rings      [][]Point `json:"rings,omitempty"`
}

// Sample 是一张图及其全部对象。
type Sample struct {
	Key     string
	Image   image.Image
	W, H    int
	Objects []Object
}

// ImageRef 是一张图片的存储定位：图片 key 与同名标签 key。
// LabelKey 指向的标签可能不存在（负样本）。
type ImageRef struct {
	ImageKey string
	LabelKey string
}
