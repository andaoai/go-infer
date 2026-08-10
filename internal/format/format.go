// Package format 定义标注格式的编解码抽象。
//
// 一个 Codec 负责"内存中的规范 []data.Object"与"磁盘上的标签字节/目录布局"
// 之间的双向转换，并知道如何在某个数据集根目录下列出图片、推导同名标签路径。
// YOLO（det 5列 + seg 多边形）是第一个实现；未来加 COCO/VOC 只新增实现，
// 不动采集、回流、校验等闭环逻辑。
package format

import (
	"context"

	"github.com/andaoai/go-infer/internal/data"
	"github.com/andaoai/go-infer/internal/storage"
)

// Codec 是一种标注格式。
type Codec interface {
	// Name 返回格式标识，如 "yolo"。
	Name() string
	// LabelExt 返回标签文件扩展名，如 ".txt"。
	LabelExt() string
	// Encode 把对象编码成标签字节（坐标按 w,h 归一化，格式自定义）。
	Encode(objs []data.Object, w, h int) ([]byte, error)
	// Decode 把标签字节解析成对象（坐标还原为像素坐标）。
	// 空/缺失标签（负样本）返回空切片、无错误。
	Decode(b []byte, w, h int) ([]data.Object, error)
	// LabelKey 由图片 key 推导同名标签 key（如 images→labels、换扩展名）。
	LabelKey(imageKey string) string
	// IsImageKey 按扩展名判断 key 是否为图片。
	IsImageKey(key string) bool
	// ListImages 枚举 root/<某个 split 目录> 下的图片及其标签 key。
	// split 对 YOLO 是 images/<split> 的 <split>；其它格式可忽略或自定义。
	ListImages(ctx context.Context, st storage.Storage, root, split string) ([]data.ImageRef, error)
}

// DatasetConfigWriter 是 Codec 的可选能力：回流后写训练配置文件
// （YOLO 写 data.yaml；COCO 等可写自己的配置或不实现）。
//
// datasetRoot 是配置文件在 Store 内写入位置的 key 前缀；configRoot 是写进
// 配置内容里的数据根路径（本地存储通常是绝对磁盘路径，供训练框架直接解析）；
// 二者在纯逻辑存储里可以相同。
type DatasetConfigWriter interface {
	WriteDatasetConfig(ctx context.Context, st storage.Storage, datasetRoot, configRoot string, classes []string) error
}
