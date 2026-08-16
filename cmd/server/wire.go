package main

import (
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/andaoai/go-infer/internal/api"
	"github.com/andaoai/go-infer/internal/capture"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/seg"
	"github.com/andaoai/go-infer/internal/format"
	"github.com/andaoai/go-infer/internal/storage"
)

// maybeLoadSeg 在模型文件存在时创建分割引擎；文件不存在返回 (nil, nil)。
func maybeLoadSeg(modelPath, name string, imgsz int, classes []string, conf, iou, maskThr float64, maxBatch int) (engine.Engine, error) {
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
		MaxBatch:   maxBatch,
	})
}

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
