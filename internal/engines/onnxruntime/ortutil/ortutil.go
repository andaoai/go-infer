// Package ortutil 存放 onnxruntime 各引擎共用的小工具：batch 维解析等。
package ortutil

import (
	"fmt"
	"log"

	ort "github.com/yalue/onnxruntime_go"
)

// DefaultMaxBatch 是动态 batch 维模型的默认合批上限。
const DefaultMaxBatch = 8

// HardMaxBatch 是合批上限的硬封顶，避免误配导致显存/内存爆炸。
const HardMaxBatch = 32

// ResolveMaxBatch 根据模型输入的 batch 维（dims[0]）与用户配置确定实际合批上限：
//   - 固定 1   → 1（固定 batch 模型只能单图，配置被忽略并告警）
//   - 固定 N>1 → N
//   - 动态(<=0) → configured（<=0 用 DefaultMaxBatch），不超过 HardMaxBatch
//
// dims 为空时按固定 1 处理。
func ResolveMaxBatch(dims ort.Shape, configured int) (int, error) {
	if len(dims) == 0 {
		return 1, nil
	}
	switch b := dims[0]; {
	case b == 1:
		if configured > 1 {
			log.Printf("warn: 模型固定 batch=1，忽略 MaxBatch=%d", configured)
		}
		return 1, nil
	case b > 1:
		return int(b), nil
	case b <= 0:
		n := configured
		if n <= 0 {
			n = DefaultMaxBatch
		}
		if n > HardMaxBatch {
			n = HardMaxBatch
		}
		if n < 1 {
			return 0, fmt.Errorf("invalid max batch %d", n)
		}
		return n, nil
	default:
		return 0, fmt.Errorf("invalid batch dim %d", b)
	}
}
