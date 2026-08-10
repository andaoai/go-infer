package seg

import (
	"context"
	"image"
	"os"
	"sync"
	"testing"

	_ "image/jpeg"

	"github.com/andaoai/go-infer/internal/engine"
	ort "github.com/yalue/onnxruntime_go"
)

var ortInitOnce sync.Once

func initORT(t *testing.T) {
	t.Helper()
	candidates := []string{
		"../../../../third_party/onnxruntime/lib/libonnxruntime.so",
		"third_party/onnxruntime/lib/libonnxruntime.so",
	}
	var lib string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			lib = c
			break
		}
	}
	if lib == "" {
		lib = os.Getenv("ORT_LIB_PATH")
	}
	if lib == "" {
		t.Skip("未找到 libonnxruntime.so，跳过（执行 make ort 或设 ORT_LIB_PATH）")
	}
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(lib)
		if err := ort.InitializeEnvironment(); err != nil {
			t.Fatalf("初始化 ONNX Runtime: %v", err)
		}
	})
}

func loadBus(t *testing.T) image.Image {
	t.Helper()
	path := "../../../../examples/bus.jpg"
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("缺少测试图 %s: %v", path, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatalf("解码 bus.jpg: %v", err)
	}
	return img
}

func loadCOCONames(t *testing.T) []string {
	t.Helper()
	data, err := os.ReadFile("../../../../models/coco.names")
	if err != nil {
		t.Skipf("缺少 coco.names: %v", err)
	}
	var names []string
	start := 0
	for i := 0; i <= len(data); i++ {
		if i == len(data) || data[i] == '\n' {
			if i > start {
				names = append(names, string(data[start:i]))
			}
			start = i + 1
		}
	}
	return names
}

// TestIntegrationYOLOv8Seg 用真实 yolov8n-seg.onnx 对 bus.jpg 端到端分割。
// 模型未下载时自动 skip（`make seg-model` 后即生效）。
func TestIntegrationYOLOv8Seg(t *testing.T) {
	initORT(t)
	modelPath := "../../../../models/yolov8n-seg.onnx"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("缺少分割模型 %s，执行 `make seg-model` 后运行（%v）", modelPath, err)
	}

	eng, err := New(Config{
		Name:      "yolov8n-seg-test",
		ModelPath: modelPath,
		InputW:    640,
		InputH:    640,
		Classes:   loadCOCONames(t),
	})
	if err != nil {
		t.Fatalf("创建分割引擎失败: %v", err)
	}
	defer eng.Close()

	if eng.Task() != engine.TaskSegmentation {
		t.Fatalf("task 错误: got %s want %s", eng.Task(), engine.TaskSegmentation)
	}

	res, err := eng.Run(context.Background(), &engine.Request{Image: loadBus(t)})
	if err != nil {
		t.Fatalf("推理失败: %v", err)
	}
	sr, ok := res.(*engine.SegmentationResult)
	if !ok {
		t.Fatalf("结果类型错误: %T", res)
	}
	if sr.Latency() <= 0 {
		t.Fatalf("耗时应 > 0, got %v", sr.Latency())
	}
	if len(sr.Instances) == 0 {
		t.Fatal("预期至少 1 个实例，got 0")
	}

	// bus.jpg 应分割出 person 和 bus，且每个实例都有非空掩膜多边形。
	var persons, buses int
	for i, ins := range sr.Instances {
		if ins.ClassName == "person" {
			persons++
		}
		if ins.ClassName == "bus" {
			buses++
		}
		if len(ins.Mask) == 0 {
			t.Errorf("实例 %d (%s) 无掩膜多边形", i, ins.ClassName)
			continue
		}
		pts := 0
		for _, poly := range ins.Mask {
			if len(poly) < 3 {
				t.Errorf("实例 %d 多边形点数 < 3: %d", i, len(poly))
			}
			pts += len(poly)
			for _, p := range poly {
				// 坐标应落在原图 810x1080 内。
				if p.X < 0 || p.Y < 0 || p.X > 811 || p.Y > 1081 {
					t.Errorf("实例 %d 轮廓点越界: (%.1f,%.1f)", i, p.X, p.Y)
				}
			}
		}
		t.Logf("  实例 %d: %s conf=%.2f 框=[%.0f,%.0f,%.0f,%.0f] 多边形点=%d",
			i, ins.ClassName, ins.Confidence, ins.X1, ins.Y1, ins.X2, ins.Y2, pts)
	}
	if persons < 2 {
		t.Errorf("预期至少 2 个 person，got %d (instances=%d)", persons, len(sr.Instances))
	}
	t.Logf("分割 %d 个实例: %d person, %d bus, 耗时 %v",
		len(sr.Instances), persons, buses, sr.Latency())
}
