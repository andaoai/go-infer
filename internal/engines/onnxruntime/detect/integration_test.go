package detect

import (
	"context"
	"image"
	"os"
	"path/filepath"
	"sync"
	"testing"

	_ "image/jpeg"

	"github.com/andaoai/go-infer/internal/engine"
	ort "github.com/yalue/onnxruntime_go"
)

var ortInitOnce sync.Once

// findORTLib 在仓库常见位置查找 libonnxruntime.so，供集成测试初始化环境。
func findORTLib(t *testing.T) string {
	t.Helper()
	candidates := []string{
		"../../../../third_party/onnxruntime/lib/libonnxruntime.so",
		"third_party/onnxruntime/lib/libonnxruntime.so",
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			abs, _ := filepath.Abs(c)
			return abs
		}
	}
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		return p
	}
	t.Skip("未找到 libonnxruntime.so，跳过需 C 库的集成测试（执行 make ort 或设 ORT_LIB_PATH）")
	return ""
}

func initORT(t *testing.T) {
	t.Helper()
	lib := findORTLib(t)
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
	if _, err := os.Stat(path); err != nil {
		t.Skipf("缺少测试图 %s，跳过", path)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
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
	path := "../../../../models/coco.names"
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("缺少 %s: %v", path, err)
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

// TestIntegrationYOLOv8 用真实 yolov8n.onnx 对 bus.jpg 做端到端检测。
func TestIntegrationYOLOv8(t *testing.T) {
	initORT(t)
	modelPath := "../../../../models/yolov8n.onnx"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("缺少模型 %s，跳过集成测试", modelPath)
	}

	eng, err := New(Config{
		Name:      "yolov8n-test",
		ModelPath: modelPath,
		InputW:    640,
		InputH:    640,
		Classes:   loadCOCONames(t),
	})
	if err != nil {
		t.Fatalf("创建引擎失败: %v", err)
	}
	defer eng.Close()

	if eng.Task() != "detection" || eng.Framework() != "onnxruntime" {
		t.Fatalf("元信息错误: task=%s framework=%s", eng.Task(), eng.Framework())
	}

	res, err := eng.Run(context.Background(), &engine.Request{Image: loadBus(t)})
	if err != nil {
		t.Fatalf("推理失败: %v", err)
	}
	dr, ok := res.(*engine.DetectionResult)
	if !ok {
		t.Fatalf("结果类型错误: %T", res)
	}
	if dr.Latency() <= 0 {
		t.Fatalf("耗时应 > 0, got %v", dr.Latency())
	}

	// bus.jpg 预期：3 个 person + 1 个 bus（与官方结果一致）。
	var persons, buses int
	for _, d := range dr.Detections {
		if d.ClassName == "person" {
			persons++
		}
		if d.ClassName == "bus" {
			buses++
		}
		// 坐标应落在原图 810x1080 内。
		if d.X1 < 0 || d.Y1 < 0 || d.X2 > 811 || d.Y2 > 1081 {
			t.Errorf("检测框越界: %+v", d)
		}
	}
	if persons < 2 {
		t.Errorf("预期至少 2 个 person，got %d (detections=%d)", persons, len(dr.Detections))
	}
	if buses < 1 {
		t.Errorf("预期至少 1 个 bus，got %d", buses)
	}
	t.Logf("检出 %d 个目标: %d person, %d bus, 耗时 %v",
		len(dr.Detections), persons, buses, dr.Latency())
}

// TestConfOverride 验证按请求覆盖 conf 阈值。
func TestConfOverride(t *testing.T) {
	initORT(t)
	modelPath := "../../../../models/yolov8n.onnx"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("缺少模型 %s", modelPath)
	}
	eng, err := New(Config{
		Name: "yolov8n-test", ModelPath: modelPath, InputW: 640, InputH: 640,
		Classes:    loadCOCONames(t),
		ConfThresh: 0.25,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	img := loadBus(t)
	low, _ := eng.Run(context.Background(), &engine.Request{Image: img})
	high, err := eng.Run(context.Background(), &engine.Request{
		Image:  img,
		Params: map[string]float32{"conf": 0.95},
	})
	if err != nil {
		t.Fatal(err)
	}
	lo := len(low.(*engine.DetectionResult).Detections)
	hi := len(high.(*engine.DetectionResult).Detections)
	if hi >= lo {
		t.Errorf("高阈值(0.95)应检出更少目标: low=%d high=%d", lo, hi)
	}
}
