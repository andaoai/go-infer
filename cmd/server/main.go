// Command go-infer 启动推理 HTTP 服务。
//
// 它是引擎无关的：所有推理逻辑由实现了 engine.Engine 的具体引擎提供，
// 本程序只负责加载 ONNX Runtime、按配置构造并注册引擎、启动 HTTP。
// 当前内置一个 ONNX Runtime YOLO 检测引擎，后续可在此注册更多后端/算法。
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/andaoai/go-infer/internal/api"
	"github.com/andaoai/go-infer/internal/engine"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/detect"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	var (
		name        = flag.String("name", "yolov8n", "引擎实例名（用于 ?engine= 选择）")
		modelPath   = flag.String("model", "models/yolov8n.onnx", "ONNX 模型路径")
		classes     = flag.String("classes", "", "类别名，逗号分隔；留空则看 -classes-file")
		classesFile = flag.String("classes-file", "models/coco.names", "类别名文件，一行一个；空字符串表示不读")
		numClass    = flag.Int("nc", 0, "类别数（前两者均未提供时用）")
		imgsz       = flag.Int("imgsz", 640, "模型输入尺寸（正方形）")
		conf        = flag.Float64("conf", 0.25, "默认置信度阈值")
		iou         = flag.Float64("iou", 0.45, "NMS IoU 阈值")
		addr        = flag.String("addr", ":8080", "监听地址")
		ortLib      = flag.String("ort-lib", "", "libonnxruntime.so 路径；留空则自动查找")
	)
	flag.Parse()

	if err := initONNXRuntime(*ortLib); err != nil {
		log.Fatalf("初始化 ONNX Runtime: %v", err)
	}
	defer ort.DestroyEnvironment()

	classList, err := loadClasses(*classes, *classesFile, *numClass)
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
	// 在此 srv.Register(...) 更多引擎（TensorRT/NCNN/llama.cpp ...）。

	log.Printf("go-infer 服务启动于 %s | 引擎=%s task=%s framework=%s classes=%d",
		*addr, eng.Name(), eng.Task(), eng.Framework(), len(classList))
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}

// initONNXRuntime 设置并初始化 ONNX Runtime 动态库。
func initONNXRuntime(explicit string) error {
	lib := explicit
	if lib == "" {
		lib = findOrtLib()
	}
	if lib == "" {
		return fmt.Errorf("未找到 libonnxruntime.so，请用 -ort-lib 指定，或执行 `make ort` 下载到 third_party/onnxruntime")
	}
	log.Printf("使用 ONNX Runtime: %s", lib)
	ort.SetSharedLibraryPath(lib)
	return ort.InitializeEnvironment()
}

// findOrtLib 在常见位置查找 libonnxruntime.so。
func findOrtLib() string {
	if matches, _ := filepath.Glob("third_party/onnxruntime/lib/libonnxruntime.so*"); len(matches) > 0 {
		return matches[0]
	}
	if p := os.Getenv("LD_LIBRARY_PATH"); p != "" {
		for _, dir := range filepath.SplitList(p) {
			if matches, _ := filepath.Glob(filepath.Join(dir, "libonnxruntime.so*")); len(matches) > 0 {
				return matches[0]
			}
		}
	}
	for _, dir := range []string{"/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"} {
		if matches, _ := filepath.Glob(filepath.Join(dir, "libonnxruntime.so*")); len(matches) > 0 {
			return matches[0]
		}
	}
	return ""
}

// loadClasses 按优先级解析类别名：-classes > -classes-file > -nc 默认名。
func loadClasses(classes, classesFile string, n int) ([]string, error) {
	if strings.TrimSpace(classes) != "" {
		var out []string
		for _, p := range strings.Split(classes, ",") {
			if name := strings.TrimSpace(p); name != "" {
				out = append(out, name)
			}
		}
		return out, nil
	}
	if classesFile != "" {
		data, err := os.ReadFile(classesFile)
		if err != nil {
			return nil, fmt.Errorf("读取类别文件 %s: %w", classesFile, err)
		}
		var out []string
		for _, line := range strings.Split(string(data), "\n") {
			if name := strings.TrimSpace(line); name != "" {
				out = append(out, name)
			}
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("类别文件 %s 为空", classesFile)
		}
		return out, nil
	}
	if n <= 0 {
		n = 1
	}
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("class_%d", i)
	}
	return out, nil
}

// 保留 engine 引用以便未来在此文件中引用接口类型。
var _ engine.Engine = (*detect.Engine)(nil)
