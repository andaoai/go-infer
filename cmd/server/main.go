// Command server 启动 YOLO ONNX 推理 HTTP 服务。
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/andaoai/yolo-onnx-go/internal/api"
	"github.com/andaoai/yolo-onnx-go/internal/detector"
	"github.com/yalue/onnxruntime_go"
)

func main() {
	var (
		modelPath   = flag.String("model", "models/best.onnx", "ONNX 模型路径")
		classes     = flag.String("classes", "", "类别名，逗号分隔（如 person,car）；留空则按 -classes-file 或 -nc")
		classesFile = flag.String("classes-file", "", "类别名文件路径，一行一个类别名（如 models/coco.names）")
		numClass    = flag.Int("nc", 0, "类别数；classes 和 classes-file 均未提供时据此生成默认名")
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
	defer onnxruntime_go.DestroyEnvironment()

	classList, err := loadClasses(*classes, *classesFile, *numClass)
	if err != nil {
		log.Fatalf("读取类别: %v", err)
	}
	cfg := detector.Config{
		ModelPath:  *modelPath,
		InputW:     *imgsz,
		InputH:     *imgsz,
		Classes:    classList,
		ConfThresh: float32(*conf),
		IoUThresh:  float32(*iou),
	}
	det, err := detector.New(cfg)
	if err != nil {
		log.Fatalf("加载模型: %v", err)
	}
	defer det.Close()

	srv := api.NewServer(det)
	log.Printf("YOLO ONNX 服务启动于 %s (model=%s imgsz=%d classes=%d)",
		*addr, *modelPath, *imgsz, len(cfg.Classes))
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
	onnxruntime_go.SetSharedLibraryPath(lib)
	return onnxruntime_go.InitializeEnvironment()
}

// findOrtLib 在常见位置查找 libonnxruntime.so。
func findOrtLib() string {
	// 1. 项目内 third_party（make ort 下载位置）
	if matches, _ := filepath.Glob("third_party/onnxruntime/lib/libonnxruntime.so*"); len(matches) > 0 {
		return matches[0]
	}
	// 2. 环境变量
	if p := os.Getenv("LD_LIBRARY_PATH"); p != "" {
		for _, dir := range filepath.SplitList(p) {
			if matches, _ := filepath.Glob(filepath.Join(dir, "libonnxruntime.so*")); len(matches) > 0 {
				return matches[0]
			}
		}
	}
	// 3. 系统标准路径
	for _, dir := range []string{"/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"} {
		if matches, _ := filepath.Glob(filepath.Join(dir, "libonnxruntime.so*")); len(matches) > 0 {
			return matches[0]
		}
	}
	return ""
}

// loadClasses 按优先级解析类别名：-classes 逗号串 > -classes-file 文件 > -nc 默认名。
func loadClasses(classes, classesFile string, n int) ([]string, error) {
	if strings.TrimSpace(classes) != "" {
		parts := strings.Split(classes, ",")
		out := make([]string, 0, len(parts))
		for _, p := range parts {
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
