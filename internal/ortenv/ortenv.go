// Package ortenv 集中 ONNX Runtime 动态库的查找与一次性初始化，
// 供 cmd/server 与 cmd/validate 等多个入口复用。
package ortenv

import (
	"fmt"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

// FindLib 在仓库、环境变量和系统目录中查找 libonnxruntime.so。
// explicit 非空时直接使用。找不到返回空字符串。
func FindLib(explicit string) string {
	if explicit != "" {
		return explicit
	}
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
	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		return p
	}
	for _, dir := range []string{"/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"} {
		if matches, _ := filepath.Glob(filepath.Join(dir, "libonnxruntime.so*")); len(matches) > 0 {
			return matches[0]
		}
	}
	return ""
}

// Init 查找并初始化 ONNX Runtime 环境。失败返回错误。
func Init(explicit string) error {
	lib := FindLib(explicit)
	if lib == "" {
		return fmt.Errorf("未找到 libonnxruntime.so，请用 -ort-lib 指定，或执行 `make ort` 下载到 third_party/onnxruntime")
	}
	ort.SetSharedLibraryPath(lib)
	return ort.InitializeEnvironment()
}
