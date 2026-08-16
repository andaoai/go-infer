// Package fsx 提供与具体存储后端无关的文件系统/逻辑路径辅助函数。
package fsx

import (
	"path/filepath"
	"strings"
)

// Join 用正斜杠拼接逻辑存储 key（storage.Storage 的 key 约定）：
// 空段被跳过，每段先经 filepath.ToSlash 规范化，避免 Windows 路径分隔符泄漏。
// capture 与 promote 原先各自维护了一份相同实现，收敛到此处。
func Join(parts ...string) string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p != "" {
			out = append(out, filepath.ToSlash(p))
		}
	}
	return strings.Join(out, "/")
}
