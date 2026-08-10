// Package appcfg 提供多个命令入口共用的配置加载逻辑（类别名等）。
package appcfg

import (
	"fmt"
	"os"
	"strings"
)

// LoadClasses 解析类别名：优先逗号分隔字符串，其次一行一个的文件，
// 都没给则用 n 生成 class_0..class_{n-1}。
func LoadClasses(classes, classesFile string, n int) ([]string, error) {
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
