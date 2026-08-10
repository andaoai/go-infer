package dataset

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/andaoai/go-infer/internal/engine"
)

// WriteBoxLabel 把检测结果写成 YOLO det 标签：每行 "cls cx cy w h"，坐标归一化到 [0,1]。
// 父目录不存在时自动创建。
func WriteBoxLabel(path string, w, h int, dets []engine.Detection) error {
	if w <= 0 || h <= 0 {
		return fmt.Errorf("非法图片尺寸 %dx%d", w, h)
	}
	var b strings.Builder
	for _, d := range dets {
		cx := (d.X1 + d.X2) / 2 / float32(w)
		cy := (d.Y1 + d.Y2) / 2 / float32(h)
		bw := (d.X2 - d.X1) / float32(w)
		bh := (d.Y2 - d.Y1) / float32(h)
		cx, cy, bw, bh = clampNorm(cx), clampNorm(cy), clampNorm(bw), clampNorm(bh)
		if bw <= 0 || bh <= 0 {
			continue
		}
		fmt.Fprintf(&b, "%d %.6f %.6f %.6f %.6f\n", d.ClassID, cx, cy, bw, bh)
	}
	return writeFile(path, []byte(b.String()))
}

// WriteSegLabel 把实例分割结果写成 YOLO seg 标签：每个外轮廓一行
// "cls x1 y1 x2 y2 ..."，坐标归一化到 [0,1]。一个实例有多条轮廓时各占一行
// （同类），读取时会被当作多个多边形对象。
func WriteSegLabel(path string, w, h int, insts []engine.Instance) error {
	if w <= 0 || h <= 0 {
		return fmt.Errorf("非法图片尺寸 %dx%d", w, h)
	}
	var b strings.Builder
	for _, ins := range insts {
		for _, ring := range ins.Mask {
			if len(ring) < 3 {
				continue
			}
			fmt.Fprintf(&b, "%d", ins.ClassID)
			for _, p := range ring {
				x := clampNorm(p.X / float32(w))
				y := clampNorm(p.Y / float32(h))
				fmt.Fprintf(&b, " %.6f %.6f", x, y)
			}
			b.WriteByte('\n')
		}
	}
	return writeFile(path, []byte(b.String()))
}

// SaveImage 把原始图片字节存到 path（保持原编码，不重新编码）。
func SaveImage(path string, data []byte) error {
	return writeFile(path, data)
}

func writeFile(path string, data []byte) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("创建目录 %s: %w", dir, err)
		}
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("写入 %s: %w", path, err)
	}
	return nil
}

func clampNorm(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
