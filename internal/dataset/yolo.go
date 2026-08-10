// Package dataset 读取 YOLO 格式的视觉数据集（图片 + 文本标签）。
//
// 支持两种标签格式：
//   - 检测：每行 "cls cx cy w h"，坐标归一化到 [0,1]（cxcywh）
//   - 分割：每行 "cls x1 y1 x2 y2 ... xn yn"，归一化多边形点列
//
// 图片按 images/<split>/*.{jpg,jpeg,png} 组织，标签同名放在 labels/<split>/*.txt。
package dataset

import (
	"bufio"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	_ "image/jpeg"
	_ "image/png"
)

// GT 是一个标注实例：轴对齐框 + 可选外轮廓多边形（像素坐标）。
type GT struct {
	ClassID        int
	X1, Y1, X2, Y2 float32
	// Polygon 为分割标签的外轮廓点列（像素坐标）；检测标签为空。
	Polygon []image.Point
}

// Sample 是数据集中的一张图及其标注。
type Sample struct {
	Path    string
	Image   image.Image
	W, H    int
	Objects []GT
}

// Load 遍历 root/images/<split> 下的图片，读取同名标签文件。
// 缺失标签文件按"无标注图"处理（负样本）。
func Load(root, split string) ([]Sample, error) {
	imgDir := filepath.Join(root, "images", split)
	entries, err := os.ReadDir(imgDir)
	if err != nil {
		return nil, fmt.Errorf("读取图片目录 %s: %w", imgDir, err)
	}

	var samples []Sample
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			continue
		}
		imgPath := filepath.Join(imgDir, e.Name())
		img, err := loadImage(imgPath)
		if err != nil {
			return nil, err
		}
		b := img.Bounds()
		w, h := b.Dx(), b.Dy()

		labelPath := filepath.Join(root, "labels", split,
			strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))+".txt")
		objs, err := loadLabels(labelPath, w, h)
		if err != nil {
			return nil, fmt.Errorf("解析标签 %s: %w", labelPath, err)
		}
		samples = append(samples, Sample{
			Path: imgPath, Image: img, W: w, H: h, Objects: objs,
		})
	}
	// 稳定排序，保证多次运行结果可复现。
	sort.Slice(samples, func(i, j int) bool { return samples[i].Path < samples[j].Path })
	return samples, nil
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("打开图片 %s: %w", path, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("解码图片 %s: %w", path, err)
	}
	return img, nil
}

// loadLabels 解析 YOLO 标签文件。缺失文件返回空切片（负样本）。
func loadLabels(path string, w, h int) ([]GT, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var out []GT
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		nums := make([]float64, 0, len(fields))
		for _, s := range fields {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return nil, fmt.Errorf("第 %d 行字段 %q 不是数字", lineNo, s)
			}
			nums = append(nums, v)
		}
		if len(nums) < 5 {
			return nil, fmt.Errorf("第 %d 行字段不足（需要 cls+4 框 或 cls+多边形）: %d", lineNo, len(nums))
		}
		cls := int(nums[0])
		coords := nums[1:]

		switch {
		case len(coords) == 4:
			// 检测框 cx cy w h（归一化）
			cx, cy, bw, bh := coords[0], coords[1], coords[2], coords[3]
			x1 := (cx - bw/2) * float64(w)
			y1 := (cy - bh/2) * float64(h)
			x2 := (cx + bw/2) * float64(w)
			y2 := (cy + bh/2) * float64(h)
			out = append(out, GT{
				ClassID: cls,
				X1:      clampF(float32(x1), 0, float32(w)), Y1: clampF(float32(y1), 0, float32(h)),
				X2: clampF(float32(x2), 0, float32(w)), Y2: clampF(float32(y2), 0, float32(h)),
			})
		case len(coords) >= 6 && len(coords)%2 == 0:
			// 分割多边形：x1 y1 x2 y2 ...（归一化）
			pts := make([]image.Point, 0, len(coords)/2)
			var minX, minY, maxX, maxY float32
			for i := 0; i+1 < len(coords); i += 2 {
				x := clampF(float32(coords[i]*float64(w)), 0, float32(w))
				y := clampF(float32(coords[i+1]*float64(h)), 0, float32(h))
				pts = append(pts, image.Pt(int(x), int(y)))
				if i == 0 || x < minX {
					minX = x
				}
				if i == 0 || y < minY {
					minY = y
				}
				if x > maxX {
					maxX = x
				}
				if y > maxY {
					maxY = y
				}
			}
			out = append(out, GT{
				ClassID: cls,
				X1:      minX, Y1: minY, X2: maxX, Y2: maxY,
				Polygon: pts,
			})
		default:
			return nil, fmt.Errorf("第 %d 行坐标个数异常: %d（检测需4，分割需偶数且≥6）", lineNo, len(coords))
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func clampF(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
