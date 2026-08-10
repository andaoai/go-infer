// Package yolo 实现 YOLO 标注格式的 Codec：检测框 cls cx cy w h，
// 分割多边形 cls x1 y1 x2 y2 ...，坐标归一化到 [0,1]。
//
// 目录布局为 ultralytics 约定：<root>/images/<split>/*.{jpg,png} 与
// 同名标签 <root>/labels/<split>/*.txt。
package yolo

import (
	"bufio"
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/andaoai/go-infer/internal/data"
	"github.com/andaoai/go-infer/internal/storage"
)

// Codec 是 YOLO 格式实现，无状态，可全局复用。
type Codec struct{}

// New 返回 YOLO Codec。
func New() *Codec { return &Codec{} }

func (c *Codec) Name() string     { return "yolo" }
func (c *Codec) LabelExt() string { return ".txt" }

// Encode 把对象编码为 YOLO 标签字节：
//   - 无 Rings → "cls cx cy w h"
//   - 有 Rings → 每个外轮廓一行 "cls x y x y ..."
//
// 多轮廓实例写成多行（同类），读取时会还原为多个对象（环分组信息在往返中
// 丢失，与 ultralytics 逐多边形约定一致）。
func (c *Codec) Encode(objs []data.Object, w, h int) ([]byte, error) {
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("非法图片尺寸 %dx%d", w, h)
	}
	var b strings.Builder
	for _, o := range objs {
		if len(o.Rings) > 0 {
			for _, ring := range o.Rings {
				if len(ring) < 3 {
					continue
				}
				fmt.Fprintf(&b, "%d", o.ClassID)
				for _, p := range ring {
					fmt.Fprintf(&b, " %.6f %.6f", clampNorm(p.X/float32(w)), clampNorm(p.Y/float32(h)))
				}
				b.WriteByte('\n')
			}
			continue
		}
		cx := (o.BBox.X1 + o.BBox.X2) / 2 / float32(w)
		cy := (o.BBox.Y1 + o.BBox.Y2) / 2 / float32(h)
		bw := (o.BBox.X2 - o.BBox.X1) / float32(w)
		bh := (o.BBox.Y2 - o.BBox.Y1) / float32(h)
		cx, cy, bw, bh = clampNorm(cx), clampNorm(cy), clampNorm(bw), clampNorm(bh)
		if bw <= 0 || bh <= 0 {
			continue
		}
		fmt.Fprintf(&b, "%d %.6f %.6f %.6f %.6f\n", o.ClassID, cx, cy, bw, bh)
	}
	return []byte(b.String()), nil
}

// Decode 解析 YOLO 标签：每行 5 个数 → 检测框，≥6 个偶数 → 分割多边形。
// 分割对象的 BBox 取点列 min/max。空字节返回 nil（负样本）。
func (c *Codec) Decode(b []byte, w, h int) ([]data.Object, error) {
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("非法图片尺寸 %dx%d", w, h)
	}
	if len(b) == 0 {
		return nil, nil
	}
	var out []data.Object
	sc := bufio.NewScanner(strings.NewReader(string(b)))
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
			cx, cy, bw, bh := coords[0], coords[1], coords[2], coords[3]
			x1 := clampF(float32((cx-bw/2)*float64(w)), 0, float32(w))
			y1 := clampF(float32((cy-bh/2)*float64(h)), 0, float32(h))
			x2 := clampF(float32((cx+bw/2)*float64(w)), 0, float32(w))
			y2 := clampF(float32((cy+bh/2)*float64(h)), 0, float32(h))
			out = append(out, data.Object{
				ClassID: cls,
				BBox:    data.BBox{X1: x1, Y1: y1, X2: x2, Y2: y2},
			})
		case len(coords) >= 6 && len(coords)%2 == 0:
			pts := make([]data.Point, 0, len(coords)/2)
			var minX, minY, maxX, maxY float32
			for i := 0; i+1 < len(coords); i += 2 {
				x := clampF(float32(coords[i]*float64(w)), 0, float32(w))
				y := clampF(float32(coords[i+1]*float64(h)), 0, float32(h))
				pts = append(pts, data.Point{X: x, Y: y})
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
			out = append(out, data.Object{
				ClassID: cls,
				BBox:    data.BBox{X1: minX, Y1: minY, X2: maxX, Y2: maxY},
				Rings:   [][]data.Point{pts},
			})
		default:
			return nil, fmt.Errorf("第 %d 行坐标个数异常: %d（检测需4，分割需偶数且≥6）", lineNo, len(coords))
		}
	}
	return out, sc.Err()
}

// LabelKey 把路径里最后一个 "images" 段替换为 "labels"，并把扩展名换成 .txt。
// 例如 root/images/train/a.jpg → root/labels/train/a.txt；
// pool/<engine>/<date>/images/a.jpg → pool/<engine>/<date>/labels/a.txt。
// 若路径里没有 images 段，仅替换扩展名。
func (c *Codec) LabelKey(imageKey string) string {
	key := filepath.ToSlash(imageKey)
	stem := strings.TrimSuffix(key, filepath.Ext(key))
	segs := strings.Split(stem, "/")
	for i := len(segs) - 2; i >= 0; i-- { // 文件名段不算，倒数第二往上找
		if segs[i] == "images" {
			segs[i] = "labels"
			return strings.Join(segs, "/") + ".txt"
		}
	}
	return stem + ".txt"
}

// IsImageKey 按扩展名判断是否为支持的图片。
func (c *Codec) IsImageKey(key string) bool {
	ext := strings.ToLower(filepath.Ext(key))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
}

// ListImages 枚举 <root>/images/<split>/ 下的图片，推导同名标签 key。
// 标签缺失不报错（负样本），LabelKey 仍返回。
func (c *Codec) ListImages(ctx context.Context, st storage.Storage, root, split string) ([]data.ImageRef, error) {
	prefix := filepath.ToSlash(filepath.Join(root, "images", split))
	infos, err := st.List(ctx, prefix)
	if err != nil {
		return nil, fmt.Errorf("枚举图片目录 %s: %w", prefix, err)
	}
	var refs []data.ImageRef
	for _, in := range infos {
		if in.IsDir || !c.IsImageKey(in.Key) {
			continue
		}
		refs = append(refs, data.ImageRef{ImageKey: in.Key, LabelKey: c.LabelKey(in.Key)})
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].ImageKey < refs[j].ImageKey })
	return refs, nil
}

// WriteDatasetConfig 写 ultralytics data.yaml（实现 format.DatasetConfigWriter）。
// 配置写入 <datasetRoot>/data.yaml，内容里的 path 记为 configRoot（通常是磁盘绝对路径）。
func (c *Codec) WriteDatasetConfig(ctx context.Context, st storage.Storage, datasetRoot, configRoot string, classes []string) error {
	var b strings.Builder
	fmt.Fprintf(&b, "path: %s\n", configRoot)
	b.WriteString("train: images/train\n")
	b.WriteString("val: images/val\n")
	b.WriteString("test: images/test\n")
	fmt.Fprintf(&b, "nc: %d\n", len(classes))
	b.WriteString("names:\n")
	for i, name := range classes {
		fmt.Fprintf(&b, "  %d: %s\n", i, name)
	}
	key := filepath.ToSlash(filepath.Join(datasetRoot, "data.yaml"))
	return st.Put(ctx, key, strings.NewReader(b.String()))
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

func clampNorm(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
