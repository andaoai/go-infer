// Package promote 把采集池里的图片与伪标签合并到按模型（engine）隔离的训练数据集。
//
// 采集池布局：
//
//	<pool>/<engine>/<YYYYMMDD>/{images,labels}/<stem>.<ext>
//
// promote 后（每个 engine 一个独立 YOLO 数据集，类别体系不同不混）：
//
//	<root>/<engine>/images/train/<date>_<stem>.<ext>
//	<root>/<engine>/labels/train/<date>_<stem>.txt
//	<root>/<engine>/data.yaml
//	<root>/versions/<engine>-<timestamp>.json
//
// 默认移动（Move=true），文件从采集池移走；也可复制。无标签的图片（hard negative，
// 推理无检测）一并迁移，保留为空标签，作为负样本。
package promote

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Options 控制一次 promote。
type Options struct {
	PoolDir     string   // 采集池根，如 dataset/pool
	DatasetRoot string   // 数据集根，如 dataset
	Engine      string   // 只提升该引擎；空则全部
	Date        string   // 只提升该日期 YYYYMMDD；空则全部
	Classes     []string // 类别名，用于生成 data.yaml
	Move        bool     // true 移动，false 复制
	Now         func() time.Time
}

// SourceStat 统计一个来源（engine/date）的提升数量。
type SourceStat struct {
	Engine string `json:"engine"`
	Date   string `json:"date"`
	Files  int    `json:"files"`
}

// Result 是一次 promote 的产出统计。
type Result struct {
	Engine   string       `json:"engine,omitempty"`
	Promoted []SourceStat `json:"promoted"`
	Total    int          `json:"total"`
	DataYAML string       `json:"data_yaml,omitempty"`
	Manifest string       `json:"manifest,omitempty"`
}

// Run 执行合并。
func Run(opts Options) (*Result, error) {
	if opts.PoolDir == "" || opts.DatasetRoot == "" {
		return nil, fmt.Errorf("PoolDir 和 DatasetRoot 均不能为空")
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	engines, err := listEngines(opts.PoolDir, opts.Engine)
	if err != nil {
		return nil, err
	}
	res := &Result{Engine: opts.Engine}
	for _, eng := range engines {
		dates, err := listDates(opts.PoolDir, eng, opts.Date)
		if err != nil {
			return nil, err
		}
		for _, date := range dates {
			n, err := promoteDate(opts, eng, date)
			if err != nil {
				return nil, err
			}
			if n > 0 {
				res.Promoted = append(res.Promoted, SourceStat{Engine: eng, Date: date, Files: n})
				res.Total += n
			}
		}
	}
	if res.Total == 0 {
		return res, nil
	}

	// 单引擎过滤时只为该引擎写 data.yaml；全量时为每个涉及到的引擎各写一份。
	enginesTouched := touchedEngines(res)
	for _, eng := range enginesTouched {
		yamlPath, err := writeDataYAML(opts.DatasetRoot, eng, opts.Classes)
		if err != nil {
			return nil, err
		}
		if opts.Engine != "" {
			res.DataYAML = yamlPath
		}
	}

	manifestPath, err := writeManifest(opts, res)
	if err != nil {
		return nil, err
	}
	res.Manifest = manifestPath
	if opts.Engine != "" && len(enginesTouched) == 1 {
		res.DataYAML = filepath.Join(opts.DatasetRoot, opts.Engine, "data.yaml")
	}
	return res, nil
}

func promoteDate(opts Options, eng, date string) (int, error) {
	srcImgDir := filepath.Join(opts.PoolDir, eng, date, "images")
	srcLblDir := filepath.Join(opts.PoolDir, eng, date, "labels")
	dstImgDir := filepath.Join(opts.DatasetRoot, eng, "images", "train")
	dstLblDir := filepath.Join(opts.DatasetRoot, eng, "labels", "train")
	if err := os.MkdirAll(dstImgDir, 0o755); err != nil {
		return 0, err
	}
	if err := os.MkdirAll(dstLblDir, 0o755); err != nil {
		return 0, err
	}

	entries, err := os.ReadDir(srcImgDir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	count := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			continue
		}
		stem := strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))
		srcImg := filepath.Join(srcImgDir, e.Name())
		// 文件名前缀加日期，避免跨天同名冲突。
		newStem := date + "_" + stem
		dstImg := filepath.Join(dstImgDir, newStem+ext)
		if err := transfer(opts.Move, srcImg, dstImg); err != nil {
			return count, err
		}
		// 标签可能不存在（hard negative）：有则迁移，无则创建空标签文件。
		srcLbl := filepath.Join(srcLblDir, stem+".txt")
		dstLbl := filepath.Join(dstLblDir, newStem+".txt")
		if _, err := os.Stat(srcLbl); err == nil {
			if lerr := transfer(opts.Move, srcLbl, dstLbl); lerr != nil {
				return count, lerr
			}
		} else if os.IsNotExist(err) {
			if werr := os.WriteFile(dstLbl, nil, 0o644); werr != nil {
				return count, werr
			}
		} else {
			return count, err
		}
		count++
	}
	// 移动后清理源目录（含可能残留的非图片/无主标签等）。
	if opts.Move {
		_ = os.RemoveAll(srcImgDir)
		_ = os.RemoveAll(srcLblDir)
		_ = os.Remove(filepath.Join(opts.PoolDir, eng, date))
	}
	return count, nil
}

func transfer(move bool, src, dst string) error {
	if move {
		if err := os.Rename(src, dst); err == nil {
			return nil
		}
		// 跨设备 rename 失败则回退到复制+删除。
		data, err := os.ReadFile(src)
		if err != nil {
			return err
		}
		if err := os.WriteFile(dst, data, 0o644); err != nil {
			return err
		}
		return os.Remove(src)
	}
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

func listEngines(poolDir, only string) ([]string, error) {
	if only != "" {
		if _, err := os.Stat(filepath.Join(poolDir, only)); err != nil {
			if os.IsNotExist(err) {
				return nil, nil
			}
			return nil, err
		}
		return []string{only}, nil
	}
	entries, err := os.ReadDir(poolDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out, nil
}

func listDates(poolDir, eng, only string) ([]string, error) {
	base := filepath.Join(poolDir, eng)
	if only != "" {
		if _, err := os.Stat(filepath.Join(base, only)); err != nil {
			if os.IsNotExist(err) {
				return nil, nil
			}
			return nil, err
		}
		return []string{only}, nil
	}
	entries, err := os.ReadDir(base)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out, nil
}

// writeDataYAML 生成 ultralytics 可用的 data.yaml。
func writeDataYAML(root, eng string, classes []string) (string, error) {
	absRoot, err := filepath.Abs(filepath.Join(root, eng))
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "path: %s\n", absRoot)
	b.WriteString("train: images/train\n")
	b.WriteString("val: images/val\n")
	b.WriteString("test: images/test\n")
	nc := len(classes)
	fmt.Fprintf(&b, "nc: %d\n", nc)
	b.WriteString("names:\n")
	for i, name := range classes {
		fmt.Fprintf(&b, "  %d: %s\n", i, name)
	}
	path := filepath.Join(root, eng, "data.yaml")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	return path, os.WriteFile(path, []byte(b.String()), 0o644)
}

func touchedEngines(res *Result) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range res.Promoted {
		if !seen[s.Engine] {
			seen[s.Engine] = true
			out = append(out, s.Engine)
		}
	}
	sort.Strings(out)
	return out
}

func writeManifest(opts Options, res *Result) (string, error) {
	dir := filepath.Join(opts.DatasetRoot, "versions")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	stamp := opts.Now().Format("20060102-150405")
	name := "promote"
	if opts.Engine != "" {
		name = opts.Engine
	}
	path := filepath.Join(dir, name+"-"+stamp+".json")
	data, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}
	return path, nil
}
