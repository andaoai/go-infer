// Package promote 把采集池里的图片与伪标签合并到按模型（engine）隔离的训练数据集。
//
// 采集池布局（以 YOLO 为例）：
//
//	<pool>/<engine>/<YYYYMMDD>/images/<stem>.<ext>
//	<pool>/<engine>/<YYYYMMDD>/labels/<stem>.txt
//
// promote 后（每个 engine 一个独立数据集，类别体系不同不混）：
//
//	<root>/<engine>/images/train/<date>_<stem>.<ext>
//	<root>/<engine>/labels/train/<date>_<stem>.txt
//	<root>/<engine>/data.yaml            （若 codec 实现 DatasetConfigWriter）
//	<root>/versions/<engine>-<timestamp>.json
//
// 枚举/搬运走注入的 storage.Storage，标签识别与 image→label key 推导走注入的
// format.Codec，从而 promote 逻辑不绑定本地文件系统或 YOLO 目录约定。
// 默认移动（Move=true），以 Get→Put→Remove 实现（跨存储后端安全）；也可复制。
// 无标签图片（hard negative）迁移后保留为空标签。
package promote

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/andaoai/go-infer/internal/format"
	"github.com/andaoai/go-infer/internal/storage"
)

// Options 控制一次 promote。
type Options struct {
	Store       storage.Storage // 字节存储（采集池与数据集共用同一 Store；可用不同前缀）
	Codec       format.Codec    // 标签格式
	PoolRoot    string          // 采集池前缀，如 "pool"
	DatasetRoot string          // 数据集前缀（Store 内逻辑 key），如 "dataset"
	// ConfigRoot 是写进训练配置（YOLO data.yaml 的 path:）的数据集根路径。
	// 本地存储通常传绝对磁盘路径，让 ultralytics 可直接解析；留空则回退用 DatasetRoot。
	ConfigRoot string
	Engine     string   // 只提升该引擎；空则全部
	Date       string   // 只提升该日期 YYYYMMDD；空则全部
	Classes    []string // 类别名，用于生成训练配置
	Move       bool     // true 移动，false 复制
	Now        func() time.Time
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
	if opts.Store == nil || opts.Codec == nil {
		return nil, fmt.Errorf("Store 和 Codec 均不能为空")
	}
	if opts.PoolRoot == "" || opts.DatasetRoot == "" {
		return nil, fmt.Errorf("PoolRoot 和 DatasetRoot 均不能为空")
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	ctx := context.Background()

	engines, err := listEngines(ctx, opts, opts.Engine)
	if err != nil {
		return nil, err
	}
	res := &Result{Engine: opts.Engine}
	for _, eng := range engines {
		dates, err := listDates(ctx, opts, eng, opts.Date)
		if err != nil {
			return nil, err
		}
		for _, date := range dates {
			n, err := promoteDate(ctx, opts, eng, date)
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

	enginesTouched := touchedEngines(res)
	for _, eng := range enginesTouched {
		if w, ok := opts.Codec.(format.DatasetConfigWriter); ok {
			dsRoot := keyJoin(opts.DatasetRoot, eng)
			cfgRoot := dsRoot
			if opts.ConfigRoot != "" {
				cfgRoot = keyJoin(opts.ConfigRoot, eng)
			}
			if err := w.WriteDatasetConfig(ctx, opts.Store, dsRoot, cfgRoot, opts.Classes); err != nil {
				return nil, fmt.Errorf("写数据集配置 (%s): %w", eng, err)
			}
			if opts.Engine != "" && len(enginesTouched) == 1 {
				res.DataYAML = keyJoin(dsRoot, "data.yaml")
			}
		}
	}

	manifestKey, err := writeManifest(ctx, opts, res)
	if err != nil {
		return nil, err
	}
	res.Manifest = manifestKey
	return res, nil
}

func promoteDate(ctx context.Context, opts Options, eng, date string) (int, error) {
	srcPrefix := keyJoin(opts.PoolRoot, eng, date)
	dstImgPrefix := keyJoin(opts.DatasetRoot, eng, "images", "train")

	infos, err := opts.Store.List(ctx, srcPrefix)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, in := range infos {
		if in.IsDir || !opts.Codec.IsImageKey(in.Key) {
			continue
		}
		stem := strings.TrimSuffix(filepath.Base(in.Key), filepath.Ext(in.Key))
		ext := filepath.Ext(in.Key)
		newStem := date + "_" + stem
		dstImg := keyJoin(dstImgPrefix, newStem+ext)
		dstLbl := opts.Codec.LabelKey(dstImg)
		srcLbl := opts.Codec.LabelKey(in.Key)

		if err := transfer(ctx, opts.Store, opts.Move, in.Key, dstImg); err != nil {
			return count, err
		}
		// 标签可能不存在（hard negative）：有则迁移，无则写空标签。
		if _, statErr := opts.Store.Stat(ctx, srcLbl); statErr == nil {
			if lerr := transfer(ctx, opts.Store, opts.Move, srcLbl, dstLbl); lerr != nil {
				return count, lerr
			}
		} else {
			if werr := opts.Store.Put(ctx, dstLbl, bytes.NewReader(nil)); werr != nil {
				return count, werr
			}
		}
		count++
	}
	// 移动后清理该日期的源前缀（含可能残留文件）。
	if opts.Move && count > 0 {
		_ = opts.Store.RemoveAll(ctx, srcPrefix)
	}
	return count, nil
}

// transfer 用 Get→Put（→Remove）搬运，跨存储后端/设备安全。
func transfer(ctx context.Context, st storage.Storage, move bool, src, dst string) error {
	rc, err := st.Get(ctx, src)
	if err != nil {
		return err
	}
	defer rc.Close()
	if err := st.Put(ctx, dst, rc); err != nil {
		return err
	}
	if move {
		return st.Remove(ctx, src)
	}
	return nil
}

// listEngines 列出采集池下的引擎目录（key 的第一段）。
func listEngines(ctx context.Context, opts Options, only string) ([]string, error) {
	if only != "" {
		if _, err := opts.Store.Stat(ctx, keyJoin(opts.PoolRoot, only)); err != nil {
			return nil, nil // 过滤不存在的引擎，静默
		}
		return []string{only}, nil
	}
	return firstSegments(ctx, opts.Store, opts.PoolRoot)
}

// listDates 列出某引擎下的日期目录（key 的第二段）。
func listDates(ctx context.Context, opts Options, eng, only string) ([]string, error) {
	base := keyJoin(opts.PoolRoot, eng)
	if only != "" {
		if _, err := opts.Store.Stat(ctx, keyJoin(base, only)); err != nil {
			return nil, nil
		}
		return []string{only}, nil
	}
	return firstSegments(ctx, opts.Store, base)
}

// firstSegments 返回 prefix 下对象 key 紧邻 prefix 的那段去重有序集合（模拟目录列表）。
func firstSegments(ctx context.Context, st storage.Storage, prefix string) ([]string, error) {
	infos, err := st.List(ctx, prefix)
	if err != nil {
		return nil, err
	}
	prefix = strings.TrimSuffix(filepath.ToSlash(prefix), "/")
	seen := map[string]bool{}
	var out []string
	for _, in := range infos {
		key := strings.TrimPrefix(filepath.ToSlash(in.Key), prefix+"/")
		if key == in.Key || key == "" {
			continue
		}
		seg := strings.SplitN(key, "/", 2)[0]
		if seg != "" && !seen[seg] {
			seen[seg] = true
			out = append(out, seg)
		}
	}
	sort.Strings(out)
	return out, nil
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

func writeManifest(ctx context.Context, opts Options, res *Result) (string, error) {
	stamp := opts.Now().Format("20060102-150405")
	name := "promote"
	if opts.Engine != "" {
		name = opts.Engine
	}
	key := keyJoin(opts.DatasetRoot, "versions", name+"-"+stamp+".json")
	data, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return "", err
	}
	if err := opts.Store.Put(ctx, key, bytes.NewReader(data)); err != nil {
		return "", err
	}
	return key, nil
}

// keyJoin 用正斜杠拼接存储 key。
func keyJoin(parts ...string) string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p != "" {
			out = append(out, filepath.ToSlash(p))
		}
	}
	return strings.Join(out, "/")
}
