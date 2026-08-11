// 采集池浏览器：用注入的 storage.Storage + format.Codec 枚举/读取/删除采集样本，
// 供前端看板查看与管理。仅暴露采集池前缀下的对象，不越界访问存储其他位置。
package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"

	"github.com/andaoai/go-infer/internal/api"
	"github.com/andaoai/go-infer/internal/format"
	"github.com/andaoai/go-infer/internal/storage"
)

type captureBrowser struct {
	st       storage.Storage
	codec    format.Codec
	poolRoot string
}

// groups 返回 引擎 → 日期 → 文件 的分组结构。
func (b *captureBrowser) Groups(ctx context.Context) ([]api.CaptureGroup, error) {
	prefix := strings.Trim(filepath.ToSlash(b.poolRoot), "/")
	infos, err := b.st.List(ctx, prefix)
	if err != nil {
		return nil, err
	}

	type fileEntry struct {
		info storage.Info
		rel  string // 相对 poolRoot 的路径
	}
	byEngine := map[string]map[string][]fileEntry{}

	for _, in := range infos {
		if in.IsDir || !b.codec.IsImageKey(in.Key) {
			continue
		}
		rel := in.Key
		if prefix != "" {
			rel = strings.TrimPrefix(rel, prefix+"/")
		}
		parts := strings.Split(rel, "/")
		if len(parts) < 4 { // <engine>/<date>/images/<file>
			continue
		}
		engine, date := parts[0], parts[1]
		if byEngine[engine] == nil {
			byEngine[engine] = map[string][]fileEntry{}
		}
		byEngine[engine][date] = append(byEngine[engine][date], fileEntry{info: in, rel: rel})
	}

	engines := make([]string, 0, len(byEngine))
	for e := range byEngine {
		engines = append(engines, e)
	}
	sort.Strings(engines)

	out := make([]api.CaptureGroup, 0, len(engines))
	for _, eng := range engines {
		dateMap := byEngine[eng]
		dates := make([]string, 0, len(dateMap))
		for d := range dateMap {
			dates = append(dates, d)
		}
		sort.Sort(sort.Reverse(sort.StringSlice(dates))) // 新日期在前

		g := api.CaptureGroup{Engine: eng}
		for _, date := range dates {
			files := dateMap[date]
			sort.Slice(files, func(i, j int) bool {
				return files[i].info.ModTime.After(files[j].info.ModTime)
			})
			cd := api.CaptureDate{Date: date}
			for _, fe := range files {
				in := fe.info
				labelKey := b.codec.LabelKey(in.Key)
				hasLabel := false
				if _, err := b.st.Stat(ctx, labelKey); err == nil {
					hasLabel = true
				}
				cd.Files = append(cd.Files, api.CaptureFile{
					Key:      in.Key,
					LabelKey: labelKey,
					Name:     filepath.Base(in.Key),
					Size:     in.Size,
					HasLabel: hasLabel,
					When:     in.ModTime.Unix(),
				})
			}
			cd.Count = len(cd.Files)
			g.Count += cd.Count
			g.Dates = append(g.Dates, cd)
		}
		out = append(out, g)
	}
	return out, nil
}

// inPool 校验 key 落在采集池前缀下，防止越界读取存储其他位置。
func (b *captureBrowser) inPool(key string) bool {
	prefix := strings.Trim(filepath.ToSlash(b.poolRoot), "/")
	key = filepath.ToSlash(key)
	if prefix == "" {
		return !strings.HasPrefix(key, "/") && key != ""
	}
	return key == prefix || strings.HasPrefix(key, prefix+"/")
}

func (b *captureBrowser) Open(ctx context.Context, key string) (io.ReadCloser, error) {
	if !b.inPool(key) {
		return nil, fmt.Errorf("key 不在采集池内")
	}
	return b.st.Get(ctx, key)
}

func (b *captureBrowser) Remove(ctx context.Context, imageKey string) error {
	if !b.inPool(imageKey) {
		return fmt.Errorf("key 不在采集池内")
	}
	if err := b.st.Remove(ctx, imageKey); err != nil {
		return err
	}
	// 成对删除标签（不存在不报错）。
	return b.st.Remove(ctx, b.codec.LabelKey(imageKey))
}
