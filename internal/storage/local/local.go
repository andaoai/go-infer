// Package local 实现基于本地文件系统的 storage.Storage。
//
// 所有 key 映射到 <root>/<key>，并在解析后校验最终路径仍位于 root 之内，
// 防止 "../" 路径穿越。Put 流式落盘，List 用 WalkDir 递归枚举。
package local

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/andaoai/go-infer/internal/storage"
)

// Storage 是本地文件系统后端。
type Storage struct {
	root string
}

// New 创建一个以 root 为根的本地存储；root 不存在时自动创建。
func New(root string) (*Storage, error) {
	if root == "" {
		return nil, fmt.Errorf("local storage root 不能为空")
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, fmt.Errorf("创建存储根目录 %s: %w", root, err)
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	return &Storage{root: abs}, nil
}

// Root 返回绝对根目录（装配日志用）。
func (s *Storage) Root() string { return s.root }

// resolve 把 key 映射到 root 内的绝对路径并做穿越校验。
func (s *Storage) resolve(key string) (string, error) {
	if key == "" {
		return s.root, nil
	}
	// 拒绝绝对路径与反斜杠/盘符等。
	native := filepath.FromSlash(key)
	if filepath.IsAbs(native) {
		return "", fmt.Errorf("非法 key（必须是相对路径）: %q", key)
	}
	// Clean 相对路径；结果若跳出当前层（以 .. 开头）则拒绝。
	rel := filepath.Clean(native)
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("非法 key（路径越界）: %q", key)
	}
	if rel == "." {
		return s.root, nil
	}
	p := filepath.Join(s.root, rel)
	// 二次确认：最终路径必须在 root 之下。
	relToRoot, err := filepath.Rel(s.root, p)
	if err != nil || relToRoot == ".." || strings.HasPrefix(relToRoot, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("非法 key（路径越界）: %q", key)
	}
	return p, nil
}

func (s *Storage) Put(ctx context.Context, key string, r io.Reader) error {
	p, err := s.resolve(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	f, err := os.Create(p)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := io.Copy(f, ctxReader(ctx, r)); err != nil {
		_ = os.Remove(p)
		return err
	}
	return nil
}

func (s *Storage) Get(ctx context.Context, key string) (io.ReadCloser, error) {
	p, err := s.resolve(key)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	return f, nil
}

func (s *Storage) List(ctx context.Context, prefix string) ([]storage.Info, error) {
	base, err := s.resolve(prefix)
	if err != nil {
		return nil, err
	}
	var out []storage.Info
	err = filepath.WalkDir(base, func(p string, d fs.DirEntry, werr error) error {
		if werr != nil {
			if os.IsNotExist(werr) {
				return nil
			}
			return werr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if p == base {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return nil
		}
		rel, err := filepath.Rel(s.root, p)
		if err != nil {
			return nil
		}
		out = append(out, storage.Info{
			Key:     filepath.ToSlash(rel),
			Size:    fi.Size(),
			ModTime: fi.ModTime(),
			IsDir:   d.IsDir(),
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (s *Storage) Stat(ctx context.Context, key string) (storage.Info, error) {
	p, err := s.resolve(key)
	if err != nil {
		return storage.Info{}, err
	}
	fi, err := os.Stat(p)
	if err != nil {
		return storage.Info{}, err
	}
	rel, _ := filepath.Rel(s.root, p)
	return storage.Info{
		Key:     filepath.ToSlash(rel),
		Size:    fi.Size(),
		ModTime: fi.ModTime(),
		IsDir:   fi.IsDir(),
	}, nil
}

func (s *Storage) Remove(ctx context.Context, key string) error {
	p, err := s.resolve(key)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (s *Storage) RemoveAll(ctx context.Context, prefix string) error {
	p, err := s.resolve(prefix)
	if err != nil {
		return err
	}
	if p == s.root {
		// 禁止清空整个存储根。
		return fmt.Errorf("拒绝 RemoveAll 存储根目录")
	}
	if err := os.RemoveAll(p); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Now 暴露给需要注入时间的测试/调用方（保留以备后用）。
var Now = time.Now

// ctxReader 在拷贝时响应 ctx 取消。
type readerCtx struct {
	ctx context.Context
	r   io.Reader
}

func (c readerCtx) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}

func ctxReader(ctx context.Context, r io.Reader) io.Reader {
	return readerCtx{ctx: ctx, r: r}
}

var _ storage.Storage = (*Storage)(nil)
