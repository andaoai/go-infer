// Package storage 定义与后端无关的字节存储抽象。
//
// 上层（采集、回流、数据集枚举）只依赖 Storage 接口，不直接调用 os 文件 API；
// 本地文件系统是第一个实现（internal/storage/local），未来可加 NAS/S3/OSS 而
// 不动闭环逻辑。key 使用正斜杠分隔的相对路径（如 "engine/20260810/images/a.jpg"），
// 实现负责把 key 安全映射到具体后端。
package storage

import (
	"context"
	"io"
	"time"
)

// Info 描述一个存储对象。
type Info struct {
	Key     string
	Size    int64
	ModTime time.Time
	IsDir   bool
}

// Storage 是只读+写入的字节存储。所有方法接收 context 以便超时/取消。
type Storage interface {
	// Put 把 r 的全部内容写到 key，覆盖已存在对象；必要时创建父"目录"。
	Put(ctx context.Context, key string, r io.Reader) error
	// Get 返回 key 的可读流；调用方负责 Close。
	Get(ctx context.Context, key string) (io.ReadCloser, error)
	// List 列出 prefix 下的全部对象（不含 prefix 自身），不保证顺序。
	List(ctx context.Context, prefix string) ([]Info, error)
	// Stat 返回单个对象信息。
	Stat(ctx context.Context, key string) (Info, error)
	// Remove 删除单个对象；不存在不应返回错误。
	Remove(ctx context.Context, key string) error
	// RemoveAll 递归删除 prefix 下全部对象（类似 os.RemoveAll）。
	RemoveAll(ctx context.Context, prefix string) error
}
