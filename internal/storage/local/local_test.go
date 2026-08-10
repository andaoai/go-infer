package local

import (
	"bytes"
	"context"
	"io"
	"sort"
	"strings"
	"testing"
)

func TestPutGetListRemove(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Put(ctx, "a/b.txt", strings.NewReader("hello")); err != nil {
		t.Fatal(err)
	}
	rc, err := s.Get(ctx, "a/b.txt")
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(rc)
	rc.Close()
	if string(got) != "hello" {
		t.Errorf("got %q", got)
	}
	infos, err := s.List(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	var keys []string
	for _, in := range infos {
		keys = append(keys, in.Key)
	}
	sort.Strings(keys)
	if len(keys) != 2 || keys[0] != "a" || keys[1] != "a/b.txt" {
		t.Errorf("List = %v", keys)
	}
	if err := s.Remove(ctx, "a/b.txt"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Get(ctx, "a/b.txt"); err == nil {
		t.Error("删除后应不可读")
	}
	// Remove 不存在的 key 不报错。
	if err := s.Remove(ctx, "missing.txt"); err != nil {
		t.Errorf("删除不存在文件不应报错: %v", err)
	}
}

func TestPathTraversalRejected(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"../etc/passwd", "a/../../b", "/etc/passwd"} {
		err := s.Put(ctx, key, bytes.NewReader([]byte("x")))
		if err == nil {
			t.Errorf("key %q 应被拒绝", key)
		}
	}
}

func TestRemoveAllRefusesRoot(t *testing.T) {
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.RemoveAll(context.Background(), ""); err == nil {
		t.Error("RemoveAll 根目录应被拒绝")
	}
}
