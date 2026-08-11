package main

import (
	"archive/zip"
	"context"
	"errors"
	"mime/multipart"
	"os"
	"path/filepath"
	"testing"

	"github.com/andaoai/go-infer/internal/api"
)

// makeZip 在 w 中写入一组 {name, content}，可选地把某个文件标记为目录。
func writeZip(t *testing.T, path string, files map[string]string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zw := zip.NewWriter(f)
	for name, content := range files {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestExtractZipTraversalRejected(t *testing.T) {
	dir := t.TempDir()
	zipPath := filepath.Join(dir, "bad.zip")
	f, _ := os.Create(zipPath)
	zw := zip.NewWriter(f)
	w, _ := zw.Create("../../../etc/passwd")
	w.Write([]byte("x"))
	zw.Close()
	f.Close()

	err := extractZip(filepath.Join(dir, "out"), zipPath, 0)
	if err == nil {
		t.Fatal("含 ../ 路径的 zip 应被拒绝")
	}
}

func TestExtractZipHappy(t *testing.T) {
	dir := t.TempDir()
	zipPath := filepath.Join(dir, "good.zip")
	writeZip(t, zipPath, map[string]string{
		"ds/images/val/a.jpg": "img",
		"ds/labels/val/a.txt": "0 0.5 0.5 0.2 0.2",
	})
	out := filepath.Join(dir, "out")
	if err := extractZip(out, zipPath, 0); err != nil {
		t.Fatalf("解压失败: %v", err)
	}
	if _, err := os.Stat(filepath.Join(out, "ds", "images", "val", "a.jpg")); err != nil {
		t.Errorf("图片未解压: %v", err)
	}
}

func TestDetectDatasetRootWithWrapper(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "coco128-seg")
	os.MkdirAll(filepath.Join(root, "images", "train2017"), 0o755)
	os.MkdirAll(filepath.Join(root, "labels", "train2017"), 0o755)

	got, split, err := detectDatasetRoot(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if got != root {
		t.Errorf("root = %s, want %s", got, root)
	}
	if split != "train2017" {
		t.Errorf("split = %s, want train2017", split)
	}
}

func TestDetectDatasetRootNoWrapper(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "images", "val"), 0o755)
	os.MkdirAll(filepath.Join(dir, "labels", "val"), 0o755)

	got, split, err := detectDatasetRoot(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if got != dir {
		t.Errorf("root = %s, want %s", got, dir)
	}
	if split != "val" {
		t.Errorf("split = %s, want val", split)
	}
}

func TestDetectDatasetRootMultiSplitRequiresUser(t *testing.T) {
	dir := t.TempDir()
	for _, s := range []string{"train", "val"} {
		os.MkdirAll(filepath.Join(dir, "images", s), 0o755)
		os.MkdirAll(filepath.Join(dir, "labels", s), 0o755)
	}
	if _, _, err := detectDatasetRoot(dir, ""); err == nil {
		t.Fatal("多 split 且未指定应报错")
	}
	got, split, err := detectDatasetRoot(dir, "val")
	if err != nil {
		t.Fatal(err)
	}
	if split != "val" || got != dir {
		t.Errorf("got root=%s split=%s", got, split)
	}
}

func TestValidateSemaphoreNoModelFastFail(t *testing.T) {
	vs, err := newValidateService(t.TempDir(), 1<<30, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// 无模型应在拿信号量之后快速返回 ErrValidateNoModel，不触碰 ORT。
	_, err = vs.Validate(context.Background(), api.ValidateRequest{})
	if !errors.Is(err, api.ErrValidateNoModel) {
		t.Fatalf("want ErrValidateNoModel, got %v", err)
	}
}

func TestValidateNoDataset(t *testing.T) {
	vs, err := newValidateService(t.TempDir(), 1<<30, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// DetModel 非空但无数据集，应在保存模型前返回 ErrValidateNoDataset。
	_, err = vs.Validate(context.Background(), api.ValidateRequest{DetModel: &multipart.FileHeader{Filename: "det.onnx"}})
	if !errors.Is(err, api.ErrValidateNoDataset) {
		t.Fatalf("want ErrValidateNoDataset, got %v", err)
	}
}

func TestValidateUnknownTestset(t *testing.T) {
	vs, _ := newValidateService(t.TempDir(), 1<<30, nil, map[string]testsetConfig{})
	_, err := vs.Validate(context.Background(), api.ValidateRequest{
		DetModel: &multipart.FileHeader{Filename: "det.onnx"}, Testset: "nope",
	})
	if !errors.Is(err, api.ErrValidateBadInput) {
		t.Fatalf("want ErrValidateBadInput, got %v", err)
	}
}

func TestTestsetsSorted(t *testing.T) {
	vs, err := newValidateService(t.TempDir(), 1<<30, nil, map[string]testsetConfig{
		"zeta":  {path: "/a", split: "v"},
		"alpha": {path: "/b", split: "v"},
	})
	if err != nil {
		t.Fatal(err)
	}
	ts := vs.Testsets()
	if len(ts) != 2 || ts[0].Name != "alpha" || ts[1].Name != "zeta" {
		t.Fatalf("testsets 未排序: %+v", ts)
	}
}
