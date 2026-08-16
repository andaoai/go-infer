// 网页模型校验服务：接收上传的 .onnx 与数据集 zip，在临时 session 上跑 mAP 校验。
//
// 实现 api.Validator 接口（接口定义在 internal/api，与 browser.go 同模式）。
// 每次请求使用独立 scratch 目录与独立 ORT session，不影响在线常驻引擎；
// 单飞串行（容量 1 信号量）避免重操作争抢 CPU/内存。请求结束后按 defer LIFO
// 先销毁引擎再删 scratch 目录。
package main

import (
	"archive/zip"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/andaoai/go-infer/internal/api"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/detect"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/seg"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/sched"
	"github.com/andaoai/go-infer/internal/storage/local"
	"github.com/andaoai/go-infer/internal/validate"
)

type testsetConfig struct {
	path  string
	split string
}

type validateService struct {
	scratchDir     string
	maxUploadBytes int64
	defaultClasses []string
	testsets       []api.Testset
	testsetPaths   map[string]testsetConfig
	sem            chan struct{}
}

func newValidateService(scratchDir string, maxUploadBytes int64, defaultClasses []string, testsets map[string]testsetConfig) (*validateService, error) {
	if err := os.MkdirAll(scratchDir, 0o755); err != nil {
		return nil, fmt.Errorf("创建校验临时目录 %s: %w", scratchDir, err)
	}
	vs := &validateService{
		scratchDir:     scratchDir,
		maxUploadBytes: maxUploadBytes,
		defaultClasses: defaultClasses,
		testsetPaths:   testsets,
		sem:            make(chan struct{}, 1),
	}
	names := make([]string, 0, len(testsets))
	for name := range testsets {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		cfg := testsets[name]
		vs.testsets = append(vs.testsets, api.Testset{Name: name, Split: cfg.split})
	}
	return vs, nil
}

func (v *validateService) Testsets() []api.Testset { return v.testsets }

func (v *validateService) Validate(ctx context.Context, req api.ValidateRequest) (*validate.Report, error) {
	select {
	case v.sem <- struct{}{}:
		defer func() { <-v.sem }()
	default:
		return nil, api.ErrValidateBusy
	}

	if req.DetModel == nil && req.SegModel == nil {
		return nil, api.ErrValidateNoModel
	}
	if req.Dataset == nil && req.Testset == "" {
		return nil, api.ErrValidateNoDataset
	}

	// 每次请求独立 scratch 目录；defer 顺序保证引擎先 Close、目录后删除。
	scratch := filepath.Join(v.scratchDir, time.Now().Format("20060102-150405")+"-"+randHex(4))
	if err := os.MkdirAll(scratch, 0o755); err != nil {
		return nil, err
	}
	defer os.RemoveAll(scratch)

	c := validateConcurrency()

	// 确定数据集来源：上传 zip 或配置的默认测试集。
	storeRoot, dsPrefix, split, err := v.prepareDataset(scratch, req)
	if err != nil {
		return nil, err
	}
	st, err := local.New(storeRoot)
	if err != nil {
		return nil, err
	}

	// 保存上传的模型文件。
	modelDir := filepath.Join(scratch, "models")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return nil, err
	}
	classes := req.Classes
	if len(classes) == 0 {
		classes = v.defaultClasses
	}

	opts := validate.Options{
		Store:   st,
		Codec:   yolo.New(),
		Root:    dsPrefix,
		Split:   split,
		Limit:   req.Limit,
		Classes: classes,
	}
	opts.Concurrency = c

	// 检测模型。
	if req.DetModel != nil {
		p := filepath.Join(modelDir, "det.onnx")
		if err := saveUpload(p, req.DetModel); err != nil {
			return nil, fmt.Errorf("保存检测模型: %w", err)
		}
		eng, err := detect.New(detect.Config{
			Name: "det-val", ModelPath: p, InputW: req.ImgSize, InputH: req.ImgSize,
			Classes: classes, ConfThresh: float32(req.Conf), IoUThresh: float32(req.IoU),
		})
		if err != nil {
			return nil, fmt.Errorf("加载检测模型: %w", err)
		}
		defer eng.Close()
		engSched := sched.New(eng, sched.Config{Workers: c, MaxBatch: 0})
		defer engSched.Close()
		opts.Jobs = append(opts.Jobs, validate.Job{
			Name: "检测模型", Engine: engSched, Metrics: []validate.MetricSpec{{Label: "box"}},
		})
	}

	// 分割模型：box + mask。
	if req.SegModel != nil {
		p := filepath.Join(modelDir, "seg.onnx")
		if err := saveUpload(p, req.SegModel); err != nil {
			return nil, fmt.Errorf("保存分割模型: %w", err)
		}
		eng, err := seg.New(seg.Config{
			Name: "seg-val", ModelPath: p, InputW: req.ImgSize, InputH: req.ImgSize,
			Classes: classes, ConfThresh: float32(req.Conf), IoUThresh: float32(req.IoU),
			MaskThresh: float32(req.MaskThr),
		})
		if err != nil {
			return nil, fmt.Errorf("加载分割模型: %w", err)
		}
		defer eng.Close()
		engSegSched := sched.New(eng, sched.Config{Workers: c, MaxBatch: 0})
		defer engSegSched.Close()
		opts.Jobs = append(opts.Jobs, validate.Job{
			Name: "分割模型", Engine: engSegSched,
			Metrics: []validate.MetricSpec{{Label: "box"}, {Label: "mask", UseMask: true}},
		})
	}

	return validate.Run(ctx, opts)
}

// prepareDataset 返回存储根、数据集相对前缀、split。
func (v *validateService) prepareDataset(scratch string, req api.ValidateRequest) (storeRoot, dsPrefix, split string, err error) {
	if req.Dataset != nil {
		zipPath := filepath.Join(scratch, "dataset.zip")
		if err := saveUpload(zipPath, req.Dataset); err != nil {
			return "", "", "", fmt.Errorf("保存数据集: %w", err)
		}
		extractDir := filepath.Join(scratch, "dataset")
		if err := os.MkdirAll(extractDir, 0o755); err != nil {
			return "", "", "", err
		}
		if err := extractZip(extractDir, zipPath, v.maxUploadBytes); err != nil {
			return "", "", "", fmt.Errorf("%w: 解压数据集: %v", api.ErrValidateBadInput, err)
		}
		root, detSplit, err := detectDatasetRoot(extractDir, req.Split)
		if err != nil {
			return "", "", "", fmt.Errorf("%w: %v", api.ErrValidateBadInput, err)
		}
		return root, "", detSplit, nil
	}

	cfg, ok := v.testsetPaths[req.Testset]
	if !ok {
		return "", "", "", fmt.Errorf("%w: 未知测试集 %q", api.ErrValidateBadInput, req.Testset)
	}
	split = cfg.split
	if req.Split != "" {
		split = req.Split
	}
	abs, err := filepath.Abs(cfg.path)
	if err != nil {
		return "", "", "", err
	}
	return filepath.Dir(abs), filepath.Base(abs), split, nil
}

// saveUpload 把 multipart 文件流式写入 dst。
func saveUpload(dst string, fh *multipart.FileHeader) error {
	src, err := fh.Open()
	if err != nil {
		return err
	}
	defer src.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, src)
	return err
}

// extractZip 安全解压 zip 到 dst：拒绝绝对路径、路径穿越、符号链接，累计解压字节防 zip bomb。
func extractZip(dst, zipPath string, maxBytes int64) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()

	dstAbs, err := filepath.Abs(dst)
	if err != nil {
		return err
	}
	var written int64
	for _, f := range r.File {
		// 防御符号链接与非常规文件。
		if f.FileInfo().Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("拒绝符号链接: %s", f.Name)
		}
		name := filepath.FromSlash(f.Name)
		if filepath.IsAbs(name) {
			return fmt.Errorf("拒绝绝对路径: %s", f.Name)
		}
		target := filepath.Join(dstAbs, name)
		rel, err := filepath.Rel(dstAbs, target)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			return fmt.Errorf("非法路径（越界）: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		if maxBytes > 0 {
			written += int64(f.UncompressedSize64)
			if written > maxBytes {
				return fmt.Errorf("解压后体积超过上限 %d 字节", maxBytes)
			}
		}
		if err := writeZipFile(f, target); err != nil {
			return err
		}
	}
	return nil
}

func writeZipFile(f *zip.File, dst string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.FileInfo().Mode().Perm())
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, rc)
	return err
}

// detectDatasetRoot 在 base 下查找同时含 images/ 与 labels/ 的目录，并探测 split。
func detectDatasetRoot(base, userSplit string) (string, string, error) {
	var roots []string
	err := filepath.WalkDir(base, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			return nil
		}
		if isDir(filepath.Join(p, "images")) && isDir(filepath.Join(p, "labels")) {
			roots = append(roots, p)
		}
		return nil
	})
	if err != nil {
		return "", "", err
	}
	if len(roots) == 0 {
		return "", "", errors.New("zip 中未找到同时含 images/ 与 labels/ 的数据集根目录")
	}
	if len(roots) > 1 {
		return "", "", fmt.Errorf("找到 %d 个数据集根目录，请在 zip 内只放一个数据集", len(roots))
	}
	root := roots[0]

	entries, err := os.ReadDir(filepath.Join(root, "images"))
	if err != nil {
		return "", "", err
	}
	var splits []string
	for _, e := range entries {
		if e.IsDir() {
			splits = append(splits, e.Name())
		}
	}
	if userSplit != "" {
		if !contains(splits, userSplit) {
			return "", "", fmt.Errorf("数据集中没有 split %q（可用：%v）", userSplit, splits)
		}
		return root, userSplit, nil
	}
	switch len(splits) {
	case 0:
		return "", "", errors.New("images/ 下没有 split 子目录")
	case 1:
		return root, splits[0], nil
	default:
		return "", "", fmt.Errorf("发现多个 split %v，请指定 split", splits)
	}
}

func isDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// testsetFlag 支持重复的 -validate-testset name=path:split 参数。
type testsetFlag struct {
	values map[string]testsetConfig
	order  []string
}

func (f *testsetFlag) String() string { return fmt.Sprintf("%v", f.values) }

func (f *testsetFlag) Set(s string) error {
	eq := strings.IndexByte(s, '=')
	if eq < 0 {
		return fmt.Errorf("testset 格式应为 name=path:split，得到 %q", s)
	}
	name, rest := s[:eq], s[eq+1:]
	colon := strings.LastIndexByte(rest, ':')
	if colon < 0 {
		return fmt.Errorf("testset 缺少 split，应为 name=path:split，得到 %q", s)
	}
	path, split := rest[:colon], rest[colon+1:]
	if name == "" || path == "" || split == "" {
		return fmt.Errorf("testset 的 name/path/split 均不能为空: %q", s)
	}
	if f.values == nil {
		f.values = map[string]testsetConfig{}
	}
	if _, exists := f.values[name]; !exists {
		f.order = append(f.order, name)
	}
	f.values[name] = testsetConfig{path: path, split: split}
	return nil
}

func (f *testsetFlag) toMap() map[string]testsetConfig { return f.values }

// cleanupStaleScratch 删除上次崩溃可能残留的临时校验目录。
func cleanupStaleScratch(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		if e.IsDir() {
			_ = os.RemoveAll(filepath.Join(dir, e.Name()))
		}
	}
}

// validateConcurrency 返回网页/CLI 校验引擎的并发度：与 validate.Options
// 的默认策略一致（min(NumCPU,4)），sched worker 数与 validate 推理并发度
// 对齐，dynamic batching 由调度层完成。
func validateConcurrency() int {
	c := runtime.NumCPU()
	if c > 4 {
		c = 4
	}
	if c < 1 {
		c = 1
	}
	return c
}
