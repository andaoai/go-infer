// 网页视频源服务：用 ffmpeg 把文件/RTSP/HLS/摄像头抽成 JPEG 帧，通过 FrameFunc
// 回调把原始 JPEG 字节交给 HTTP 层做 MJPEG 推送。纯播放，不做任何推理。
//
// 实现 api.VideoService 接口（与 browser.go/validate.go 同模式：接口在 internal/api，
// 实现在 cmd/server）。每次会话独立 ffmpeg 进程；实时流断线在 ctx 内指数退避重连；
// 点播文件到 EOF 正常结束。并发会话数受容量 N 的信号量约束。
package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image/jpeg"
	"io"
	"mime/multipart"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/andaoai/go-infer/internal/api"
	"github.com/andaoai/go-infer/internal/video"
)

type videoService struct {
	ffmpeg         string
	scratch        string
	sources        []api.VideoSource
	maxUploadBytes int64
	maxSessions    int
	uploadTTL      time.Duration

	sem      chan struct{}
	probeSem chan struct{} // 探针独立限流，不占播放会话槽
	mu       sync.Mutex
	uploads  map[string]string // id -> 临时文件绝对路径
}

func newVideoService(ffmpegPath, scratch string, maxUploadBytes int64, maxSessions int, extra []video.Source) (*videoService, error) {
	ff, err := video.LookPath(ffmpegPath)
	if err != nil {
		// ffmpeg 缺失不致命：返回带 ffmpeg="" 的 service，Enabled()=false，前端降级提示。
		ff = ""
	}
	// 启动时清掉上次运行的残留，再重建 uploads 目录。
	cleanupStaleVideo(scratch)
	if err := os.MkdirAll(filepath.Join(scratch, "uploads"), 0o755); err != nil {
		return nil, fmt.Errorf("创建视频临时目录 %s: %w", scratch, err)
	}
	merged := video.MergeSources(extra)
	sources := make([]api.VideoSource, len(merged))
	for i, s := range merged {
		sources[i] = api.VideoSource(s)
	}
	svc := &videoService{
		ffmpeg:         ff,
		scratch:        scratch,
		sources:        sources,
		maxUploadBytes: maxUploadBytes,
		maxSessions:    maxSessions,
		uploadTTL:      2 * time.Hour,
		sem:            make(chan struct{}, maxSessions),
		probeSem:       make(chan struct{}, 4),
		uploads:        map[string]string{},
	}
	go svc.gcUploads()
	return svc, nil
}

func (v *videoService) Enabled() bool              { return v.ffmpeg != "" }
func (v *videoService) Sources() []api.VideoSource { return v.sources }
func (v *videoService) MaxUploadBytes() int64      { return v.maxUploadBytes }

func (v *videoService) resolveSource(src string) (string, bool, error) {
	switch {
	case strings.HasPrefix(src, "upload:"):
		id := strings.TrimPrefix(src, "upload:")
		v.mu.Lock()
		path, ok := v.uploads[id]
		v.mu.Unlock()
		if !ok {
			return "", false, fmt.Errorf("%w: 上传已过期或不存在", api.ErrVideoBadSrc)
		}
		return path, false, nil
	case strings.HasPrefix(src, "url:"):
		u := strings.TrimPrefix(src, "url:")
		return u, video.IsLive(u), nil
	case strings.HasPrefix(src, "v4l2:"):
		return src, true, nil
	default:
		// 内置源 ID
		for _, s := range v.sources {
			if s.ID == src {
				return s.URL, video.IsLive(s.URL), nil
			}
		}
		// 也允许直接把完整 rtsp/https 地址当 src 传。
		if strings.Contains(src, "://") {
			return src, video.IsLive(src), nil
		}
		return "", false, fmt.Errorf("%w: 未知源 %q", api.ErrVideoBadSrc, src)
	}
}

// Acquire 占用一个并发会话槽。
func (v *videoService) Acquire() (func(), error) {
	if !v.Enabled() {
		return func() {}, api.ErrVideoDisabled
	}
	select {
	case v.sem <- struct{}{}:
		return func() { <-v.sem }, nil
	default:
		return func() {}, api.ErrVideoBusy
	}
}

// probeTimeout 是单次探针的上限：连不上或拿不到首帧即判失败。
const probeTimeout = 10 * time.Second

// Probe 用与播放一致的 ffmpeg 参数只抓第一帧，判断 src 能否拉到画面。
// 探针有独立的并发上限（4），不占播放会话槽；失败原因尽量取自 ffmpeg stderr。
func (v *videoService) Probe(ctx context.Context, src string) (api.ProbeResult, error) {
	if !v.Enabled() {
		return api.ProbeResult{}, api.ErrVideoDisabled
	}
	input, live, err := v.resolveSource(src)
	if err != nil {
		return api.ProbeResult{}, err
	}

	// 独立限流：测连通性不应被正在观看的会话挡住，也不应被「全部测试」打爆。
	select {
	case v.probeSem <- struct{}{}:
		defer func() { <-v.probeSem }()
	case <-ctx.Done():
		return api.ProbeResult{}, ctx.Err()
	}

	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()

	start := time.Now()
	var stderr bytes.Buffer
	// 用与播放相同的 low-delay 参数（经 live 传入），只把 fps/长边压小一点省带宽；
	// 这样「探针通过」≈「真的能播」。
	dec, err := video.New(ctx, video.Config{
		FFmpeg: v.ffmpeg, Input: input, FPS: 2, MaxW: 640, Live: live, Stderr: &stderr,
	})
	latency := time.Since(start).Milliseconds()
	if err != nil {
		return api.ProbeResult{OK: false, LatencyMS: latency, Reason: probeReason(err, &stderr)}, nil
	}
	defer dec.Close()

	frame, err := dec.NextJPEG()
	latency = time.Since(start).Milliseconds()
	if err != nil {
		return api.ProbeResult{OK: false, LatencyMS: latency, Reason: probeReason(err, &stderr)}, nil
	}
	res := api.ProbeResult{OK: true, LatencyMS: latency}
	// 顺手解一下首帧分辨率（只解头不解像素，很便宜），给前端展示用。
	if cfg, e := jpeg.DecodeConfig(bytes.NewReader(frame)); e == nil {
		res.Width, res.Height = cfg.Width, cfg.Height
	}
	return res, nil
}

// probeReason 从 ffmpeg stderr 取最后一行非空内容作为失败原因；没有就用 err。
// ffmpeg 以 -loglevel error 运行，最后一行通常就是 Connection refused / timed out 之类。
func probeReason(err error, stderr *bytes.Buffer) string {
	if stderr != nil {
		lines := strings.Split(strings.TrimSpace(stderr.String()), "\n")
		for i := len(lines) - 1; i >= 0; i-- {
			if l := strings.TrimSpace(lines[i]); l != "" {
				return truncateRune(l, 200)
			}
		}
	}
	return truncateRune(err.Error(), 200)
}

// truncateRune 按 rune 截断字符串到至多 n 个字符，避免切断 UTF-8。
func truncateRune(s string, n int) string {
	if n <= 0 {
		return ""
	}
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// Run 驱动一次播放会话。每帧解码出原始 JPEG → cb。
// 调用方需先 Acquire 成功（这样 503 能在写响应头之前返回）。
func (v *videoService) Run(ctx context.Context, src string, opts api.RunOpts, cb api.FrameFunc) error {
	if !v.Enabled() {
		return api.ErrVideoDisabled
	}
	input, live, err := v.resolveSource(src)
	if err != nil {
		return err
	}
	if opts.FPS <= 0 {
		opts.FPS = 5
	}

	backoff := 250 * time.Millisecond
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		err := v.runOnce(ctx, input, live, opts, cb)
		if err == nil || errors.Is(err, io.EOF) {
			// 点播文件到 EOF 是正常结束；实时流意外 EOF 才重连。
			if !live {
				return nil
			}
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if !live {
			// 非实时流出错直接返回（文件不会自己好）。
			return err
		}
		// 实时流断线：退避重连，封顶 8s。
		t := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
		backoff *= 2
		if backoff > 8*time.Second {
			backoff = 8 * time.Second
		}
	}
}

func (v *videoService) runOnce(ctx context.Context, input string, live bool, opts api.RunOpts, cb api.FrameFunc) error {
	dec, err := video.New(ctx, video.Config{
		FFmpeg: v.ffmpeg, Input: input, FPS: opts.FPS, MaxW: opts.MaxW, Live: live,
	})
	if err != nil {
		return err
	}
	defer dec.Close()

	// ffmpeg 往管道写帧是全速的（文件会瞬间解完），播放器需要按帧率匀速输出，
	// 否则浏览器只看到最后一帧。实时流 ffmpeg 本身按源节奏出帧，这里只补一个
	// 最小帧间隔，不会比实时更慢。
	frameGap := time.Second / time.Duration(opts.FPS)
	next := time.Now()
	for {
		frame, err := dec.NextJPEG()
		if err != nil {
			return err
		}
		if !live {
			next = next.Add(frameGap)
			if wait := time.Until(next); wait > 0 {
				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-time.After(wait):
				}
			} else {
				// 解码/发送跟不上帧率时不累积延迟，以当前时间为基准。
				next = time.Now()
			}
		}
		if err := cb(frame); err != nil {
			return err
		}
	}
}

// Upload 保存上传视频到 scratch/uploads，返回引用。
func (v *videoService) Upload(ctx context.Context, fh *multipart.FileHeader) (api.VideoRef, error) {
	if !v.Enabled() {
		return api.VideoRef{}, api.ErrVideoDisabled
	}
	if v.maxUploadBytes > 0 && fh.Size > v.maxUploadBytes {
		return api.VideoRef{}, fmt.Errorf("%w（%d 字节）", api.ErrVideoTooLarge, v.maxUploadBytes)
	}
	id := randHex(8)
	ext := strings.ToLower(filepath.Ext(fh.Filename))
	// 只允许常见视频扩展名，避免把任意文件落地。
	switch ext {
	case ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v":
	default:
		ext = ".mp4"
	}
	dst := filepath.Join(v.scratch, "uploads", id+ext)
	if err := saveUpload(dst, fh); err != nil {
		return api.VideoRef{}, err
	}
	v.mu.Lock()
	v.uploads[id] = dst
	v.mu.Unlock()
	return api.VideoRef{ID: id, Name: fh.Filename}, nil
}

// gcUploads 周期性清理超过 TTL 的上传文件与索引。
func (v *videoService) gcUploads() {
	ticker := time.NewTicker(15 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		v.mu.Lock()
		now := time.Now()
		for id, p := range v.uploads {
			if fi, err := os.Stat(p); err == nil && now.Sub(fi.ModTime()) > v.uploadTTL {
				_ = os.Remove(p)
				delete(v.uploads, id)
			} else if err != nil {
				delete(v.uploads, id)
			}
		}
		v.mu.Unlock()
	}
}

// cleanupStaleVideo 删除 scratch 下的残留（启动时调用，与 validate 的清理一致）。
func cleanupStaleVideo(dir string) {
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

// videoSourceFlag 支持重复的 -video-source name=url 参数（自定义源）。
type videoSourceFlag []video.Source

func (f *videoSourceFlag) String() string {
	ids := make([]string, 0, len(*f))
	for _, s := range *f {
		ids = append(ids, s.ID)
	}
	return strings.Join(ids, ",")
}

func (f *videoSourceFlag) Set(s string) error {
	eq := strings.IndexByte(s, '=')
	if eq < 0 {
		return fmt.Errorf("video-source 格式应为 name=url，得到 %q", s)
	}
	name, url := s[:eq], s[eq+1:]
	if name == "" || url == "" {
		return fmt.Errorf("video-source 的 name/url 均不能为空: %q", s)
	}
	kind := "rtsp"
	category := video.CatTraffic
	switch {
	case strings.Contains(url, ".m3u8"):
		kind = "hls"
		category = video.CatTV
	case strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://"):
		kind = "http"
	}
	*f = append(*f, video.Source{
		ID: "custom-" + sanitizeID(name), Name: name, Kind: kind, Category: category,
		URL: url, BestEffort: true,
	})
	return nil
}

// list 返回自定义源切片。
func (f *videoSourceFlag) list() []video.Source { return []video.Source(*f) }

func sanitizeID(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else {
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}
