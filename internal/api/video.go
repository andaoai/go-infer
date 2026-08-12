package api

import (
	"context"
	"errors"
	"fmt"
	"log"
	"mime/multipart"
	"net/http"
)

// 视频相关错误哨兵。
var (
	ErrVideoDisabled = errors.New("视频源未开启（未找到 ffmpeg）")
	ErrVideoBusy     = errors.New("视频会话数已达上限")
	ErrVideoBadSrc   = errors.New("无效的视频源")
	ErrVideoTooLarge = errors.New("视频体积超过上限")
)

// VideoSource 是前端可选的一个视频源。
type VideoSource struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Kind       string `json:"kind"`
	Category   string `json:"category"`
	URL        string `json:"url"`
	Note       string `json:"note"`
	BestEffort bool   `json:"best_effort"`
}

// VideoRef 是一次上传视频的引用。
type VideoRef struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// RunOpts 控制一次视频播放会话。
type RunOpts struct {
	FPS  int // 抽帧率（降负载/带宽）
	MaxW int // 长边等比缩放宽度（0 不缩放）
}

// FrameFunc 由 handler 提供：把一帧原始 JPEG 字节写进 MJPEG 响应。
type FrameFunc func(jpegBytes []byte) error

// VideoService 是视频源能力（cmd/server 实现）。
// 它只负责拉流/解码，不做任何推理。
type VideoService interface {
	// Enabled 返回 ffmpeg 是否可用。
	Enabled() bool
	// Sources 返回内置 + 自定义视频源列表。
	Sources() []VideoSource
	// Upload 保存上传的视频到临时区，返回引用。
	Upload(ctx context.Context, file *multipart.FileHeader) (VideoRef, error)
	// MaxUploadBytes 返回上传体积上限（0 表示不限），供 MaxBytesReader 提前截断。
	MaxUploadBytes() int64
	// Acquire 在写 MJPEG 响应头之前占用一个会话槽。
	// 会话已满时返回 ErrVideoBusy（handler 得以在头发送前回 503）。
	Acquire() (release func(), err error)
	// Run 解析 src（内置 ID / upload:<id> / 原始 URL / v4l2:），用 ffmpeg 抽帧，
	// 每帧以原始 JPEG 字节回调 cb。实时流断线时在 ctx 内退避重连；点播源到 EOF 返回 nil。
	Run(ctx context.Context, src string, opts RunOpts, cb FrameFunc) error
	// Probe 用与播放相同的 ffmpeg 参数只抓第一帧，快速判断 src 能否拉到画面。
	// 连通性失败不返回 error，而是体现在 ProbeResult.OK=false（便于前端直接展示原因）；
	// 仅当服务未开启或 src 非法时才返回 error。
	Probe(ctx context.Context, src string) (ProbeResult, error)
}

// ProbeResult 是一次视频源探针的结果。
type ProbeResult struct {
	OK        bool   `json:"ok"`
	LatencyMS int64  `json:"latency_ms"`        // 启动 ffmpeg 到拿到首帧的耗时（毫秒）
	Width     int    `json:"width,omitempty"`   // 首帧宽（能解出时）
	Height    int    `json:"height,omitempty"`  // 首帧高（能解出时）
	Reason    string `json:"reason,omitempty"`  // OK=false 时的失败原因（取自 ffmpeg stderr）
}

// SetVideo 挂载视频服务；传 nil 关闭视频 tab。
func (s *Server) SetVideo(v VideoService) { s.video = v }

func (s *Server) handleVideoSources(w http.ResponseWriter, r *http.Request) {
	enabled := s.video != nil && s.video.Enabled()
	resp := map[string]any{"enabled": enabled, "sources": []VideoSource{}}
	if enabled {
		resp["sources"] = s.video.Sources()
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleVideoUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.video == nil || !s.video.Enabled() {
		http.Error(w, ErrVideoDisabled.Error(), http.StatusServiceUnavailable)
		return
	}
	if cap := s.video.MaxUploadBytes(); cap > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, cap)
	}
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			http.Error(w, "视频体积超过上限", http.StatusRequestEntityTooLarge)
		} else {
			http.Error(w, "parse form: "+err.Error(), http.StatusBadRequest)
		}
		return
	}
	fhs := r.MultipartForm.File["video"]
	if len(fhs) == 0 {
		http.Error(w, "缺少 video 字段", http.StatusBadRequest)
		return
	}
	ref, err := s.video.Upload(r.Context(), fhs[0])
	if err != nil {
		if errors.Is(err, ErrVideoTooLarge) {
			http.Error(w, err.Error(), http.StatusRequestEntityTooLarge)
		} else {
			http.Error(w, "upload: "+err.Error(), http.StatusInternalServerError)
		}
		return
	}
	writeJSON(w, http.StatusOK, ref)
}

func (s *Server) handleVideoStream(w http.ResponseWriter, r *http.Request) {
	if s.video == nil || !s.video.Enabled() {
		http.Error(w, ErrVideoDisabled.Error(), http.StatusServiceUnavailable)
		return
	}
	q := r.URL.Query()
	opts := RunOpts{
		FPS:  atoiDefault(q.Get("fps"), 5),
		MaxW: atoiDefault(q.Get("maxw"), 960),
	}
	src := q.Get("src")
	if src == "" {
		http.Error(w, "缺少 src", http.StatusBadRequest)
		return
	}

	// 先占会话槽：满了在写响应头之前回 503，而不是把错误塞进 MJPEG 流里。
	release, err := s.video.Acquire()
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer release()

	// MJPEG：multipart/x-mixed-replace，浏览器用一个 <img> 直接播放。
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	flusher, canFlush := w.(http.Flusher)
	if !canFlush {
		return
	}

	// ffmpeg 已产出 JPEG，直接把原始字节作为一个 multipart 分片转发，无服务端重编码。
	cb := func(frame []byte) error {
		fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", len(frame))
		if _, err := w.Write(frame); err != nil {
			return err
		}
		if _, err := w.Write([]byte("\r\n")); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	if err := s.video.Run(r.Context(), src, opts, cb); err != nil {
		// 响应头已发出，无法再改状态码；真实原因打到服务端日志。
		// 客户端断开（ctx.Canceled）属正常停止，不记为错误。
		if !errors.Is(err, context.Canceled) {
			log.Printf("视频流 %s 结束: %v", src, err)
		}
	}
}

func (s *Server) handleVideoProbe(w http.ResponseWriter, r *http.Request) {
	if s.video == nil || !s.video.Enabled() {
		http.Error(w, ErrVideoDisabled.Error(), http.StatusServiceUnavailable)
		return
	}
	src := r.URL.Query().Get("src")
	if src == "" {
		http.Error(w, "缺少 src", http.StatusBadRequest)
		return
	}
	res, err := s.video.Probe(r.Context(), src)
	if err != nil {
		// 非法 src（未知内置 ID / 过期上传引用）→ 400；其余异常 → 500。
		if errors.Is(err, ErrVideoBadSrc) {
			http.Error(w, err.Error(), http.StatusBadRequest)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}
	writeJSON(w, http.StatusOK, res)
}
