// Package api 提供推理服务的 HTTP 层。
//
// 不感知具体推理框架，只依赖 engine.Engine 接口；可同时挂载多个引擎，
// 通过 ?engine=<name> 选择。检测类结果支持 vis=1 可视化。
package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
)

// Server 持有已注册引擎并路由请求。
type Server struct {
	engines        map[string]engine.Engine
	order          []string // 保持注册顺序，第一个为默认
	mux            *http.ServeMux
	recorder       Recorder
	browser        CaptureBrowser
	validator      Validator
	video          VideoService
	maxUploadBytes int64
}

func NewServer() *Server {
	s := &Server{engines: make(map[string]engine.Engine), mux: http.NewServeMux()}
	s.routes()
	return s
}

// CaptureSample 是一次推理结果的采集输入（引擎无关）。
// Result 携带引擎原生结果，采集器自行用 data.ObjectsFromResult 统一转换，
// 避免 api 层重复 type switch。
type CaptureSample struct {
	Engine string
	Image  []byte // 原始上传字节
	W, H   int
	ImgExt string // ".jpg"/".png"
	Result engine.Result
}

// Recorder 是可选的推理采集器（internal/capture 实现）。
type Recorder interface {
	Record(s CaptureSample)
}

// CaptureFile 是采集池里的一张图片（及是否有对应标签）。
type CaptureFile struct {
	Key      string `json:"key"`
	LabelKey string `json:"label_key"`
	Name     string `json:"name"`
	Size     int64  `json:"size"`
	HasLabel bool   `json:"has_label"`
	When     int64  `json:"when"` // mtime，Unix 秒
}

// CaptureDate 是某引擎某天的采集批次。
type CaptureDate struct {
	Date  string        `json:"date"`
	Count int           `json:"count"`
	Files []CaptureFile `json:"files"`
}

// CaptureGroup 是某引擎的全部采集批次。
type CaptureGroup struct {
	Engine string        `json:"engine"`
	Count  int           `json:"count"`
	Dates  []CaptureDate `json:"dates"`
}

// CaptureBrowser 是可选的采集池浏览能力（cmd/server 用 storage+codec 实现）。
type CaptureBrowser interface {
	Groups(ctx context.Context) ([]CaptureGroup, error)
	Open(ctx context.Context, key string) (io.ReadCloser, error)
	Remove(ctx context.Context, imageKey string) error
}

// SetRecorder 挂载采集器；传 nil 关闭采集。
func (s *Server) SetRecorder(r Recorder) { s.recorder = r }

// SetBrowser 挂载采集池浏览器（供前端查看/删除已采集样本）；传 nil 关闭。
func (s *Server) SetBrowser(b CaptureBrowser) { s.browser = b }

// Register 挂载一个引擎。
func (s *Server) Register(e engine.Engine) {
	name := e.Name()
	if _, exists := s.engines[name]; !exists {
		s.order = append(s.order, name)
	}
	s.engines[name] = e
}

func (s *Server) routes() {
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/engines", s.handleEngines)
	s.mux.HandleFunc("/predict", s.handlePredict)
	s.mux.HandleFunc("/api/captures", s.handleCaptures)
	s.mux.HandleFunc("/api/captures/file", s.handleCaptureFile)
	s.mux.HandleFunc("/api/validate/testsets", s.handleValidateTestsets)
	s.mux.HandleFunc("/api/validate", s.handleValidate)
	s.mux.HandleFunc("/api/video/sources", s.handleVideoSources)
	s.mux.HandleFunc("/api/video/upload", s.handleVideoUpload)
	s.mux.HandleFunc("/api/video/stream", s.handleVideoStream)
	s.mux.HandleFunc("/api/video/probe", s.handleVideoProbe)
	s.mux.Handle("/", staticHandler())
}

func (s *Server) Handler() http.Handler { return logging(s.mux) }

func (s *Server) resolve(name string) (engine.Engine, error) {
	if name != "" {
		e, ok := s.engines[name]
		if !ok {
			return nil, fmt.Errorf("engine %q not found", name)
		}
		return e, nil
	}
	if len(s.order) == 0 {
		return nil, fmt.Errorf("no engine registered")
	}
	return s.engines[s.order[0]], nil
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "engines": len(s.engines)})
}

// statsProvider 由带可观测计数的引擎实现（如 *sched.Scheduler）。
// api 不直接依赖 sched，通过本地鸭子接口解耦。
type statsProvider interface {
	Stats() any
}

func (s *Server) handleEngines(w http.ResponseWriter, r *http.Request) {
	infos := make([]map[string]any, 0, len(s.order))
	for _, name := range s.order {
		e := s.engines[name]
		info := map[string]any{
			"name":      name,
			"task":      string(e.Task()),
			"framework": e.Framework(),
		}
		// 若引擎被调度器包装，附带背压/吞吐计数，便于调 -sched-queue。
		if sp, ok := e.(statsProvider); ok {
			info["stats"] = sp.Stats()
		}
		infos = append(infos, info)
	}
	writeJSON(w, http.StatusOK, map[string]any{"count": len(infos), "engines": infos})
}

type predictResponse struct {
	Engine     string             `json:"engine"`
	Task       string             `json:"task"`
	TookMs     int64              `json:"took_ms"`
	Detections []engine.Detection `json:"detections,omitempty"`
	Instances  []engine.Instance  `json:"instances,omitempty"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	engName := r.URL.Query().Get("engine")
	vis := r.URL.Query().Get("vis") == "1"
	var conf float32
	if v := r.URL.Query().Get("conf"); v != "" {
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			conf = float32(f)
		}
	}

	img, raw, format, err := readImage(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	eng, err := s.resolve(engName)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	req := &engine.Request{Image: img}
	if conf > 0 {
		req.Params = map[string]float32{"conf": conf}
	}

	start := time.Now()
	res, err := eng.Run(r.Context(), req)
	if err != nil {
		// 调度层背压/关闭：队列满或引擎关闭时返回 503，客户端可重试。
		if errors.Is(err, engine.ErrBusy) || errors.Is(err, engine.ErrClosed) {
			http.Error(w, "engine busy", http.StatusServiceUnavailable)
			return
		}
		http.Error(w, "inference: "+err.Error(), http.StatusInternalServerError)
		return
	}
	took := time.Since(start).Milliseconds()

	// 视觉类结果支持可视化
	if vis {
		var out image.Image
		switch r := res.(type) {
		case *engine.DetectionResult:
			out = drawBoxes(img, r.Detections)
		case *engine.SegmentationResult:
			out = drawInstances(img, r.Instances)
		default:
			http.Error(w, "engine "+eng.Name()+" does not support visualization", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "image/jpeg")
		w.Header().Set("X-Took-Ms", strconv.FormatInt(took, 10))
		_ = jpeg.Encode(w, out, &jpeg.Options{Quality: 90})
		return
	}

	resp := predictResponse{Engine: eng.Name(), Task: string(res.Task()), TookMs: took}
	switch r := res.(type) {
	case *engine.DetectionResult:
		resp.Detections = r.Detections
	case *engine.SegmentationResult:
		resp.Instances = r.Instances
	}
	if s.recorder != nil {
		b := img.Bounds()
		s.recorder.Record(CaptureSample{
			Engine: eng.Name(),
			Image:  raw,
			W:      b.Dx(),
			H:      b.Dy(),
			ImgExt: "." + format,
			Result: res,
		})
	}
	writeJSON(w, http.StatusOK, resp)
}

// readImage 支持 multipart（字段 file/image）或原始请求体。
// 返回图片、原始字节与格式（"jpeg"/"png"，由 image.DecodeConfig 探测）。
func readImage(r *http.Request) (image.Image, []byte, string, error) {
	var data []byte
	var err error
	ct := r.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "multipart/form-data") {
		if e := r.ParseMultipartForm(32 << 20); e != nil {
			return nil, nil, "", fmt.Errorf("parse multipart: %w", e)
		}
		for _, field := range []string{"file", "image"} {
			f, _, e := r.FormFile(field)
			if e == nil {
				defer f.Close()
				data, err = io.ReadAll(f)
				break
			}
		}
		if data == nil {
			return nil, nil, "", fmt.Errorf("multipart: missing 'file' or 'image' field")
		}
	} else {
		defer r.Body.Close()
		data, err = io.ReadAll(r.Body)
		if err != nil {
			return nil, nil, "", err
		}
	}
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, nil, "", fmt.Errorf("decode image: %w", err)
	}
	return img, data, format, nil
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

// ---------- 采集池浏览 ----------

func (s *Server) handleCaptures(w http.ResponseWriter, r *http.Request) {
	if s.browser == nil {
		writeJSON(w, http.StatusOK, map[string]any{"enabled": false, "groups": []CaptureGroup{}})
		return
	}
	groups, err := s.browser.Groups(r.Context())
	if err != nil {
		http.Error(w, "list captures: "+err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"enabled": true, "groups": groups})
}

func (s *Server) handleCaptureFile(w http.ResponseWriter, r *http.Request) {
	if s.browser == nil {
		http.Error(w, "capture disabled", http.StatusNotFound)
		return
	}
	key := r.URL.Query().Get("key")
	if key == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}
	switch r.Method {
	case http.MethodGet:
		rc, err := s.browser.Open(r.Context(), key)
		if err != nil {
			http.Error(w, "open: "+err.Error(), http.StatusNotFound)
			return
		}
		defer rc.Close()
		switch strings.ToLower(filepath.Ext(key)) {
		case ".jpg", ".jpeg":
			w.Header().Set("Content-Type", "image/jpeg")
		case ".png":
			w.Header().Set("Content-Type", "image/png")
		case ".txt":
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		default:
			w.Header().Set("Content-Type", "application/octet-stream")
		}
		_, _ = io.Copy(w, rc)
	case http.MethodDelete:
		if err := s.browser.Remove(r.Context(), key); err != nil {
			http.Error(w, "remove: "+err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"deleted": key})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func logging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(rw, r)
		log.Printf("%s %s -> %d (%s)", r.Method, r.URL.Path, rw.status, time.Since(start))
	})
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (s *statusWriter) WriteHeader(c int) {
	s.status = c
	s.ResponseWriter.WriteHeader(c)
}

// Flush 透传给底层 ResponseWriter，使 MJPEG 等流式响应能逐帧刷新。
func (s *statusWriter) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap 让 http.NewResponseController/类型断言能访问底层连接能力。
func (s *statusWriter) Unwrap() http.ResponseWriter { return s.ResponseWriter }
