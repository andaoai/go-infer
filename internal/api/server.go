// Package api 提供推理服务的 HTTP 层。
//
// 不感知具体推理框架，只依赖 engine.Engine 接口；可同时挂载多个引擎，
// 通过 ?engine=<name> 选择。检测类结果支持 vis=1 可视化。
package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/andaoai/go-infer/internal/engine"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// Server 持有已注册引擎并路由请求。
type Server struct {
	engines map[string]engine.Engine
	order   []string // 保持注册顺序，第一个为默认
	mux     *http.ServeMux
}

func NewServer() *Server {
	s := &Server{engines: make(map[string]engine.Engine), mux: http.NewServeMux()}
	s.routes()
	return s
}

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

type engineInfo struct {
	Name      string `json:"name"`
	Task      string `json:"task"`
	Framework string `json:"framework"`
}

func (s *Server) handleEngines(w http.ResponseWriter, r *http.Request) {
	infos := make([]engineInfo, 0, len(s.order))
	for _, name := range s.order {
		e := s.engines[name]
		infos = append(infos, engineInfo{Name: name, Task: string(e.Task()), Framework: e.Framework()})
	}
	writeJSON(w, http.StatusOK, map[string]any{"count": len(infos), "engines": infos})
}

type predictResponse struct {
	Engine     string             `json:"engine"`
	Task       string             `json:"task"`
	TookMs     int64              `json:"took_ms"`
	Detections []engine.Detection `json:"detections,omitempty"`
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

	img, _, err := readImage(r)
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
		http.Error(w, "inference: "+err.Error(), http.StatusInternalServerError)
		return
	}
	took := time.Since(start).Milliseconds()

	// 检测类结果支持可视化
	if vis {
		if dr, ok := res.(*engine.DetectionResult); ok {
			out := drawBoxes(img, dr.Detections)
			w.Header().Set("Content-Type", "image/jpeg")
			w.Header().Set("X-Took-Ms", strconv.FormatInt(took, 10))
			_ = jpeg.Encode(w, out, &jpeg.Options{Quality: 90})
			return
		}
		http.Error(w, "engine "+eng.Name()+" does not support visualization", http.StatusBadRequest)
		return
	}

	resp := predictResponse{Engine: eng.Name(), Task: string(res.Task()), TookMs: took}
	if dr, ok := res.(*engine.DetectionResult); ok {
		resp.Detections = dr.Detections
	}
	writeJSON(w, http.StatusOK, resp)
}

// readImage 支持 multipart（字段 file/image）或原始请求体。
func readImage(r *http.Request) (image.Image, []byte, error) {
	var data []byte
	var err error
	ct := r.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "multipart/form-data") {
		if e := r.ParseMultipartForm(32 << 20); e != nil {
			return nil, nil, fmt.Errorf("parse multipart: %w", e)
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
			return nil, nil, fmt.Errorf("multipart: missing 'file' or 'image' field")
		}
	} else {
		defer r.Body.Close()
		data, err = io.ReadAll(r.Body)
		if err != nil {
			return nil, nil, err
		}
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("decode image: %w", err)
	}
	return img, data, nil
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
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

// ---------- 检测结果可视化 ----------

func drawBoxes(src image.Image, dets []engine.Detection) image.Image {
	b := src.Bounds()
	canvas, ok := src.(*image.RGBA)
	if !ok {
		canvas = image.NewRGBA(b)
		draw.Draw(canvas, b, src, b.Min, draw.Src)
	}
	out := image.NewRGBA(b)
	draw.Draw(out, b, canvas, b.Min, draw.Src)

	palette := []color.RGBA{
		{0, 255, 0, 255}, {255, 0, 0, 255}, {0, 255, 255, 255},
		{255, 255, 0, 255}, {255, 0, 255, 255}, {0, 128, 255, 255},
	}
	for _, d := range dets {
		c := palette[d.ClassID%len(palette)]
		x1, y1, x2, y2 := int(d.X1), int(d.Y1), int(d.X2), int(d.Y2)
		drawRect(out, x1, y1, x2, y2, c)
		label := fmt.Sprintf("%s %.2f", d.ClassName, d.Confidence)
		drawLabel(out, x1, y1, label, c)
	}
	return out
}

func drawRect(img *image.RGBA, x1, y1, x2, y2 int, c color.Color) {
	for x := x1; x <= x2; x++ {
		img.Set(x, y1, c)
		img.Set(x, y2, c)
	}
	for y := y1; y <= y2; y++ {
		img.Set(x1, y, c)
		img.Set(x2, y, c)
	}
}

func drawLabel(img *image.RGBA, x, y int, text string, c color.Color) {
	const (
		fontW = 7
		fontH = 13
		pad   = 2
	)
	bgW := len(text)*fontW + pad*2
	bgH := fontH + pad*2
	bgY := y - bgH
	if bgY < 0 {
		bgY = y
	}
	bounds := img.Bounds()
	for dy := 0; dy < bgH; dy++ {
		for dx := 0; dx < bgW; dx++ {
			px, py := x+dx, bgY+dy
			if px >= bounds.Min.X && px < bounds.Max.X && py >= bounds.Min.Y && py < bounds.Max.Y {
				img.Set(px, py, c)
			}
		}
	}
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.Black),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(x+pad, bgY+pad+fontH-1),
	}
	d.DrawString(text)
}
