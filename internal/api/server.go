// Package api 提供 YOLO 推理的 HTTP 服务。
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

	"github.com/andaoai/yolo-onnx-go/internal/detector"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

// Server 持有检测器并注册路由。
type Server struct {
	det *detector.Detector
	mux *http.ServeMux
}

func NewServer(det *detector.Detector) *Server {
	s := &Server{det: det, mux: http.NewServeMux()}
	s.routes()
	return s
}

func (s *Server) routes() {
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/predict", s.handlePredict)
}

func (s *Server) Handler() http.Handler {
	return logging(s.mux)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

type predictResponse struct {
	TookMs     int64                `json:"took_ms"`
	Count      int                  `json:"count"`
	Detections []detector.Detection `json:"detections"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	confThresh := parseFloatDefault(r.URL.Query().Get("conf"), 0)
	// 可视化参数: vis=1 返回标注后的图片
	vis := r.URL.Query().Get("vis") == "1"

	imgBytes, err := readImageBytes(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		http.Error(w, "decode image: "+err.Error(), http.StatusBadRequest)
		return
	}

	start := time.Now()
	predOpts := []detector.Option(nil)
	if confThresh > 0 {
		predOpts = append(predOpts, detector.WithConfThresh(confThresh))
	}
	dets, err := s.det.Predict(img, predOpts...)
	if err != nil {
		http.Error(w, "inference: "+err.Error(), http.StatusInternalServerError)
		return
	}
	took := time.Since(start).Milliseconds()

	if vis {
		boxed := drawBoxes(img, dets)
		w.Header().Set("Content-Type", "image/jpeg")
		w.Header().Set("X-Took-Ms", strconv.FormatInt(took, 10))
		_ = jpeg.Encode(w, boxed, &jpeg.Options{Quality: 90})
		return
	}

	writeJSON(w, http.StatusOK, predictResponse{
		TookMs:     took,
		Count:      len(dets),
		Detections: dets,
	})
}

// readImageBytes 支持 multipart 上传（字段名 file/image）或原始请求体。
func readImageBytes(r *http.Request) ([]byte, error) {
	ct := r.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "multipart/form-data") {
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			return nil, fmt.Errorf("parse multipart: %w", err)
		}
		for _, field := range []string{"file", "image"} {
			f, _, err := r.FormFile(field)
			if err == nil {
				defer f.Close()
				return io.ReadAll(f)
			}
		}
		return nil, fmt.Errorf("multipart: missing 'file' or 'image' field")
	}
	return io.ReadAll(r.Body)
}

func parseFloatDefault(s string, def float32) float32 {
	if s == "" {
		return def
	}
	v, err := strconv.ParseFloat(s, 32)
	if err != nil {
		return def
	}
	return float32(v)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

// logging 是一个最小的访问日志中间件。
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

// drawBoxes 在图上绘制检测框与标签。
func drawBoxes(src image.Image, dets []detector.Detection) image.Image {
	b := src.Bounds()
	rgba, ok := src.(*image.RGBA)
	if !ok {
		rgba = image.NewRGBA(b)
		draw.Draw(rgba, b, src, b.Min, draw.Src)
	}
	// 复制一份避免改原图
	out := image.NewRGBA(b)
	draw.Draw(out, b, rgba, b.Min, draw.Src)

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
		bgY = y // 顶部空间不够时画在框内顶部
	}

	// 半透明背景条
	bounds := img.Bounds()
	for dy := 0; dy < bgH; dy++ {
		for dx := 0; dx < bgW; dx++ {
			px, py := x+dx, bgY+dy
			if px >= bounds.Min.X && px < bounds.Max.X && py >= bounds.Min.Y && py < bounds.Max.Y {
				img.Set(px, py, c)
			}
		}
	}

	// 文字（黑色，basicfont 仅支持 ASCII，类别名含中文时显示为方块）
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.Black),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(x+pad, bgY+pad+fontH-1),
	}
	d.DrawString(text)
}
