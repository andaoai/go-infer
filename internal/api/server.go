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
	engines  map[string]engine.Engine
	order    []string // 保持注册顺序，第一个为默认
	mux      *http.ServeMux
	recorder Recorder
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

// SetRecorder 挂载采集器；传 nil 关闭采集。
func (s *Server) SetRecorder(r Recorder) { s.recorder = r }

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

// ---------- 实例分割可视化 ----------

func drawInstances(src image.Image, insts []engine.Instance) image.Image {
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
	for _, ins := range insts {
		c := palette[ins.ClassID%len(palette)]
		for _, poly := range ins.Mask {
			fillPolygon(out, poly, color.RGBA{c.R, c.G, c.B, 90})
			drawPolygon(out, poly, c)
		}
		x1, y1, x2, y2 := int(ins.X1), int(ins.Y1), int(ins.X2), int(ins.Y2)
		drawRect(out, x1, y1, x2, y2, c)
		label := fmt.Sprintf("%s %.2f", ins.ClassName, ins.Confidence)
		drawLabel(out, x1, y1, label, c)
	}
	return out
}

// fillPolygon 用扫描线算法填充多边形（半透明覆盖）。
func fillPolygon(img *image.RGBA, poly []engine.Point, c color.Color) {
	if len(poly) < 3 {
		return
	}
	bounds := img.Bounds()
	minY, maxY := int(poly[0].Y), int(poly[0].Y)
	for _, p := range poly {
		y := int(p.Y)
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}
	}
	if minY < bounds.Min.Y {
		minY = bounds.Min.Y
	}
	if maxY >= bounds.Max.Y {
		maxY = bounds.Max.Y - 1
	}
	cr, cg, cb, ca := c.RGBA()
	fillA := uint8(ca >> 8)
	for y := minY; y <= maxY; y++ {
		// 求扫描线与各边的交点 x。
		var xs []float32
		for i := 0; i < len(poly); i++ {
			a := poly[i]
			b := poly[(i+1)%len(poly)]
			ay, by := int(a.Y), int(b.Y)
			if ay == by {
				continue
			}
			if y >= min(ay, by) && y < max(ay, by) {
				t := float32(y-ay) / float32(by-ay)
				xs = append(xs, a.X+t*(b.X-a.X))
			}
		}
		// 交点排序后两两配对填充。
		for i := 1; i < len(xs); i++ {
			for j := i; j > 0 && xs[j-1] > xs[j]; j-- {
				xs[j-1], xs[j] = xs[j], xs[j-1]
			}
		}
		for i := 0; i+1 < len(xs); i += 2 {
			x1 := int(xs[i])
			x2 := int(xs[i+1])
			if x1 < bounds.Min.X {
				x1 = bounds.Min.X
			}
			if x2 >= bounds.Max.X {
				x2 = bounds.Max.X - 1
			}
			for x := x1; x <= x2; x++ {
				alphaBlend(img, x, y, uint8(cr>>8), uint8(cg>>8), uint8(cb>>8), fillA)
			}
		}
	}
}

// drawPolygon 连接多边形顶点画边线。
func drawPolygon(img *image.RGBA, poly []engine.Point, c color.Color) {
	for i := 0; i < len(poly); i++ {
		a := poly[i]
		b := poly[(i+1)%len(poly)]
		drawLine(img, int(a.X), int(a.Y), int(b.X), int(b.Y), c)
	}
}

// drawLine 用 Bresenham 算法画线。
func drawLine(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx := 1
	if x0 >= x1 {
		sx = -1
	}
	sy := 1
	if y0 >= y1 {
		sy = -1
	}
	err := dx + dy
	bounds := img.Bounds()
	for {
		if x0 >= bounds.Min.X && x0 < bounds.Max.X && y0 >= bounds.Min.Y && y0 < bounds.Max.Y {
			img.Set(x0, y0, c)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

// alphaBlend 将 (r,g,b,a) 以 source-over 混合到 img 的 (x,y)。
func alphaBlend(img *image.RGBA, x, y int, r, g, b, a uint8) {
	i := img.PixOffset(x, y)
	if i < 0 {
		return
	}
	dst := img.Pix[i : i+4 : i+4]
	sa := uint32(a)
	da := uint32(dst[3])
	oa := sa + (da*(255-sa))/255 // 输出 alpha
	if oa == 0 {
		return
	}
	blend := func(s, d uint32) uint8 {
		// (s*sa/255 + d*da/255*(1-sa/255)) / (oa/255)
		return uint8((s*sa*255 + d*da*(255-sa)) / (oa * 255))
	}
	dst[0] = blend(uint32(r), uint32(dst[0]))
	dst[1] = blend(uint32(g), uint32(dst[1]))
	dst[2] = blend(uint32(b), uint32(dst[2]))
	dst[3] = uint8(oa)
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
