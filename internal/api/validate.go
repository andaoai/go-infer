package api

import (
	"context"
	"errors"
	"mime/multipart"
	"net/http"
	"strconv"

	"github.com/andaoai/go-infer/internal/validate"
)

// 校验相关错误哨兵：实现方返回这些错误，handler 映射到对应 HTTP 状态码。
var (
	ErrValidateBusy      = errors.New("another validation is in progress")
	ErrValidateNoModel   = errors.New("至少上传一个 .onnx 模型")
	ErrValidateNoDataset = errors.New("请选择测试集或上传数据集 zip")
	ErrValidateBadInput  = errors.New("invalid request")
)

// Testset 描述一个启动时配置的默认测试集。
type Testset struct {
	Name  string `json:"name"`
	Split string `json:"split"`
}

// ValidateRequest 是一次网页校验请求的字段（文件句柄 + 表单参数）。
type ValidateRequest struct {
	DetModel *multipart.FileHeader
	SegModel *multipart.FileHeader
	Dataset  *multipart.FileHeader
	Testset  string
	Split    string
	Classes  []string
	ImgSize  int
	Conf     float64
	IoU      float64
	MaskThr  float64
	Limit    int
}

// Validator 是模型校验能力（cmd/server 实现）。
type Validator interface {
	Testsets() []Testset
	Validate(ctx context.Context, req ValidateRequest) (*validate.Report, error)
}

// SetValidator 挂载校验器；传 nil 关闭网页校验。
func (s *Server) SetValidator(v Validator) { s.validator = v }

// SetMaxUploadBytes 设置 multipart 上传体上限（含模型与数据集）。
func (s *Server) SetMaxUploadBytes(n int64) { s.maxUploadBytes = n }

func (s *Server) handleValidateTestsets(w http.ResponseWriter, r *http.Request) {
	if s.validator == nil {
		writeJSON(w, http.StatusOK, map[string]any{"enabled": false, "testsets": []Testset{}})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"enabled": true, "testsets": s.validator.Testsets()})
}

func (s *Server) handleValidate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.validator == nil {
		http.Error(w, "validate disabled", http.StatusNotFound)
		return
	}
	if s.maxUploadBytes > 0 {
		r.Body = http.MaxBytesReader(w, r.Body, s.maxUploadBytes)
	}
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		// MaxBytesReader 在超限时返回 *http.MaxBytesError，语义对应 413。
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			http.Error(w, "上传体积超过上限", http.StatusRequestEntityTooLarge)
		} else {
			http.Error(w, "parse form: "+err.Error(), http.StatusBadRequest)
		}
		return
	}

	req := ValidateRequest{
		Testset: r.FormValue("testset"),
		Split:   r.FormValue("split"),
	}
	if fhs := r.MultipartForm.File["det_model"]; len(fhs) > 0 {
		req.DetModel = fhs[0]
	}
	if fhs := r.MultipartForm.File["seg_model"]; len(fhs) > 0 {
		req.SegModel = fhs[0]
	}
	if fhs := r.MultipartForm.File["dataset"]; len(fhs) > 0 {
		req.Dataset = fhs[0]
	}
	if v := r.FormValue("classes"); v != "" {
		req.Classes = splitCSV(v)
	}
	req.ImgSize = atoiDefault(r.FormValue("imgsz"), 640)
	req.Conf = atofDefault(r.FormValue("conf"), 0.001)
	req.IoU = atofDefault(r.FormValue("iou"), 0.7)
	req.MaskThr = atofDefault(r.FormValue("mask_thr"), 0.5)
	req.Limit = atoiDefault(r.FormValue("limit"), 0)

	report, err := s.validator.Validate(r.Context(), req)
	if err != nil {
		switch {
		case errors.Is(err, ErrValidateBusy):
			http.Error(w, err.Error(), http.StatusConflict)
		case errors.Is(err, ErrValidateNoModel), errors.Is(err, ErrValidateNoDataset), errors.Is(err, ErrValidateBadInput):
			http.Error(w, err.Error(), http.StatusBadRequest)
		default:
			http.Error(w, "validate: "+err.Error(), http.StatusInternalServerError)
		}
		return
	}
	writeJSON(w, http.StatusOK, report)
}

func atoiDefault(s string, def int) int {
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return n
}

func atofDefault(s string, def float64) float64 {
	if s == "" {
		return def
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return def
	}
	return f
}

func splitCSV(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ',' {
			if cur != "" {
				out = append(out, cur)
				cur = ""
			}
			continue
		}
		cur += string(r)
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}
