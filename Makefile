# go-infer：以 Go 高并发为中心的多框架/多算法推理服务
#
# 注意：当前 ONNX Runtime 引擎依赖 CGO 与运行时动态库 libonnxruntime.so，
# 不是纯静态二进制（这与 wfmon 不同；纯 Go/端侧后端是后续探索方向）。
# `make ort` 会把 ONNX Runtime 下载到 third_party/，不污染系统目录。

ORT_VERSION ?= 1.20.0
ORT_DIR     := third_party/onnxruntime
ORT_LIB     := $(ORT_DIR)/lib/libonnxruntime.so
BINARY      := bin/go-infer
SEG_MODEL   := models/yolov8n-seg.onnx

.PHONY: all build ort run test vet fmt clean tidy

all: build

## 下载 ONNX Runtime（Linux x64）
$(ORT_LIB):
	@echo ">> 下载 onnxruntime v$(ORT_VERSION)"
	mkdir -p third_party
	curl -L -o third_party/ort.tgz \
		"https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/onnxruntime-linux-x64-$(ORT_VERSION).tgz"
	mkdir -p $(ORT_DIR)
	tar -xzf third_party/ort.tgz -C $(ORT_DIR) --strip-components=1
	rm third_party/ort.tgz
	@echo ">> 完成: $(ORT_LIB)"

ort: $(ORT_LIB)

## 下载官方 yolov8n-seg.onnx 案例模型（约 6.7MB，需要能访问 github）
$(SEG_MODEL):
	@echo ">> 下载 yolov8n-seg.onnx"
	curl -L -o $(SEG_MODEL) \
		https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.onnx

seg-model: $(SEG_MODEL)

build: ort
	@echo ">> 构建 $(BINARY)"
	CGO_ENABLED=1 go build -o $(BINARY) ./cmd/server

run: build
	./bin/go-infer $(ARGS)

test:
	CGO_ENABLED=1 go test ./...

test-v:
	CGO_ENABLED=1 go test -v ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

tidy:
	go mod tidy

clean:
	rm -rf bin
