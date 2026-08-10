# go-infer：以 Go 高并发为中心的多框架/多算法推理服务
#
# 注意：当前 ONNX Runtime 引擎依赖 CGO 与运行时动态库 libonnxruntime.so，
# 不是纯静态二进制（这与 wfmon 不同；纯 Go/端侧后端是后续探索方向）。
# `make ort` 会把 ONNX Runtime 下载到 third_party/，不污染系统目录。

ORT_VERSION ?= 1.20.0
ORT_DIR     := third_party/onnxruntime
ORT_LIB     := $(ORT_DIR)/lib/libonnxruntime.so
BINARY      := bin/go-infer
VALIDATE    := bin/validate
SEG_MODEL   := models/yolov8n-seg.onnx
COCO128     := testdata/coco128-seg

.PHONY: all build ort run test test-short vet fmt clean tidy validate coco128-seg

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

$(VALIDATE): ort
	@echo ">> 构建 $(VALIDATE)"
	CGO_ENABLED=1 go build -o $(VALIDATE) ./cmd/validate

## 下载 coco128-seg 校验数据集（约 7MB，需要能访问 github）
$(COCO128):
	@echo ">> 下载 coco128-seg.zip"
	mkdir -p testdata
	curl -L -o testdata/coco128-seg.zip \
		https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip
	cd testdata && unzip -q -o coco128-seg.zip && rm coco128-seg.zip
	@echo ">> 完成: $(COCO128)"

coco128-seg: $(COCO128)

## 在 coco128-seg 上校验检测/分割模型，输出 mAP
validate: $(VALIDATE) $(COCO128)
	$(VALIDATE) -data $(COCO128) -split train2017

run: build
	./bin/go-infer $(ARGS)

test:
	CGO_ENABLED=1 go test ./...

test-short:
	CGO_ENABLED=1 go test -short ./...

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
