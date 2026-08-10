# YOLO ONNX Go 推理服务
#
# 注意：onnxruntime_go 依赖 CGO 与运行时动态库 libonnxruntime.so，
# 因此本项目不是纯静态二进制（与 wfmon 不同）。
# `make ort` 会把 ONNX Runtime 下载到 third_party/，不污染系统目录。

ORT_VERSION ?= 1.20.0
ORT_DIR     := third_party/onnxruntime
ORT_LIB     := $(ORT_DIR)/lib/libonnxruntime.so
BINARY      := bin/yolo-server

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

build: ort
	@echo ">> 构建 $(BINARY)"
	CGO_ENABLED=1 go build -o $(BINARY) ./cmd/server

run: build
	./$(BINARY) $(ARGS)

test:
	go test ./...

vet:
	go vet ./...

fmt:
	gofmt -w .

tidy:
	go mod tidy

clean:
	rm -rf bin
