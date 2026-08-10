# yolo-onnx-go

使用 Go 语言 + ONNX Runtime 进行 YOLO 模型推理的 HTTP 服务。单模型检测，对齐 ultralytics 导出的 ONNX 检测模型。

## 特性

- **纯 Go 推理逻辑**：letterbox 预处理、NCHW 归一化、NMS 后处理全部 Go 实现
- **兼容官方导出**：自动识别 `[1, 4+nc, anchors]`（默认）与 `[1, anchors, 4+nc]`（转置）两种输出排布
- **自动读取模型元数据**：输入/输出名、输出形状从 ONNX 模型解析，无需手写
- **letterbox 坐标还原**：检测框映射回原图坐标并 clamp
- **HTTP API**：`/health`、`/predict`（JSON 或可视化图片）
- **并发安全**：共享张量串行推理，`conf` 阈值支持按请求覆盖

## 依赖说明

本项目与 wfmon 不同，**不是纯静态二进制**：onnxruntime_go 通过 CGO 绑定 ONNX Runtime，运行时需要 `libonnxruntime.so`。

`make ort` 会自动下载 ONNX Runtime 1.20.0 到 `third_party/`（不污染系统目录），程序启动时按以下顺序查找：

1. 命令行 `-ort-lib` 指定路径
2. `third_party/onnxruntime/lib/`
3. `LD_LIBRARY_PATH`
4. `/usr/lib`、`/usr/local/lib` 等系统路径

## 快速开始

```bash
# 1. 下载 ONNX Runtime + 构建
make build

# 2. 放模型
cp /path/to/your/best.onnx models/best.onnx

# 3. 启动（类别名逗号分隔；或用 -nc 指定类别数生成默认名）
./bin/yolo-server \
  -model models/best.onnx \
  -classes "person,car,dog" \
  -imgsz 640 \
  -conf 0.25 \
  -addr :8080
```

### 案例：官方 yolov8n + bus.jpg

仓库已附带官方权重 [models/yolov8n.onnx](models/yolov8n.onnx)、COCO 80 类名 [models/coco.names](models/coco.names) 和测试图 [examples/bus.jpg](examples/bus.jpg)，可直接跑：

```bash
# 启动服务
./bin/yolo-server -model models/yolov8n.onnx -classes-file models/coco.names

# 另开一个终端：JSON 结果
curl -X POST -F "file=@examples/bus.jpg" "http://localhost:8080/predict?conf=0.5"

# 或生成画框结果图
curl -X POST -F "file=@examples/bus.jpg" "http://localhost:8080/predict?conf=0.5&vis=1" -o out.jpg
```

预期输出 3 个 person + 1 个 bus，参考效果见 [examples/bus_result.jpg](examples/bus_result.jpg)。

## API

### `GET /health`

健康检查。

```json
{"status": "ok"}
```

### `POST /predict`

上传图片做检测。支持两种提交方式：

- `multipart/form-data`，字段名 `file` 或 `image`
- 原始请求体（Content-Type 为 image/jpeg、image/png 等）

**Query 参数：**

| 参数 | 说明 | 默认 |
|------|------|------|
| `conf` | 本次请求的置信度阈值（不影响服务默认值） | 服务启动的 `-conf` |
| `vis` | 设为 `1` 时返回画好框的 JPEG 图片 | 关 |

**JSON 响应：**

```bash
curl -X POST -F "file=@test.jpg" "http://localhost:8080/predict?conf=0.5"
```

```json
{
  "took_ms": 39,
  "count": 4,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.89,
      "x1": 671.0, "y1": 384.4,
      "x2": 810.0, "y2": 880.2
    }
  ]
}
```

**可视化响应：**

```bash
curl -X POST -F "file=@test.jpg" "http://localhost:8080/predict?conf=0.5&vis=1" -o out.jpg
```

## 命令行参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `-model` | ONNX 模型路径 | `models/best.onnx` |
| `-classes` | 类别名，逗号分隔 | 按 `-classes-file` 或 `-nc` |
| `-classes-file` | 类别名文件，一行一个 | 空 |
| `-nc` | 类别数（前两者均未提供时用） | 1 |
| `-imgsz` | 模型输入尺寸（正方形） | 640 |
| `-conf` | 默认置信度阈值 | 0.25 |
| `-iou` | NMS IoU 阈值 | 0.45 |
| `-addr` | 监听地址 | `:8080` |
| `-ort-lib` | `libonnxruntime.so` 路径 | 自动查找 |

## 项目结构

```
yolo-onnx-go/
├── cmd/server/main.go        # 入口：参数解析、ORT 初始化、HTTP 启动
├── internal/
│   ├── detector/             # 模型加载/预处理/推理/后处理（NMS）
│   │   ├── detector.go       # ONNX session、Predict、输出解析
│   │   ├── preprocess.go     # letterbox、双线性缩放、NCHW 归一化
│   │   └── detector_test.go  # 纯 Go 逻辑单测
│   └── api/                  # HTTP handler 与可视化
│       └── server.go
├── models/                   # 放 .onnx 模型（gitignored）
├── third_party/onnxruntime/  # make ort 下载位置（gitignored）
├── Makefile
└── go.mod
```

## 导出模型（ultralytics）

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

> 注意：`-imgsz` 必须与导出时一致；输出锚点数由模型形状决定（640 输入通常为 8400）。

## 测试

```bash
# 纯逻辑单测（不需要 ONNX Runtime .so 也能编译运行）
go test ./...

# 完整构建
make build
```

## 已知限制

- 输入尺寸固定（启动时指定），不支持动态输入尺寸
- 不支持动态输出维度（`-1`），需导出为固定 shape
- 可视化标签使用 Go 内置点阵字体，仅支持 ASCII；中文类别名会显示为方块（JSON 返回不受影响）
- 单实例串行推理；如需更高吞吐可运行多实例前置负载均衡
