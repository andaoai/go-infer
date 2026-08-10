# go-infer

以 **Go 高并发为中心**的多框架、多算法推理服务。在各种机器（x86 服务器 / ARM 工控机 / 带 GPU 的工作站 / 半受信边缘节点）上完成推理任务，**不绑定具体算法，也不绑定具体推理框架**。

> 当前状态：第一个引擎已落地——ONNX Runtime + YOLO 检测。架构已经按"可插拔引擎"搭好，后续加后端/算法只新增 `internal/engines/<框架>/<任务>/`，不动 HTTP 与调度层。

## 设计理念

```
            ┌─────────────────────────────┐
  HTTP/SSE  │  internal/api   (路由/可视化) │  只依赖 engine 接口
 ─────────► │  internal/sched (worker pool/ │  ← 高并发探索在这里
            │                 batching)    │
            ├─────────────────────────────┤
            │  internal/engine  (抽象接口)  │  Engine/Request/Result
            ├─────────────────────────────┤
            │ engines/onnxruntime/detect   │  ← 已有（检测）
            │ engines/onnxruntime/seg      │  ← 已有（实例分割）
            │ engines/tensorrt/detect      │  ← 待加
            │ engines/ncnn/...             │  ← 待加
            │ engines/llamacpp/generate    │  ← 待加（流式）
            └─────────────────────────────┘
```

- **算法无关**：`Engine` 接口用 `Task` 区分检测/分类/分割/姿态/生成，结果按任务类型返回
- **框架无关**：ONNX Runtime 只是第一个 `Framework()`；TensorRT/NCNN/OpenVINO/CoreML/llama.cpp 都是平行实现
- **机器无关**：通过编译标签、执行提供者（EP）、纯 Go 后端等方式适配不同硬件，目标是"一条命令在目标机上跑起来"
- **并发为核心**：Go 的 goroutine + channel 天然适合 preprocess/infer/postprocess 流水线、攒批（dynamic batching）、多模型并行、流式输出

## 引擎路线图

**推理后端（Framework）**
- ✅ ONNX Runtime（CPU，当前）—— 跨平台、模型生态最广
- ⬜ ONNX Runtime Execution Providers：CUDA / TensorRT / CoreML / OpenVINO / DirectML（一个后端覆盖多硬件）
- ⬜ TensorRT（NVIDIA，FP16/INT8 极致吞吐）
- ⬜ NCNN / MNN（ARM/端侧，无 C++ 运行时依赖，趋近单二进制）
- ⬜ OpenVINO（Intel CPU/iGPU）
- ⬜ llama.cpp / gguf（LLM/VLM 流式生成）
- ⬜ 纯 Go 路径（gorgonia 等）—— 换回 wfmon 式无动态库部署

**算法任务（Task）**
- ✅ 目标检测（YOLO 系列，当前）
- ✅ 实例分割（YOLOv8-seg，输出框 + 掩膜多边形）
- ⬜ 图像分类、旋转框检测、姿态估计
- ⬜ OCR、SAM、CLIP 等视觉模型
- ⬜ LLM/VLM 流式生成（SSE/WebSocket token 流）
- ⬜ 多阶段 pipeline（T1 整图检测 → 裁切 → T2 小图二次检测，对齐已有业务）

**高并发探索方向（`internal/sched`，待建）**
- worker pool：N 个推理 worker 消费请求队列，背压控制
- dynamic batching：在延迟预算内把多个请求攒成一个 batch 喂给模型
- 流水线并行：preprocess（CPU）与 infer（GPU/设备）解耦，重叠执行
- 多引擎路由：按 `?engine=name` 分发，支持同模型多后端 A/B
- 跨机器调度：与 EasyTier 组网联动，把任务调度到有 GPU 的节点
- 流式：生成类任务 SSE/WebSocket 边推理边吐 token

## 当前引擎：ONNX Runtime YOLO 检测

适配 ultralytics 导出的 ONNX 检测模型：
- 输入 `float32[1,3,H,W]`，NCHW，letterbox 等比缩放（填充 114），像素归一化 0~1
- 输出自动识别官方 `[1,4+nc,anchors]` 与转置 `[1,anchors,4+nc]` 两种排布
- 从 ONNX 模型元数据读取真实输入/输出名与输出形状，无需手写
- letterbox 坐标自动还原回原图并 clamp
- 共享张量串行推理，`conf` 阈值支持按请求覆盖

### 实例分割引擎（YOLOv8-seg）

第二个引擎，与检测引擎平行，挂载为 `?engine=yolov8n-seg`。适配 ultralytics 导出的 YOLOv8-seg ONNX：

- 输入同检测：`float32[1,3,H,W]` NCHW letterbox
- **两个输出**：检测头 `[1,4+nc+nm,anchors]`（`nm` 掩膜系数维度，通常 32）+ 原型掩膜 `[1,nm,160,160]`
- 每个保留实例的掩膜 = `sigmoid(系数 · 原型)`，双线性采样、按检测框裁切、`-mask-thr`（默认 0.5）二值化
- 用 Moore 邻域追踪外轮廓，RDP 算法压缩点数，经 letterbox 映射回原图，以多边形点列返回
- 模型文件不存在时**自动跳过**，不影响检测服务；用 `-no-seg` 显式关闭

```bash
# 导出/获取 yolov8n-seg.onnx 放到 models/ 后启动即自动注册
./bin/go-infer
# 或显式指定
./bin/go-infer -seg-name my-seg -seg-model models/best-seg.onnx -mask-thr 0.5

# JSON：每个实例含框 + mask 多边形
curl -X POST -F "file=@examples/bus.jpg" \
  "http://localhost:8080/predict?engine=yolov8n-seg&conf=0.5"

# 可视化：半透明填充 + 多边形轮廓 + 框
curl -X POST -F "file=@examples/bus.jpg" \
  "http://localhost:8080/predict?engine=yolov8n-seg&conf=0.5&vis=1" \
  -o seg.jpg
```

## 快速开始

```bash
# 1. 下载 ONNX Runtime 1.20.0 到 third_party/ + 构建
make build

# 可选：下载官方 yolov8n-seg.onnx 案例模型（需要能访问 github）
make seg-model

# 2. 启动（仓库已附带官方 yolov8n 权重与 COCO 80 类名；seg 模型存在则自动注册）
./bin/go-infer
# 等价于：
# ./bin/go-infer -model models/yolov8n.onnx -classes-file models/coco.names \
#                -imgsz 640 -conf 0.25 -addr :8080
```

### 验证

```bash
# 健康检查
curl http://localhost:8080/health

# 列出已注册引擎
curl http://localhost:8080/engines

# JSON 检测结果
curl -X POST -F "file=@examples/bus.jpg" \
  "http://localhost:8080/predict?conf=0.5"

# 生成画框结果图
curl -X POST -F "file=@examples/bus.jpg" \
  "http://localhost:8080/predict?conf=0.5&vis=1" \
  -o out.jpg
```

仓库附官方权重 [models/yolov8n.onnx](models/yolov8n.onnx)、类名 [models/coco.names](models/coco.names)、测试图 [examples/bus.jpg](examples/bus.jpg)，以及参考输出 [examples/bus_result.jpg](examples/bus_result.jpg)（检测，约 39ms）。分割参考输出见 [examples/bus_seg_result.jpg](examples/bus_seg_result.jpg)。

### 用自己的模型

```python
# ultralytics 导出
from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", imgsz=640, opset=12, simplify=True)
```

```bash
./bin/go-infer -name my-model -model models/best.onnx \
  -classes-file models/my.names -imgsz 640
```

> 注意：`-imgsz` 必须与导出一致；类别数必须与模型输出的 `4+nc` 匹配，否则启动报错。

## HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回已注册引擎数 |
| GET | `/engines` | 列出所有引擎（name/task/framework） |
| POST | `/predict` | 推理，`?engine=` 选引擎（默认第一个），`?conf=` 覆盖阈值，`?vis=1` 返回画框 JPEG |

`/predict` 提交方式：`multipart/form-data`（字段 `file` 或 `image`）或原始请求体（image/* ）。

JSON 响应：
```json
{
  "engine": "yolov8n",
  "task": "detection",
  "took_ms": 39,
  "detections": [
    {"class_id": 0, "class_name": "person", "confidence": 0.89,
     "x1": 671.0, "y1": 384.4, "x2": 810.0, "y2": 880.2}
  ]
}
```

## 命令行参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `-name` | 检测引擎实例名（`?engine=` 选择） | `yolov8n` |
| `-model` | 检测 ONNX 模型路径 | `models/yolov8n.onnx` |
| `-seg-name` | 分割引擎实例名 | `yolov8n-seg` |
| `-seg-model` | 分割 ONNX 模型路径；文件不存在则跳过 | `models/yolov8n-seg.onnx` |
| `-no-seg` | 禁用分割引擎（即使模型存在） | false |
| `-mask-thr` | 分割掩膜二值化阈值 | 0.5 |
| `-classes` | 类别名，逗号分隔 | 看 `-classes-file` |
| `-classes-file` | 类别名文件，一行一个 | `models/coco.names` |
| `-nc` | 类别数（前两者都没给时用） | 1 |
| `-imgsz` | 模型输入尺寸（正方形） | 640 |
| `-conf` | 默认置信度阈值 | 0.25 |
| `-iou` | NMS IoU 阈值 | 0.45 |
| `-addr` | 监听地址 | `:8080` |
| `-ort-lib` | `libonnxruntime.so` 路径 | 自动查找 |

## 项目结构

```
go-infer/
├── cmd/server/main.go              入口：加载 ORT、注册引擎、启动 HTTP
├── internal/
│   ├── engine/                     引擎无关抽象：Engine/Request/Result/Task
│   ├── api/                        HTTP 层，只依赖 engine 接口
│   ├── preprocess/                 可复用视觉工具：letterbox/缩放/NCHW
│   └── engines/
│       └── onnxruntime/
│           ├── detect/             ORT + YOLO 检测引擎
│           └── seg/                ORT + YOLOv8-seg 实例分割引擎
├── models/                         模型权重与类名（yolov8n 为案例）
├── examples/                       bus.jpg 与参考结果图
├── third_party/onnxruntime/        make ort 下载位置（gitignored）
├── Makefile
└── go.mod
```

## 加一个新引擎

实现 [internal/engine/engine.go](internal/engine/engine.go) 的 `Engine` 接口即可：

```go
type Engine interface {
    Name() string
    Task() engine.Task
    Framework() string
    Run(ctx context.Context, req *engine.Request) (engine.Result, error)
    Close() error
}
```

然后在 [cmd/server/main.go](cmd/server/main.go) 里 `srv.Register(myEngine)`。检测类任务返回 `*engine.DetectionResult` 即可自动复用 HTTP 可视化；新任务类型可在 `engine` 包加对应的 `Result` 实现，并在 api 层加渲染。

## 依赖说明

当前 ONNX Runtime 引擎经 CGO 绑定，运行时需要 `libonnxruntime.so`，**不是纯静态二进制**（这一点与 wfmon 不同；端侧/纯 Go 后端是后续探索方向）。`make ort` 自动下载 ORT 1.20.0 到 `third_party/`，程序按 `-ort-lib` → `third_party/` → `LD_LIBRARY_PATH` → 系统路径顺序查找。

## 测试

```bash
make test     # 纯逻辑单测（NMS/letterbox/shape 解析），不需要 ORT .so 也能编译
make build    # 下载 ORT + 构建
```

## 已知限制

- 输入尺寸固定（启动时指定），不支持动态输入尺寸；输出维度不支持 `-1`
- 可视化标签用 Go 内置点阵字体，仅支持 ASCII（中文类别名显示为方块，JSON 不受影响）
- 单引擎实例共享张量、推理串行；高并发/攒批能力在 `internal/sched` 规划中，尚未实现
