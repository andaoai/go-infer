# go-infer

以 **Go 高并发为中心**的多框架、多算法推理服务。在各种机器（x86 服务器 / ARM 工控机 / 带 GPU 的工作站 / 半受信边缘节点）上完成推理任务，**不绑定具体算法，也不绑定具体推理框架**。

> 当前状态：首个示范组合已落地——ONNX Runtime + YOLO 检测/分割 + 本地 FS + YOLO 标签。架构沿**任务 / 后端 / 存储 / 格式**四轴可插拔（`engine`/`storage`/`format` + 规范模型 `data`），换后端、存储或标签格式只新增实现，不动 HTTP、采集、回流、校验等闭环逻辑。
>
> 📐 架构与数据闭环（训练→部署→采集→回流重训）见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

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
| `-capture-*` | 推理采集（目录/采样率/低置信/配额），见[闭环章节](#部署--采集--回流闭环) | 默认关闭 |
| `-validate-*` | 网页校验（临时目录/上传上限/默认测试集），见[网页校验](#网页校验上传新模型直接跑-map) | — |
| `-video-*` | 视频源实时预览（ffmpeg/临时目录/会话上限/上传上限/自定义源），见[视频源](#视频源实时预览视频rtsphls摄像头) | ffmpeg 可用则开 |

## 项目结构

```
go-infer/
├── cmd/
│   ├── server/main.go              HTTP 服务入口：加载 ORT、注册引擎、启动服务（含采集开关）
│   ├── validate/main.go            数据集校验入口：跑模型、算 mAP
│   └── dataset/main.go             数据集回流入口：采集池 → 训练集
├── internal/
│   ├── engine/                     后端轴抽象：Engine/Request/Result/Task
│   ├── data/                       规范模型 Object/BBox/Point + ObjectsFromResult
│   ├── storage/                    存储轴：Storage 接口 + local/ 实现
│   ├── format/                     格式轴：Codec 接口 + yolo/ 实现
│   ├── metric/                     IoU、栅格化、COCO AP/mAP（消费 data.Object）
│   ├── api/                        HTTP 层，只依赖接口；可选 Recorder 钩子
│   ├── capture/                    推理采集器：策略采样 + 异步落盘 + 配额淘汰
│   ├── promote/                    采集池按引擎合并进训练集 + 生成 data.yaml
│   ├── validate/                   网页/CLI 共用的 mAP 计算库
│   ├── video/                      ffmpeg 抽帧解码器 + 内置视频源（纯标准库）
│   ├── appcfg/                     类别名加载（server/validate/dataset 共用）
│   ├── ortenv/                     ORT 动态库查找与初始化（共用）
│   ├── preprocess/                 可复用视觉工具：letterbox/缩放/NCHW
│   └── engines/
│       └── onnxruntime/
│           ├── detect/             ORT + YOLO 检测引擎
│           └── seg/                ORT + YOLOv8-seg 实例分割引擎
├── models/                         模型权重与类名（yolov8n 为案例）
├── examples/                       bus.jpg 与参考结果图
├── dataset/                        运行期数据（gitignored）：pool 采集池 / <engine> 训练集 / versions
├── testdata/                       coco128-seg 等校验数据集（gitignored）
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

## 数据集校验（mAP）

`cmd/validate` 在带标注的 YOLO 数据集上跑检测/分割模型并输出 mAP，用于回归校验（防坐标错位、阈值退化、后处理 bug）。读取标准 YOLO 目录结构：

```
<root>/images/<split>/*.jpg
<root>/labels/<split>/*.txt   # cls x1 y1 x2 y2 ...（seg 多边形）或 cls cx cy w h（det 框）
```

一份 seg 标签同时提供掩膜 GT（多边形本身）和检测框 GT（点列 min/max），因此可同时校验两个模型、三种指标。

```bash
# 下载官方 coco128-seg（COCO train2017 前 128 张，约 7MB）+ 构建校验器 + 跑全量
make validate

# 等价于：
make coco128-seg
CGO_ENABLED=1 go build -o bin/validate ./cmd/validate
./bin/validate -data testdata/coco128-seg -split train2017

# 只跑前 N 张冒烟；指定自己的数据集/模型
./bin/validate -data /path/to/dataset -split val -limit 16 \
  -det-model models/best-det.onnx -seg-model models/best-seg.onnx \
  -classes-file models/my.names
```

输出三张表：检测模型 box mAP、分割模型 box mAP、分割模型 mask mAP，各含 `mAP@.5` 与 COCO 风格 `mAP@.50:.95`，并附每类 AP@.5。

实现要点（`internal/`）：
- `format/yolo` 解析 YOLO det（5 列）/ seg（多边形）标签，图片用标准库解码，无新依赖
- `metric` 框 IoU、多边形扫描线栅格化（掩膜降采样到长边 256 的位集 + popcount）、COCO 风格按类贪心匹配 + 全点插值 AP；匹配和栅格化只做一次，10 个 IoU 阈值共享
- 全程消费规范模型 `data.Object`，引擎结果经唯一一处 `data.ObjectsFromResult` 转换；校验逻辑与后端、标签格式、存储均解耦

> 定位是回归/冒烟校验，不是中立 benchmark：coco128 是 COCO train2017 子集，官方 nano 权重在其上训练过，指标偏高；且 Go 掩膜双线性采样约定与 ultralytics 略有差异（raw mask IoU≈0.92），mask mAP 系统性低几个点属正常。实测全 128 张与 ultralytics `model.val()` 同参数结果接近（det box mAP@.5 ≈0.55 vs 官方 0.61，seg mask mAP@.5 ≈0.46 vs 官方 0.55）。

### 网页校验（上传新模型直接跑 mAP）

看板第三个 tab「验证」让你在更新算法模型时直接把新的 .onnx 拖进网页、选测试集，一键拿到 mAP 报告——不用 SSH 上机器敲命令。检测/分割模型各传一个（缺哪个自动跳过），测试集二选一：启动时用 `-validate-testset name=path:split` 配好的默认集，或当场上传一个数据集 ZIP。

```bash
./bin/go-infer \
  -validate-dir runs/validate \
  -validate-max-upload 2GB \
  -validate-testset coco128-seg=testdata/coco128-seg:train2017
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `-validate-dir` | 上传模型/数据集解压的临时目录；每次请求独立子目录，跑完即删 | `runs/validate` |
| `-validate-max-upload` | 单次上传体上限（模型+数据集，`500MB`/`2GB`） | `2GB` |
| `-validate-testset` | 网页可选默认测试集，格式 `name=path:split`；可重复指定多个 | 空 |

行为与边界：
- 与 `cmd/validate` 共用同一份 `internal/validate` 逻辑，网页结果和命令行逐位一致（已对拍 coco128-seg 前 16 张）。
- 每次请求用**独立 scratch 目录 + 独立 ORT session**，跑完 `eng.Close()` 再删目录，不动在线常驻引擎；单飞串行（容量 1 信号量），第二个并发请求返回 `409`。
- 上传 ZIP 自动探测数据集根（含 `images/`+`labels/` 的目录，是否带顶层包裹目录均可）与 split；多 split 时需在高级参数里指定。
- 安全：`http.MaxBytesReader` 限体积（超限 `413`）；ZIP 解压防 zip-slip（拒绝绝对路径/`..` 越界/符号链接）并累计解压字节防 zip bomb；结果**只在响应里返回、不落盘、不存历史**。

### 视频源（纯播放器：视频/RTSP/HLS/摄像头）

看板第四个 tab「视频源」是一个**独立的视频播放器**，与推理无关：选源 → ffmpeg 拉流 → 浏览器里直接看原始画面，不跑模型、不画框、不选引擎。三种来源：

1. **上传本地视频文件**（mp4/mov/mkv/avi/webm）——临时落到 `runs/video/uploads/`，预览期间可用，2 小时后自动清理；
2. **自定义 URL**——直接填 `rtsp://…`、`https://….m3u8`（HLS）、`rtmp://…`、`v4l2:/dev/video0` 等；
3. **内置免费公共源**——按「演示流 / 电视直播 / 自然风光」分类，内置 Big Buck Bunny RTSP、Apple/Mux HLS、Al Jazeera、France 24、DW、NHK World、Red Bull TV、NASA 频道等公开流，点开即用（双击源直接播放）。

解码走机器上的 **ffmpeg 外部进程**（不引入任何 Go 侧视频依赖）：ffmpeg 按指定 fps 抽帧、长边缩放到 `maxw` 减带宽，以 `mjpeg` 打到 stdout，Go 侧按 JPEG SOI/EOI 切出原始 JPEG 字节，**不再解码/重编码**，直接以 **MJPEG over HTTP**（`multipart/x-mixed-replace`）转发给浏览器——前端就是一个 `<img>`，无需任何播放库。点播文件按帧率匀速输出（实时播放），实时流按源节奏出帧。

```bash
./bin/go-infer \
  -video-ffmpeg ffmpeg \
  -video-scratch runs/video \
  -video-max-sessions 3 \
  -video-max-upload 1GB \
  -video-source 前门=rtsp://user:pass@192.168.1.10:554/stream1
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `-video-ffmpeg` | ffmpeg 可执行路径；找不到则视频 tab 自动降级为"不可用"，其他 tab 不受影响 | `ffmpeg` |
| `-video-scratch` | 上传视频临时目录（启动时清空残留） | `runs/video` |
| `-video-max-sessions` | 并发播放会话上限；满了在写响应头前返回 `503` | `3` |
| `-video-max-upload` | 单次上传视频体积上限（`500MB`/`1GB`），超限 `413` | `1GB` |
| `-video-source` | 追加/覆盖内置源，格式 `name=url`；与内置同名则覆盖；可重复指定 | 空 |

行为与边界：
- **纯播放**：不跑任何模型、不做检测/分割、不导出视频、不存帧、不产生持久产物；停止/关页面/切 tab 即断开，ffmpeg 子进程随之被 kill 并回收。
- 点播文件按设定 fps **匀速播放**（不是瞬间解完）；实时流（RTSP/RTMP/v4l2）走 TCP、低延迟参数，意外断线在连接上下文内指数退避重连（250ms→8s 封顶），HLS/文件到 EOF 正常结束。
- 并发会话数受容量 N 的信号量约束；会话槽在写 MJPEG 响应头**之前**占用，所以满员能干净地返回 `503` 而不是把错误塞进流里。
- ffmpeg 子进程设置了 Linux `Pdeathsig`：即使 go-infer 被 `kill -9`，内核也会回收其 ffmpeg，不留孤儿进程。
- 内置公共源标记为 **best-effort**：公共直播（尤其 IPTV）经常限流/防盗链/下线，连不上时前端提示失败（真实原因在服务端日志）。要看稳定的监控画面，主路径是填自己摄像头的 RTSP，或用 `-video-source` 预置；公共交通/监控 HLS 很少长期可用，故内置以稳定演示流 + 公开电视直播打底。
- 安全与定位：这是内网受信操作者工具，自定义 URL 直接交给 ffmpeg，**不要把服务暴露到公网**；上传文件用随机 id 命名、只保留白名单扩展名、按 TTL 清理。
- ffmpeg 是**运行时可选依赖**（不是编译依赖，单二进制部署不变）：装了就开视频 tab，没装就优雅关闭。

## 部署 → 采集 → 回流闭环

训练好新模型后的标准流程是：导出 ONNX → 用 `cmd/validate` 在固定 test 集上跑 mAP 门禁 → 停服替换模型 → 起服上线。线上推理时按策略把**原图 + YOLO 伪标签**采集下来，审核修正后合并回训练集，交给 Python 侧重训。Go 侧只负责"攒数据 + 出数据集版本"，重训仍在 ultralytics。

### 1. 开启推理采集（server）

采集默认**关闭**，显式指定目录才开启，避免意外落盘：

```bash
./bin/go-infer \
  -capture-dir dataset/pool \
  -capture-rate 0.1 \
  -capture-low-conf 0.25 \
  -capture-quota 5GB
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `-capture-dir` | 采集池根目录；留空不采集 | 空 |
| `-capture-rate` | 普通样本采样概率 0~1 | 0.1 |
| `-capture-low-conf` | 最高置信度低于此值的"不确定样本"必存 | 0.25 |
| `-capture-quota` | 采集池总容量上限（`500MB`/`5GB`/`0` 不限） | 5GB |
| `-capture-buffer` | 异步写入队列长度 | 256 |

采集策略：**无检测的 hard negative 必存**、**最高置信度低于阈值的不确定样本必存**、其余按 `-capture-rate` 采样。写入是异步的，不阻塞推理响应；队列满时丢最旧待写以保响应不卡。

落盘布局按**模型名 + 日期**两层分组，按天分开、按引擎隔离：

```
dataset/pool/<engine>/<YYYYMMDD>/images/HHMMSS_<rand>.jpg
dataset/pool/<engine>/<YYYYMMDD>/labels/HHMMSS_<rand>.txt
```

标签直接是 YOLO 格式（det `cls cx cy w h`；seg `cls x1 y1 ...` 多边形），所以采集池本身就是一个能用 X-AnyLabeling/Label Studio 打开、改标签的数据集。超出 `-capture-quota` 时按文件 mtime 从最旧的天目录开始成对删除 image+label。

### 2. 审核

用任意 YOLO 标注工具直接打开 `dataset/pool/<engine>/<date>/` 修正标签即可，Go 不内置审核 UI。无标签的图片是 hard negative（推理无检测），审核时按需补标签。

### 3. 回流到训练集（dataset promote）

```bash
make dataset                        # 构建 cmd/dataset（纯 Go，不依赖 ORT）

# 全部引擎、全部日期合并（移动，清空采集池）
./bin/dataset promote --pool dataset/pool --into dataset --classes models/coco.names

# 只提升某个引擎、某天的批次
./bin/dataset promote --engine yolov8n-seg --date 20260810 ...

# 复制而非移动（保留采集池原件）
./bin/dataset promote --copy ...
```

每个引擎对应一个**独立** YOLO 数据集 `dataset/<engine>/`（不同模型可能类别体系不同，不混标签），并生成可直接喂给 ultralytics 的 `data.yaml`：

```
dataset/<engine>/
├── images/train/<date>_<stem>.jpg
├── labels/train/<date>_<stem>.txt
└── data.yaml
dataset/versions/<engine>-<timestamp>.json   # 本次提升的来源/数量清单
```

`val/`、`test/` 目录按需自行补充（`data.yaml` 已预留路径）；`test/` 仅供 `cmd/validate` 做发版门禁，不进训练。

## 测试

```bash
make test         # 全量单测 + 集成测试（需要 ORT .so；缺模型/数据时自动 skip）
make test-short   # 跳过 coco128 大数据集集成测试，只跑纯逻辑单测
make build        # 下载 ORT + 构建服务
```

单元测试覆盖：标签解析、框/掩膜 IoU、AP 的完美/全错/假阳性边界。集成测试 [cmd/validate/main_test.go](cmd/validate/main_test.go) 在 coco128-seg 前 16 张上端到端跑两个模型，断言 mAP@.5 不低于回归下限；缺 `.so`/模型/数据时自动 skip。

## 已知限制

- 输入尺寸固定（启动时指定），不支持动态输入尺寸；输出维度不支持 `-1`
- 可视化标签用 Go 内置点阵字体，仅支持 ASCII（中文类别名显示为方块，JSON 不受影响）
- 单引擎实例共享张量、推理串行；高并发/攒批能力在 `internal/sched` 规划中，尚未实现
