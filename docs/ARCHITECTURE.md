# go-infer 架构与数据闭环

本文档描述 go-infer 的**整体架构**和**训练 → 部署 → 推理采集 → 回流重训**的完整数据闭环，作为讨论与迭代的共同语言。

---

## 一、整体分层架构

引擎无关是第一原则：HTTP 层和调度层只依赖 `internal/engine` 的接口，不知道背后是 ONNX Runtime、TensorRT 还是 NCNN。加一个后端/算法只是新增一个 `Engine` 实现并注册。

```mermaid
flowchart TB
    subgraph Client["调用方"]
        C1["HTTP 客户端<br/>curl / 业务系统 / 可视化"]
    end

    subgraph App["cmd（进程入口）"]
        SRV["cmd/server<br/>加载 ORT、构造引擎、启动 HTTP、挂载采集器"]
        VAL["cmd/validate<br/>数据集 mAP 校验门禁"]
        DS["cmd/dataset<br/>采集池回流训练集"]
    end

    subgraph API["internal/api —— HTTP 层（只依赖接口）"]
        ROUTE["路由 /health /engines /predict"]
        REC["Recorder 钩子（可选）"]
        VIS["可视化 drawBoxes/drawInstances"]
    end

    subgraph Core["internal（引擎无关核心）"]
        ENG["engine<br/>Engine/Request/Result/Task"]
        SCHED["sched（规划中）<br/>worker pool / dynamic batching"]
        DS2["dataset<br/>YOLO 标签读写"]
        EVAL["eval<br/>IoU / 栅格化 / COCO AP·mAP"]
        CAP["capture<br/>异步采集 + 配额淘汰"]
        PROM["promote<br/>采集池合并 + data.yaml"]
        PRE["preprocess<br/>letterbox / NCHW / NMS"]
        CFG["appcfg / ortenv<br/>类别名 / ORT 初始化"]
    end

    subgraph Backends["internal/engines（可插拔实现）"]
        DET["onnxruntime/detect<br/>YOLO 检测"]
        SEG["onnxruntime/seg<br/>YOLOv8-seg 分割"]
        FUTURE["tensorrt / ncnn / llamacpp ...<br/>（待加）"]
    end

    ORT["libonnxruntime.so（CGO 动态库）"]

    C1 -->|POST /predict| ROUTE
    ROUTE --> ENG
    ROUTE -.推理后异步入队.-> REC
    REC --> CAP
    SRV --> API
    SRV --> Core
    SRV --> Backends
    VAL --> EVAL
    VAL --> DS2
    VAL --> Backends
    DS --> PROM
    DS --> DS2
    API --> Core
    SCHED -.-> ENG
    DET --> ORT
    SEG --> ORT
    FUTURE -.-> ORT

    classDef planned fill:#f4f4f4,stroke:#999,stroke-dasharray:4 4,color:#888;
    class SCHED,FUTURE planned;
```

- 当前并发模型：引擎内部 `runMu` 串行（共享张量），高并发/攒批能力在规划中的 `internal/sched`。
- `cmd/validate` 与 `cmd/dataset` 是**纯离线工具**，不启动服务。

---

## 二、训练 → 部署 → 采集 → 回流闭环

这是核心业务流。Go 侧只负责**校验、部署、攒数据、出数据集版本**；重训仍在 Python（ultralytics）。

```mermaid
flowchart LR
    PY["Python / ultralytics<br/>训练 + 导出 ONNX"]
    REG["新模型<br/>models/*.onnx"]
    GATE{"cmd/validate<br/>test 集 mAP 达标？"}
    DEPLOY["停服 → 替换模型 → 起服<br/>（中断式部署）"]
    ONLINE["线上服务<br/>POST /predict"]
    CAP["internal/capture<br/>按策略落盘"]
    POOL["dataset/pool/<br/>&lt;engine&gt;/&lt;YYYYMMDD&gt;/"]
    REVIEW["外部工具审核<br/>X-AnyLabeling / Label Studio"]
    PROM["cmd/dataset promote<br/>合并 + 生成 data.yaml + 版本清单"]
    TRAIN["dataset/&lt;engine&gt;/<br/>images,labels/train"]
    TEST["dataset/&lt;engine&gt;/<br/>images,labels/test（不参与训练）"]

    PY --> REG --> GATE
    GATE -- 否 --> PY
    GATE -- 是 --> DEPLOY --> ONLINE
    ONLINE --> CAP --> POOL
    POOL --> REVIEW --> PROM --> TRAIN
    TRAIN --> PY
    TEST -. 只读校验 .-> GATE

    classDef ext fill:#fff6e6,stroke:#e0a000;
    classDef store fill:#eef6ff,stroke:#2a7;
    class PY,REVIEW ext;
    class POOL,TRAIN,TEST store;
```

### 阶段说明

| 阶段 | 责任方 | 产物 / 动作 |
|------|--------|-------------|
| ① 训练导出 | Python | `best.onnx`，配套类别名/输入尺寸 |
| ② 校验门禁 | Go `cmd/validate` | 在固定 `test/` 上算 det box / seg box / seg mask 的 mAP@.5 与 @.5:.95；不达标不允许上线 |
| ③ 部署 | 运维 | 停服、替换 onnx、起服（当前为中断式，不做热加载） |
| ④ 线上推理 | Go `cmd/server` | `/predict` 返回检测/分割结果 |
| ⑤ 采集 | Go `internal/capture` | 按策略异步写原图 + YOLO 伪标签 |
| ⑥ 审核 | 外部工具 | 在采集池上修正标签、补漏检 |
| ⑦ 回流 | Go `cmd/dataset` `promote` | 合并进按引擎隔离的训练集，生成 `data.yaml` 与版本清单 |
| ⑧ 重训 | Python | 拉走 `dataset/<engine>/` 重训，版本迭代 |

---

## 三、采集策略与磁盘布局

### 采集判定（每次推理后）

```mermaid
flowchart TD
    A["一次推理结果"] --> B{"有检测？"}
    B -- 无 --> SAVE["必存（hard negative）"]
    B -- 有 --> C{"最高置信度<br/>&lt; capture-low-conf？"}
    C -- 是 --> SAVE
    C -- 否 --> D{"采样命中<br/>capture-rate？"}
    D -- 是 --> SAVE
    D -- 否 --> SKIP["不存"]
    SAVE --> Q["写入队列（异步）"]
    Q --> ENSURE{"超过 capture-quota？"}
    ENSURE -- 是 --> EVICT["按 mtime 成对删除<br/>最旧 image+label"]
    ENSURE -- 否 --> DONE["完成"]
    EVICT --> DONE
```

- 默认：`rate=0.1`（普通样本采 10%）、`low-conf=0.25`（不确定样本必存）、无检测必存。
- 写入走带缓冲 channel，**不阻塞推理响应**；队列满时丢最旧待写以保响应不卡。
- 配额按采集池**总字节**计（图片+标签都计入），超量从最旧的天目录成对淘汰。

### 目录结构

```
dataset/
├── pool/                              # 采集池（按引擎/日期分组）
│   └── <engine>/
│       └── <YYYYMMDD>/
│           ├── images/HHMMSS_<hex>.jpg
│           └── labels/HHMMSS_<hex>.txt
├── <engine>/                          # 回流后的独立 YOLO 数据集
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml                      # ultralytics 直接可用
└── versions/
    └── <engine>-<YYYYMMDD-HHMMSS>.json  # 每次 promote 的来源/数量清单
```

- **每个引擎一个独立数据集**，避免不同模型类别体系混淆。
- 标签就是 YOLO 原文格式：det `cls cx cy w h`；seg `cls x1 y1 x2 y2 ...`（归一化）。
- `test/` 只给 `cmd/validate` 做发版门禁，绝不进训练。
- `val/test` 目录按需自行补充，`data.yaml` 已预留路径。

---

## 四、一次请求的数据流（含采集）

```mermaid
sequenceDiagram
    participant C as 客户端
    participant H as api /predict
    participant E as Engine 实现
    participant R as capture.Recorder
    participant FS as 磁盘 dataset/pool

    C->>H: POST 图片（multipart/raw）
    H->>H: 解码图片（保留原始字节）
    H->>E: Run(ctx, Request)
    E-->>H: DetectionResult / SegmentationResult
    alt vis=1
        H-->>C: 渲染后的 JPEG（不采集）
    else JSON
        H-->>C: JSON 结果
        opt 采集已开启且命中策略
            H-)R: Record(sample)（异步，不阻塞响应）
            R->>R: 判定策略 / 写标签 / 写图片
            R->>FS: 存 image + YOLO label
            R->>R: 配额检查与淘汰
        end
    end
```

关键点：
- 采集只在 **JSON 响应路径**触发，`vis=1`（可视化查看）不采集，避免把调试点击算进数据。
- 落盘的是**原始上传字节**，不重新编码；标签由 `internal/dataset` 的 Writer 归一化写出。

---

## 五、离线工具调用链

```mermaid
flowchart LR
    subgraph Validate["cmd/validate（发版门禁）"]
        V1["dataset.Load<br/>读 test 图片+标签"]
        V2["Engine.Run<br/>逐张推理"]
        V3["eval.AP / MAPOverThresholds<br/>框/掩膜 mAP"]
        V1 --> V2 --> V3
    end
    subgraph Dataset["cmd/dataset promote（回流）"]
        D1["扫描 pool/&lt;engine&gt;/&lt;date&gt;"]
        D2["移动/复制 到 dataset/&lt;engine&gt;/train"]
        D3["写 data.yaml"]
        D4["写 versions/*.json"]
        D1 --> D2 --> D3 --> D4
    end
```

---

## 六、代码地图

| 目录 | 职责 |
|------|------|
| `cmd/server` | HTTP 服务入口：加载 ORT、注册引擎、挂采集器 |
| `cmd/validate` | 数据集 mAP 校验（发版门禁） |
| `cmd/dataset` | 采集池回流（promote） |
| `internal/engine` | 引擎无关抽象：`Engine`/`Request`/`Result`/`Task` |
| `internal/api` | HTTP/SSE、可视化、`Recorder` 钩子 |
| `internal/dataset` | YOLO 标签读取与写入 |
| `internal/eval` | IoU、多边形栅格化、COCO AP/mAP |
| `internal/capture` | 异步推理采集 + 策略 + 配额淘汰 |
| `internal/promote` | 采集池合并、`data.yaml`、版本清单 |
| `internal/preprocess` | letterbox / NCHW / NMS |
| `internal/appcfg`、`internal/ortenv` | 类别名与 ORT 初始化（多入口共用） |
| `internal/engines/<框架>/<任务>` | 具体引擎实现（当前 ONNX Runtime detect/seg） |

### 关键设计约束

- **引擎无关**：上层只认 `engine.Engine`，新后端实现接口即可，不动 HTTP/采集/校验。
- **数据格式统一**：采集池、训练集、校验集全部是 YOLO 目录结构，`dataset`/`eval` 复用同一套读写。
- **采集默认关闭**：必须显式 `-capture-dir` 才落盘，避免意外数据留存。
- **数据不入库**：`dataset/`、`testdata/`、`third_party/`、模型与 `.so` 全部 gitignored。
- **部署形态**：当前 ORT 引擎依赖动态库（非纯静态）；纯 Go/端侧后端是后续方向。
