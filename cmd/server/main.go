// Command go-infer 启动推理 HTTP 服务。
//
// 它是引擎无关的：所有推理逻辑由实现了 engine.Engine 的具体引擎提供，
// 本程序只负责加载 ONNX Runtime、按配置构造并注册引擎、启动 HTTP。
// flag 解析见 config.go，引擎/采集器装配辅助见 wire.go，main 只做
// ORT 初始化 → 构造调度器/服务 → 启动带优雅停机的 HTTP。
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/andaoai/go-infer/internal/api"
	"github.com/andaoai/go-infer/internal/appcfg"
	"github.com/andaoai/go-infer/internal/engines/onnxruntime/detect"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/ortenv"
	"github.com/andaoai/go-infer/internal/sched"
	"github.com/andaoai/go-infer/internal/storage"
	"github.com/andaoai/go-infer/internal/storage/local"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	cfg, err := parseConfig(os.Args[1:])
	if err != nil {
		log.Fatalf("解析参数: %v", err)
	}

	schedCfg := sched.Config{
		Workers:    cfg.SchedWorkers,
		QueueDepth: cfg.SchedQueue,
		MaxBatch:   cfg.SchedMaxBatch,
		MaxWait:    cfg.SchedMaxWait,
	}
	log.Printf("调度器: workers=%d queue=%d max-batch=%d max-wait=%s",
		cfg.SchedWorkers, cfg.SchedQueue, cfg.SchedMaxBatch, cfg.SchedMaxWait)

	if lib := ortenv.FindLib(cfg.OrtLib); lib != "" {
		log.Printf("使用 ONNX Runtime: %s", lib)
	}
	if err := ortenv.Init(cfg.OrtLib); err != nil {
		log.Fatalf("初始化 ONNX Runtime: %v", err)
	}
	defer ort.DestroyEnvironment()

	classList, err := appcfg.LoadClasses(cfg.Classes, cfg.ClassesFile, cfg.NumClass)
	if err != nil {
		log.Fatalf("读取类别: %v", err)
	}

	eng, err := detect.New(detect.Config{
		Name:       cfg.Name,
		ModelPath:  cfg.Model,
		InputW:     cfg.ImgSize,
		InputH:     cfg.ImgSize,
		Classes:    classList,
		ConfThresh: float32(cfg.Conf),
		IoUThresh:  float32(cfg.IoU),
		MaxBatch:   cfg.SchedMaxBatch,
	})
	if err != nil {
		log.Fatalf("创建引擎 %s: %v", cfg.Name, err)
	}
	defer eng.Close() // LIFO：调度器先停 goroutine，再销毁底层 ORT session
	detSched := sched.New(eng, schedCfg)
	defer detSched.Close()

	srv := api.NewServer()
	srv.Register(detSched)

	// 分割引擎：模型文件存在才注册，缺失不影响检测服务。
	if !cfg.NoSeg {
		if segEng, err := maybeLoadSeg(cfg.SegModel, cfg.SegName, cfg.ImgSize, classList, cfg.Conf, cfg.IoU, cfg.MaskThr, cfg.SchedMaxBatch); err != nil {
			log.Printf("警告: 分割引擎加载失败，已跳过: %v", err)
		} else if segEng != nil {
			defer segEng.Close()
			segSched := sched.New(segEng, schedCfg)
			defer segSched.Close()
			srv.Register(segSched)
			log.Printf("已注册分割引擎: %s (%s)", segEng.Name(), cfg.SegModel)
		}
	}

	// 推理采集（原图 + YOLO 伪标签），默认关闭。
	// store/codec 在采集器与采集池浏览器之间共享。
	var capStore storage.Storage
	if cfg.CaptureDir != "" {
		st, err := local.New(cfg.CaptureDir)
		if err != nil {
			log.Fatalf("初始化采集存储: %v", err)
		}
		capStore = st
	}
	var capCodec = yolo.New()
	if rec, err := newCaptureRecorder(capStore, capCodec, cfg.CapturePool, cfg.CaptureRate, float32(cfg.CaptureLowConf), cfg.CaptureQuota, cfg.CaptureBuffer); err != nil {
		log.Fatalf("初始化采集器: %v", err)
	} else if rec != nil {
		srv.SetRecorder(recAdapter{rec})
		defer rec.Close()
		srv.SetBrowser(&captureBrowser{st: capStore, codec: capCodec, poolRoot: cfg.CapturePool})
		log.Printf("推理采集已开启: dir=%s pool=%s rate=%.2f low-conf=%.2f quota=%s",
			cfg.CaptureDir, cfg.CapturePool, cfg.CaptureRate, cfg.CaptureLowConf, cfg.CaptureQuota)
	}

	// 网页模型校验（上传 onnx/zip → 临时 session 跑 mAP）。
	maxUpload, err := parseSize(cfg.ValidateMaxUp)
	if err != nil {
		log.Fatalf("解析 -validate-max-upload: %v", err)
	}
	if vs, err := newValidateService(cfg.ValidateDir, maxUpload, classList, cfg.ValidateTests); err != nil {
		log.Fatalf("初始化校验服务: %v", err)
	} else {
		srv.SetValidator(vs)
		srv.SetMaxUploadBytes(vs.maxUploadBytes)
		cleanupStaleScratch(cfg.ValidateDir)
		log.Printf("网页校验已开启: scratch=%s max-upload=%s testsets=%d",
			cfg.ValidateDir, cfg.ValidateMaxUp, len(vs.Testsets()))
	}

	// 网页视频源（ffmpeg 抽帧 → 引擎推理 → MJPEG 实时预览；不导出、不落盘）。
	vidMaxUp, err := parseSize(cfg.VideoMaxUp)
	if err != nil {
		log.Fatalf("解析 -video-max-upload: %v", err)
	}
	if vs, err := newVideoService(cfg.VideoFFmpeg, cfg.VideoScratch, vidMaxUp, cfg.VideoMaxSessions, cfg.VideoSources); err != nil {
		log.Fatalf("初始化视频服务: %v", err)
	} else {
		srv.SetVideo(vs)
		if vs.Enabled() {
			log.Printf("网页视频源已开启: ffmpeg=%s scratch=%s max-sessions=%d max-upload=%s",
				cfg.VideoFFmpeg, cfg.VideoScratch, cfg.VideoMaxSessions, cfg.VideoMaxUp)
		} else {
			log.Printf("网页视频源已禁用：未在 %q 找到 ffmpeg，视频 tab 将提示不可用", cfg.VideoFFmpeg)
		}
	}

	// 信号驱动的优雅停机：SIGINT/SIGTERM 后给在途请求 10s 排空，再让 defer
	// 依次关 recorder、sched、引擎（defer LIFO：sched 先于引擎 Close，正确）。
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	httpSrv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		// ReadTimeout/WriteTimeout 留 0：2GB 校验上传与 MJPEG 长连接不能被整体超时杀掉。
	}
	go func() {
		log.Printf("go-infer 服务启动于 %s | 默认引擎=%s task=%s framework=%s classes=%d",
			cfg.Addr, eng.Name(), eng.Task(), eng.Framework(), len(classList))
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP 服务: %v", err)
		}
	}()

	<-ctx.Done()
	log.Printf("收到停机信号，开始优雅排空（最多 10s）...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		log.Printf("优雅停机超时或出错: %v（在途的 /api/validate 等重操作可能被取消）", err)
	} else {
		log.Printf("HTTP 已排空，开始关闭调度器与引擎...")
	}
}
