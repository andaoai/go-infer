// Package video 用外部 ffmpeg 进程把视频/实时流抽成逐帧 image.Image。
//
// 它是引擎/存储无关的纯解码层：调用方提供一个输入（本地文件、rtsp://、
// http(s):// 的 HLS/mp4、或 v4l2 摄像头），Decoder.Next() 逐张返回已解码
// 的 image.Image。所有编解码工作交给 ffmpeg，本包只负责拼参数、切分 stdout
// 上的 MJPEG 流、jpeg.Decode。不引入任何 Go 侧视频依赖。
package video

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg" // ffmpeg 输出 mjpeg，需要注册 jpeg 解码器
	"io"
	"os/exec"
	"strings"
)

// Config 控制一次抽帧。
type Config struct {
	FFmpeg string // ffmpeg 可执行路径；空则用 "ffmpeg"
	Input  string // 文件路径 / rtsp(s):// / http(s):// / v4l2:/dev/videoN
	FPS    int    // 输出抽帧率；0 表示不额外指定（用源帧率）
	MaxW   int    // 长边等比缩放到此宽度以减推理负载；0 不缩放
	Live   bool   // 实时流：启用低延迟参数；EOF/断线后由调用方决定是否重启
	// Stderr 可选：接收 ffmpeg 日志（探针用于回传真实失败原因）；nil 时丢弃。
	Stderr io.Writer
}

// Decoder 封装一个运行中的 ffmpeg 子进程及其 stdout MJPEG 流。
type Decoder struct {
	cmd *exec.Cmd
	out io.ReadCloser
	r   *bufio.Reader
}

// IsLive 根据输入 scheme 判断是否为实时流（RTSP/RTMP/SDP/V4L2）。
// http(s) 默认按点播/HLS 处理（到 EOF 结束），不自动重连。
func IsLive(input string) bool {
	low := strings.ToLower(input)
	switch {
	case strings.HasPrefix(low, "rtsp://"), strings.HasPrefix(low, "rtsps://"),
		strings.HasPrefix(low, "rtmp://"), strings.HasPrefix(low, "rtmps://"),
		strings.HasPrefix(low, "v4l2:"), strings.HasPrefix(low, "/dev/video"):
		return true
	default:
		return false
	}
}

// New 启动 ffmpeg 并准备抽帧。返回的 Decoder 在用完后必须 Close。
func New(ctx context.Context, cfg Config) (*Decoder, error) {
	ff := cfg.FFmpeg
	if ff == "" {
		ff = "ffmpeg"
	}
	args := buildArgs(cfg)
	cmd := exec.CommandContext(ctx, ff, args...)
	// 进程组/Pdeathsig：父进程（go-infer）退出时内核回收 ffmpeg，不留孤儿。
	cmd.SysProcAttr = sysProcAttr()
	// ffmpeg 日志量级大，默认丢弃；探针场景调用方可通过 Config.Stderr 收走错误原因。
	if cfg.Stderr != nil {
		cmd.Stderr = cfg.Stderr
	} else {
		cmd.Stderr = io.Discard
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("启动 ffmpeg: %w", err)
	}
	return &Decoder{cmd: cmd, out: out, r: bufio.NewReaderSize(out, 1<<20)}, nil
}

func buildArgs(cfg Config) []string {
	args := []string{"-hide_banner", "-loglevel", "error"}
	isV4L2 := strings.HasPrefix(cfg.Input, "v4l2:") || strings.HasPrefix(cfg.Input, "/dev/video")
	if cfg.Live || IsLive(cfg.Input) {
		// 低延迟：走 TCP、缩短探测、关缓冲，避免实时流积压。
		args = append(args,
			"-rtsp_transport", "tcp",
			"-fflags", "nobuffer",
			"-flags", "low_delay",
			"-probesize", "100k",
			"-analyzeduration", "0",
		)
	}
	if isV4L2 {
		dev := strings.TrimPrefix(cfg.Input, "v4l2:")
		args = append(args, "-f", "v4l2", "-i", dev)
	} else {
		args = append(args, "-i", cfg.Input)
	}
	if cfg.FPS > 0 {
		args = append(args, "-r", fmt.Sprintf("%d", cfg.FPS))
	}
	if cfg.MaxW > 0 {
		// -1 让 ffmpeg 按比例算高度，保证偶数（某些编码器要求）。
		args = append(args, "-vf", fmt.Sprintf("scale=%d:-2", cfg.MaxW))
	}
	args = append(args, "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "-")
	return args
}

// jpeg 的 SOI/EOI 标记。
const (
	soi0, soi1 = 0xFF, 0xD8
	eoi0, eoi1 = 0xFF, 0xD9
)

// Next 返回下一帧图像。流正常结束时返回 io.EOF。
func (d *Decoder) Next() (image.Image, error) {
	frame, err := d.NextJPEG()
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(frame))
	if err != nil {
		return nil, fmt.Errorf("jpeg decode: %w", err)
	}
	return img, nil
}

// NextJPEG 返回下一帧的原始 JPEG 字节，不做图像解码。
// 纯播放场景下 ffmpeg 已产出 JPEG，直接把字节转发给浏览器即可，省一次解码+重编码。
func (d *Decoder) NextJPEG() ([]byte, error) {
	return d.nextJPEG()
}

// nextJPEG 从 stdout 切分出一个完整的 JPEG（从 SOI 到配对的 EOI）。
//
// ffmpeg 的 image2pipe 把每帧的完整 JPEG 连续写到 stdout，帧之间没有分隔符，
// 因此按 SOI(FFD8)/EOI(FFD9) 标记定位。缓冲里可能残留上一帧尾部，先丢到下一个 SOI。
func (d *Decoder) nextJPEG() ([]byte, error) {
	if err := d.skipToSOI(); err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	buf.WriteByte(soi0)
	buf.WriteByte(soi1)
	for {
		b, err := d.r.ReadByte()
		if err != nil {
			return nil, err // 包含 io.EOF
		}
		if b == eoi0 {
			// 预读一个字节确认是不是 EOI；FF 后面可能跟其他 marker。
			nb, err := d.r.ReadByte()
			if err != nil {
				return nil, err
			}
			buf.WriteByte(b)
			buf.WriteByte(nb)
			if nb == eoi1 {
				return buf.Bytes(), nil
			}
			continue
		}
		buf.WriteByte(b)
	}
}

// skipToSOI 丢弃字节直到遇到 FFD8。
func (d *Decoder) skipToSOI() error {
	prev := false
	for {
		b, err := d.r.ReadByte()
		if err != nil {
			return err
		}
		if prev && b == soi1 {
			return nil
		}
		prev = b == soi0
	}
}

// Close 终止 ffmpeg 进程并关闭管道。可重复调用。
func (d *Decoder) Close() error {
	var err error
	if d.cmd != nil && d.cmd.Process != nil {
		// Kill 在进程已退出时返回错误，忽略。
		_ = d.cmd.Process.Kill()
		err = d.cmd.Wait()
	}
	if d.out != nil {
		_ = d.out.Close()
	}
	return err
}

// ErrFFmpegMissing 表示系统里找不到 ffmpeg 可执行文件。
var ErrFFmpegMissing = errors.New("未找到 ffmpeg 可执行文件")

// LookPath 查找 ffmpeg：path 非空时直接检查该路径，否则在 PATH 中查找。
// 找不到返回 ErrFFmpegMissing。
func LookPath(path string) (string, error) {
	if path != "" {
		if _, err := exec.LookPath(path); err != nil {
			return "", fmt.Errorf("%w: %s", ErrFFmpegMissing, path)
		}
		return path, nil
	}
	p, err := exec.LookPath("ffmpeg")
	if err != nil {
		return "", ErrFFmpegMissing
	}
	return p, nil
}
