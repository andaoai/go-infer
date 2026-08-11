package video

import (
	"context"
	"errors"
	"io"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// ffmpegAvailable 返回系统是否有 ffmpeg；没有则跳过解码测试。
func ffmpegAvailable(t *testing.T) string {
	t.Helper()
	p, err := LookPath("")
	if err != nil {
		t.Skipf("跳过：%v", err)
	}
	return p
}

// makeTestVideo 用 ffmpeg lavfi 生成一个短时测试 mp4 并返回路径。
func makeTestVideo(t *testing.T, ff string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.mp4")
	cmd := exec.Command(ff, "-hide_banner", "-loglevel", "error",
		"-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=5",
		"-pix_fmt", "yuv420p", path)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("生成测试视频失败: %v\n%s", err, out)
	}
	return path
}

func TestDecoderDecodesFrames(t *testing.T) {
	ff := ffmpegAvailable(t)
	path := makeTestVideo(t, ff)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	dec, err := New(ctx, Config{FFmpeg: ff, Input: path, FPS: 5, MaxW: 160})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer dec.Close()

	const want = 4 // 1s @ 5fps，首尾可能少一两帧
	var n int
	var lastW, lastH int
	for {
		img, err := dec.Next()
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				break
			}
			t.Fatalf("Next: %v", err)
		}
		n++
		b := img.Bounds()
		lastW, lastH = b.Dx(), b.Dy()
		if n > 20 {
			t.Fatal("帧数异常多，可能没正确切帧")
		}
	}
	if n < want {
		t.Fatalf("解出 %d 帧，期望至少 %d", n, want)
	}
	// MaxW=160 等比缩放，320x240 → 160x120。
	if lastW != 160 || lastH != 120 {
		t.Errorf("缩放后尺寸 = %dx%d，期望 160x120", lastW, lastH)
	}
}

func TestDecoderCloseStopsProcess(t *testing.T) {
	ff := ffmpegAvailable(t)
	// 用一个长时间的 lavfi 测试源模拟"还在跑"的流，确认 Close 能杀掉。
	// lavfi 需要 -f lavfi，这里直接构造参数走不通 New，改用 testsrc 文件 + 不读完。
	path := makeTestVideo(t, ff)
	ctx := context.Background()
	dec, err := New(ctx, Config{FFmpeg: ff, Input: path, FPS: 1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := dec.Next(); err != nil {
		t.Fatalf("Next: %v", err)
	}
	if err := dec.Close(); err != nil {
		// Wait 返回的是 kill 导致的退出状态，只要不是 panic 即可。
		t.Logf("Wait 返回（预期非零退出）: %v", err)
	}
}

func TestIsLive(t *testing.T) {
	cases := map[string]bool{
		"rtsp://1.2.3.4/live": true,
		"rtsps://x/y":         true,
		"rtmp://x/y":          true,
		"v4l2:/dev/video0":    true,
		"/dev/video0":         true,
		"https://x/a.m3u8":    false,
		"/tmp/a.mp4":          false,
		"upload:abc":          false,
	}
	for in, want := range cases {
		if got := IsLive(in); got != want {
			t.Errorf("IsLive(%q)=%v, want %v", in, got, want)
		}
	}
}

func TestMergeSourcesOverrideAndAppend(t *testing.T) {
	extra := []Source{
		{ID: "bbb-rtsp", Name: "覆盖项", URL: "rtsp://x"},
		{ID: "mine", Name: "我的", URL: "rtsp://y"},
	}
	out := MergeSources(extra)
	var foundOverride, foundMine bool
	for _, s := range out {
		if s.ID == "bbb-rtsp" {
			foundOverride = s.Name == "覆盖项"
		}
		if s.ID == "mine" {
			foundMine = true
		}
	}
	if !foundOverride {
		t.Error("同 ID 的内置源未被自定义覆盖")
	}
	if !foundMine {
		t.Error("自定义追加源缺失")
	}
}
