//go:build linux

package video

import "syscall"

// sysProcAttr 在 Linux 上设置 Pdeathsig：当本 Go 进程被 kill/退出时，
// 内核会向 ffmpeg 子进程发 SIGKILL，避免服务异常退出后留下孤儿 ffmpeg。
func sysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
}
