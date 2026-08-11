//go:build !linux

package video

import "syscall"

// sysProcAttr 在非 Linux 平台不做特殊设置。
func sysProcAttr() *syscall.SysProcAttr {
	return nil
}
