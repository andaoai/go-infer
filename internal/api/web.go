// 内嵌前端单页（暗色主题），随二进制一起分发，无需额外静态目录。
package api

import (
	"embed"
	"io/fs"
	"net/http"
)

// staticFS 持有前端单页资源（internal/api/static 目录在编译期内嵌）。
//
//go:embed static
var staticFS embed.FS

// staticHandler 返回前端单页资源。
func staticHandler() http.Handler {
	sub, err := fs.Sub(staticFS, "static")
	if err != nil {
		// 内嵌目录由编译期保证存在，此处出错只可能是构建异常。
		panic(err)
	}
	return http.FileServer(http.FS(sub))
}
