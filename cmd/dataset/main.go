// Command dataset 管理数据集回流：把采集池的图片与伪标签合并进训练集。
//
// 典型用法：
//
//	# 审核完采集池（用 X-AnyLabeling 等工具在 dataset/pool 上改标签）后：
//	./bin/dataset promote --pool dataset/pool --into dataset \
//	    --classes models/coco.names
//
//	# 只提升某个引擎、某天的批次：
//	./bin/dataset promote --engine yolov8n-seg --date 20260810 ...
//
//	# 复制而非移动（保留采集池原件）：
//	./bin/dataset promote --copy ...
//
// 每个引擎对应一个独立数据集（<into>/<engine>/），按注入的 Codec 生成训练
// 配置（YOLO 写 data.yaml），避免不同类别体系的标签互相污染。提升记录写入
// <into>/versions/。
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/andaoai/go-infer/internal/appcfg"
	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/promote"
	"github.com/andaoai/go-infer/internal/storage/local"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "promote":
		runPromote(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "未知子命令: %s\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `dataset — 数据集回流工具

用法:
  dataset promote [flags]   把采集池合并进训练数据集

promote flags:
  --pool <dir>        采集池根目录（默认 dataset/pool）
  --into <dir>        数据集根目录（默认 dataset）
  --engine <name>     只提升该引擎；不指定则全部
  --date <YYYYMMDD>   只提升该日期；不指定则全部
  --classes <file>    类别名文件，用于生成 data.yaml（默认 models/coco.names）
  --names <a,b,c>     直接传类别名（逗号分隔），优先于 --classes
  --copy              复制而非移动（默认移动，清空采集池）`)
}

func runPromote(args []string) {
	fs := flag.NewFlagSet("promote", flag.ExitOnError)
	poolDir := fs.String("pool", "dataset/pool", "采集池根目录")
	intoDir := fs.String("into", "dataset", "数据集根目录")
	engine := fs.String("engine", "", "只提升该引擎")
	date := fs.String("date", "", "只提升该日期 YYYYMMDD")
	classesFile := fs.String("classes", "models/coco.names", "类别名文件")
	names := fs.String("names", "", "类别名逗号分隔，优先于 --classes")
	copyMode := fs.Bool("copy", false, "复制而非移动")
	_ = fs.Parse(args)

	var classList []string
	var err error
	if strings.TrimSpace(*names) != "" {
		classList = strings.Split(*names, ",")
	} else {
		classList, err = appcfg.LoadClasses("", *classesFile, 0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "读取类别名: %v\n", err)
			os.Exit(1)
		}
	}

	// pool 与 into 映射到同一个本地存储：取两者公共父目录为存储根，
	// 各自相对路径作为存储前缀。
	storeRoot, poolPrefix, dsPrefix, err := resolveStore(*poolDir, *intoDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "解析路径: %v\n", err)
		os.Exit(1)
	}
	st, err := local.New(storeRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "打开存储 %s: %v\n", storeRoot, err)
		os.Exit(1)
	}

	intoAbs, _ := filepath.Abs(*intoDir)
	res, err := promote.Run(promote.Options{
		Store:       st,
		Codec:       yolo.New(),
		PoolRoot:    poolPrefix,
		DatasetRoot: dsPrefix,
		ConfigRoot:  intoAbs,
		Engine:      *engine,
		Date:        *date,
		Classes:     classList,
		Move:        !*copyMode,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "promote 失败: %v\n", err)
		os.Exit(1)
	}
	if res.Total == 0 {
		fmt.Println("没有可提升的样本（采集池为空或筛选无匹配）。")
		return
	}
	fmt.Printf("已提升 %d 个样本：\n", res.Total)
	for _, s := range res.Promoted {
		fmt.Printf("  - %s / %s: %d 个\n", s.Engine, s.Date, s.Files)
	}
	if res.DataYAML != "" {
		fmt.Printf("data.yaml: %s\n", filepath.Join(storeRoot, res.DataYAML))
	}
	if res.Manifest != "" {
		fmt.Printf("版本清单: %s\n", filepath.Join(storeRoot, res.Manifest))
	}
}

// resolveStore 取 pool 与 into 的公共父目录作为本地存储根，返回各自相对前缀。
func resolveStore(pool, into string) (root, poolPrefix, dsPrefix string, err error) {
	poolAbs, err := filepath.Abs(pool)
	if err != nil {
		return "", "", "", err
	}
	intoAbs, err := filepath.Abs(into)
	if err != nil {
		return "", "", "", err
	}
	common := commonParent(poolAbs, intoAbs)
	rel := func(p string) string {
		r, e := filepath.Rel(common, p)
		if e != nil {
			return filepath.ToSlash(p)
		}
		return filepath.ToSlash(r)
	}
	return common, rel(poolAbs), rel(intoAbs), nil
}

// commonParent 返回两个绝对路径最长的公共目录前缀。
func commonParent(a, b string) string {
	as := strings.Split(filepath.ToSlash(a), "/")
	bs := strings.Split(filepath.ToSlash(b), "/")
	n := len(as)
	if len(bs) < n {
		n = len(bs)
	}
	i := 0
	for ; i < n; i++ {
		if as[i] != bs[i] {
			break
		}
	}
	// 绝对路径以 "/" 开头，split 后第一段为空串；用 Join 拼回再补前导分隔符。
	joined := filepath.Join(as[1:i]...)
	if joined == "" {
		return string(filepath.Separator)
	}
	return string(filepath.Separator) + joined
}
