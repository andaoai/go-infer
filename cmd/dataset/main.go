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
// 每个引擎对应一个独立 YOLO 数据集（dataset/<engine>/），各自生成 data.yaml，
// 避免不同类别体系的标签互相污染。提升记录写入 dataset/versions/。
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/andaoai/go-infer/internal/appcfg"
	"github.com/andaoai/go-infer/internal/promote"
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
	pool := fs.String("pool", "dataset/pool", "采集池根目录")
	into := fs.String("into", "dataset", "数据集根目录")
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

	res, err := promote.Run(promote.Options{
		PoolDir:     *pool,
		DatasetRoot: *into,
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
		fmt.Printf("data.yaml: %s\n", res.DataYAML)
	}
	if res.Manifest != "" {
		fmt.Printf("版本清单: %s\n", res.Manifest)
	}
}
