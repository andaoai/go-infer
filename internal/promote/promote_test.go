package promote

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/andaoai/go-infer/internal/format/yolo"
	"github.com/andaoai/go-infer/internal/fsx"
	"github.com/andaoai/go-infer/internal/storage"
	"github.com/andaoai/go-infer/internal/storage/local"
)

// seedPool 在 store 的 pool/<eng>/<date>/ 下种一张图（可选标签）。
func seedPool(t *testing.T, st *local.Storage, pool, eng, date, stem string, withLabel bool) {
	t.Helper()
	ctx := context.Background()
	imgKey := fsx.Join(pool, eng, date, "images", stem+".jpg")
	if err := st.Put(ctx, imgKey, bytes.NewReader([]byte("img"))); err != nil {
		t.Fatal(err)
	}
	if withLabel {
		lblKey := fsx.Join(pool, eng, date, "labels", stem+".txt")
		if err := st.Put(ctx, lblKey, strings.NewReader("0 0.5 0.5 0.2 0.2\n")); err != nil {
			t.Fatal(err)
		}
	}
}

func fixedNow() time.Time { return time.Date(2026, 8, 10, 15, 0, 0, 0, time.Local) }

func newTestOpts(t *testing.T, move bool) (Options, *local.Storage) {
	t.Helper()
	root := t.TempDir()
	st, err := local.New(root)
	if err != nil {
		t.Fatal(err)
	}
	return Options{
		Store: st, Codec: yolo.New(),
		PoolRoot: "pool", DatasetRoot: "dataset",
		ConfigRoot: st.Root(), // 绝对磁盘路径，写进 data.yaml 的 path
		Move:       move, Classes: []string{"person"}, Now: fixedNow,
	}, st
}

func countFiles(infos []storage.Info) int {
	n := 0
	for _, in := range infos {
		if !in.IsDir {
			n++
		}
	}
	return n
}

func TestPromoteMovesAndWritesYAML(t *testing.T) {
	opts, st := newTestOpts(t, true)
	seedPool(t, st, "pool", "yolov8n", "20260810", "143052_ab", true)
	seedPool(t, st, "pool", "yolov8n", "20260810", "143100_cd", false) // hard negative

	res, err := Run(opts)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 2 {
		t.Fatalf("want 2 promoted, got %d", res.Total)
	}
	ctx := context.Background()
	imgs, _ := st.List(ctx, "dataset/yolov8n/images/train")
	if n := countFiles(imgs); n != 2 {
		t.Errorf("目标图片数 = %d, want 2", n)
	}
	lbls, _ := st.List(ctx, "dataset/yolov8n/labels/train")
	if n := countFiles(lbls); n != 2 {
		t.Errorf("目标标签数 = %d, want 2", n)
	}
	if left, _ := st.List(ctx, "pool/yolov8n/20260810"); len(left) != 0 {
		t.Errorf("移动后源应清空，剩余 %d", len(left))
	}
	rc, err := st.Get(ctx, "dataset/yolov8n/data.yaml")
	if err != nil {
		t.Fatalf("data.yaml 未生成: %v", err)
	}
	defer rc.Close()
	buf := new(bytes.Buffer)
	buf.ReadFrom(rc)
	if !strings.Contains(buf.String(), "nc: 1") || !strings.Contains(buf.String(), "person") {
		t.Errorf("data.yaml 内容异常:\n%s", buf.String())
	}
	if _, err := st.Stat(ctx, res.Manifest); err != nil {
		t.Errorf("版本清单未生成: %v", err)
	}
}

func TestPromoteCopyKeepsPool(t *testing.T) {
	opts, st := newTestOpts(t, false)
	seedPool(t, st, "pool", "yolov8n-seg", "20260810", "a", true)

	res, err := Run(opts)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 1 {
		t.Fatalf("want 1, got %d", res.Total)
	}
	if _, err := st.Stat(context.Background(), "pool/yolov8n-seg/20260810/images/a.jpg"); err != nil {
		t.Errorf("复制模式不应删除源文件: %v", err)
	}
}

func TestPromoteFilterByEngineAndDate(t *testing.T) {
	opts, st := newTestOpts(t, true)
	seedPool(t, st, "pool", "yolov8n", "20260810", "a", true)
	seedPool(t, st, "pool", "yolov8n-seg", "20260810", "b", true)
	seedPool(t, st, "pool", "yolov8n", "20260811", "c", true)

	opts.Engine = "yolov8n"
	opts.Date = "20260810"
	res, err := Run(opts)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 1 {
		t.Fatalf("过滤后应只提升 1 个，got %d (%+v)", res.Total, res.Promoted)
	}
}
