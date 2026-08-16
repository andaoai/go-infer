// Package yolohead 把 YOLO 输出头（检测/分割共用的 [B,4+nc(+nm),anchors]
// 或其转置）的逐 anchor 解析逻辑收敛到一处，detect 与 seg 不再各自维护两份
// transposed 分支与类循环。
package yolohead

// Walk 遍历第 slot 张图的输出头，对每个"类别最高分 >= baseConf"的 anchor
// 回调一次：
//   - a 为 anchor 索引、cls 为获胜类、cx/cy/w/h 为框坐标、score 为获胜类分数；
//   - coeffs 为该 anchor 的掩膜系数（长度 nm=attrs-4-nc；detect 模型 nm=0，
//     传入空切片）。coeffs 仅在回调期间有效，seg 需保留时自行 copy——对转置
//     布局它是 buf 的子切片，对非转置布局它由 Walk 内部一块 scratch 填充，
//     从而对 seg 屏蔽两种布局的跨步差异。
//
// 参数：
//   - buf       整批输出缓冲；
//   - slot      图在批中的下标；
//   - stride    每图输出元素数（anchors*attrs）；
//   - anchors   anchor 数；
//   - attrs     每 anchor 属性数（4+nc 或 4+nc+nm）；
//   - nc        类别数；
//   - transposed true 表示布局为 [B,anchors,attrs]，false 为官方 [B,attrs,anchors]；
//   - baseConf  初始最佳分（低于该分的 anchor 不回调，通常传引擎置信度阈值）；
//   - fn        回调。
func Walk(
	buf []float32,
	slot, stride, anchors, attrs, nc int,
	transposed bool,
	baseConf float32,
	fn func(a, cls int, cx, cy, w, h, score float32, coeffs []float32),
) {
	nm := attrs - 4 - nc
	// 非转置布局下掩膜系数按 anchors 跨步，需要一块 scratch 收拢；转置布局
	// 下系数在 attrBase 之后连续，直接切 buf 即可。
	var strided []float32
	if !transposed && nm > 0 {
		strided = make([]float32, nm)
	}
	base0 := slot * stride
	for a := 0; a < anchors; a++ {
		var cx, cy, w, h float32
		var attrBase int
		var coeffs []float32
		if transposed {
			attrBase = base0 + a*attrs
			cx = buf[attrBase]
			cy = buf[attrBase+1]
			w = buf[attrBase+2]
			h = buf[attrBase+3]
			if nm > 0 {
				coeffs = buf[attrBase+4+nc : attrBase+4+nc+nm]
			}
		} else {
			attrBase = base0 + a // 标量在 attrs 维上按 anchors 跨步
			cx = buf[attrBase]
			cy = buf[attrBase+anchors]
			w = buf[attrBase+2*anchors]
			h = buf[attrBase+3*anchors]
			if nm > 0 {
				for k := 0; k < nm; k++ {
					strided[k] = buf[attrBase+(4+nc+k)*anchors]
				}
				coeffs = strided
			}
		}

		// 按类取最大分；best 初值取 baseConf，cls=-1 表示没有任何类超过阈值，
		// 与各引擎历史行为一致（无检测时整 anchor 丢弃）。
		cls, best := -1, baseConf
		for c := 0; c < nc; c++ {
			var s float32
			if transposed {
				s = buf[attrBase+4+c]
			} else {
				s = buf[attrBase+(4+c)*anchors]
			}
			if s > best {
				best, cls = s, c
			}
		}
		if cls < 0 {
			continue
		}
		fn(a, cls, cx, cy, w, h, best, coeffs)
	}
}
