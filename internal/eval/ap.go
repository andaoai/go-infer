package eval

import (
	"image"
	"sort"
)

// Prediction 是一个模型输出实例，按 ImageID 关联到某张图。
type Prediction struct {
	ImageID        int
	ClassID        int
	Confidence     float32
	X1, Y1, X2, Y2 float32
	// Rings 为分割任务的预测外轮廓（可能多条，对应断开区域；像素坐标）；
	// 框任务为空。多条环在栅格化时取并集。
	Rings [][]image.Point
}

// GroundTruth 是一个标注实例。W/H 为该实例所在图片尺寸，供掩膜栅格化。
type GroundTruth struct {
	ImageID        int
	ClassID        int
	X1, Y1, X2, Y2 float32
	Rings          [][]image.Point
	W, H           int
}

// AP 计算全部类别的平均精度（AP），返回各类 AP 和总体 mAP。
//
// useMask=false 时按框 IoU 匹配；useMask=true 时按多边形掩膜 IoU 匹配。
// 采用 COCO 评测惯例：按类分组，预测按置信度降序，贪心匹配同类中 IoU
// 最高且未被占用的 GT，累积 TP/FP 后用全点插值计算单类 AP。
func AP(preds []Prediction, gts []GroundTruth, iouThresh float64, useMask bool) (perClass map[int]float64, mAP float64) {
	matches := buildMatches(preds, gts, useMask)
	perClass = map[int]float64{}
	var sum float64
	n := 0
	for c, cm := range matches {
		ap := classAP(cm, iouThresh)
		perClass[c] = ap
		if cm.nGT > 0 {
			sum += ap
			n++
		}
	}
	if n > 0 {
		mAP = sum / float64(n)
	}
	return perClass, mAP
}

// match 记录一个预测在按置信度降序贪心匹配后对应的 IoU；
// isTP=false 表示该预测没有匹配到任何 GT（IoU 视为 0，是 FP）。
type match struct {
	conf float32
	iou  float64
}

// classMatches 为一个类的所有预测，按置信度降序记录其与同类 GT 的最佳匹配 IoU。
type classMatches struct {
	nGT    int
	values []match // 已按置信度降序
}

// buildMatches 对每个类做一次贪心匹配，记录每个预测的最佳 GT IoU。
// 掩膜位集在所有类、所有阈值之间只计算一次。
func buildMatches(preds []Prediction, gts []GroundTruth, useMask bool) map[int]*classMatches {
	// 预计算掩膜位集（仅 useMask）。
	var predMasks [][]uint64
	var gtMasks [][]uint64
	if useMask {
		predMasks = make([][]uint64, len(preds))
		for i, p := range preds {
			if len(p.Rings) > 0 {
				// 用第一张图尺寸；同图预测/GT 尺寸一致。
				w, h := imageDimsFor(gts, p.ImageID)
				bits, _, _ := maskBits(p.Rings, w, h)
				predMasks[i] = bits
			}
		}
		gtMasks = make([][]uint64, len(gts))
		for i, g := range gts {
			if len(g.Rings) > 0 {
				bits, _, _ := maskBits(g.Rings, g.W, g.H)
				gtMasks[i] = bits
			}
		}
	}

	// GT 按类 + 图组织。
	gtIdxByClass := map[int]map[int][]int{}
	predIdxByClass := map[int][]int{}
	nGTByClass := map[int]int{}

	for i, g := range gts {
		c := g.ClassID
		if gtIdxByClass[c] == nil {
			gtIdxByClass[c] = map[int][]int{}
		}
		gtIdxByClass[c][g.ImageID] = append(gtIdxByClass[c][g.ImageID], i)
		nGTByClass[c]++
	}
	for i, p := range preds {
		predIdxByClass[p.ClassID] = append(predIdxByClass[p.ClassID], i)
	}

	out := map[int]*classMatches{}
	for c, cp := range predIdxByClass {
		// 预测按置信度降序。
		sort.SliceStable(cp, func(a, b int) bool {
			return preds[cp[a]].Confidence > preds[cp[b]].Confidence
		})
		matched := make([]bool, len(gts))
		cm := &classMatches{nGT: nGTByClass[c]}
		byImg := gtIdxByClass[c]
		for _, pi := range cp {
			p := preds[pi]
			bestIOU := 0.0
			bestGT := -1
			for _, gi := range byImg[p.ImageID] {
				if matched[gi] {
					continue
				}
				var iou float64
				if useMask {
					if predMasks[pi] != nil && gtMasks[gi] != nil {
						iou = bitsetIoU(predMasks[pi], gtMasks[gi])
					}
				} else {
					iou = BoxIoU(
						Box{X1: p.X1, Y1: p.Y1, X2: p.X2, Y2: p.Y2},
						Box{X1: gts[gi].X1, Y1: gts[gi].Y1, X2: gts[gi].X2, Y2: gts[gi].Y2},
					)
				}
				if iou > bestIOU {
					bestIOU = iou
					bestGT = gi
				}
			}
			if bestGT >= 0 {
				matched[bestGT] = true
			}
			cm.values = append(cm.values, match{conf: p.Confidence, iou: bestIOU})
		}
		// 没有该类预测但有 GT 的类也要占位（AP=0）。
		if _, exists := out[c]; !exists {
			out[c] = cm
		}
	}
	// 补全"有 GT 无预测"的类。
	for c, n := range nGTByClass {
		if _, ok := out[c]; !ok {
			out[c] = &classMatches{nGT: n}
		}
	}
	return out
}

// imageDimsFor 返回某张图的尺寸（取该图第一个 GT 的 W/H）。
func imageDimsFor(gts []GroundTruth, imageID int) (w, h int) {
	for _, g := range gts {
		if g.ImageID == imageID {
			return g.W, g.H
		}
	}
	return 0, 0
}

// classAP 给定一次匹配结果和 IoU 阈值，用全点插值计算 AP。
func classAP(cm *classMatches, iouThresh float64) float64 {
	if cm == nil || cm.nGT == 0 {
		return 0
	}
	// values 已按置信度降序。
	nGT := float64(cm.nGT)
	prec := make([]float64, len(cm.values))
	rec := make([]float64, len(cm.values))
	tp := 0
	for i, m := range cm.values {
		if m.iou >= iouThresh {
			tp++
		}
		prec[i] = float64(tp) / float64(i+1)
		rec[i] = float64(tp) / nGT
	}
	if tp == 0 {
		return 0
	}
	// 全点插值：pInterp(r_i) = max_{k>=i} P(r_k)，
	// AP = sum_i (r_i - r_{i-1}) * pInterp(r_i)，r_{-1}=0。
	pInterp := make([]float64, len(prec))
	maxP := 0.0
	for i := len(prec) - 1; i >= 0; i-- {
		if prec[i] > maxP {
			maxP = prec[i]
		}
		pInterp[i] = maxP
	}
	ap := 0.0
	prevR := 0.0
	for i := 0; i < len(rec); i++ {
		ap += (rec[i] - prevR) * pInterp[i]
		prevR = rec[i]
	}
	return ap
}

// MAPOverThresholds 返回 COCO 风格 mAP@[0.50:0.95]（步长 0.05，共 10 个阈值），
// 以及 mAP@0.5。匹配只计算一次，掩膜只栅格化一次，多阈值共享。
func MAPOverThresholds(preds []Prediction, gts []GroundTruth, useMask bool) (map50, map5095 float64) {
	matches := buildMatches(preds, gts, useMask)
	map50 = mAPFromMatches(matches, 0.50)
	var sum float64
	for t := 0.50; t <= 0.95001; t += 0.05 {
		sum += mAPFromMatches(matches, roundT(t))
	}
	map5095 = sum / 10.0
	return map50, map5095
}

// mAPFromMatches 在已有匹配上按阈值计算 mAP（只对 GT 中出现的类求平均）。
func mAPFromMatches(matches map[int]*classMatches, thresh float64) float64 {
	var sum float64
	n := 0
	for c, cm := range matches {
		if cm.nGT == 0 {
			continue
		}
		_ = c
		sum += classAP(cm, thresh)
		n++
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

func roundT(t float64) float64 {
	return float64(int(t*100+0.5)) / 100.0
}
