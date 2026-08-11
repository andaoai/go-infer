package video

// 源分类。
const (
	CatDemo    = "demo"    // 播放器测试/演示流（稳定）
	CatTV      = "tv"      // 公开 IPTV / 电视台直播
	CatNature  = "nature"  // 自然风景 / 延时 / 空间站
	CatTraffic = "traffic" // 监控 / 交通摄像头
)

// CategoryOrder 是前端展示分类的顺序与中文名。
var CategoryOrder = []struct {
	ID   string
	Name string
}{
	{CatDemo, "演示流"},
	{CatTV, "电视直播"},
	{CatNature, "自然风光"},
	{CatTraffic, "监控/交通"},
}

// Source 描述一个可在视频 tab 直接观看的视频源。
type Source struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Kind       string `json:"kind"`        // "rtsp" / "hls" / "http"（交给 ffmpeg 的输入类型）
	Category   string `json:"category"`    // Cat* 之一
	URL        string `json:"url"`         // 交给 ffmpeg 的输入
	Note       string `json:"note"`        // 给用户看的说明
	BestEffort bool   `json:"best_effort"` // 公共源可能随时失效
}

// BuiltinSources 是开箱内置的免费公共视频源。
//
// 全部为各厂商/社区长期公开的演示或直播流，标记为 BestEffort：公共直播
// （尤其 IPTV/交通摄像头）可能限流、防盗链或随时下线，连不上时前端提示失败，
// 不影响其他源。要看稳定的监控画面，主路径是"自定义 URL"填自己摄像头的 RTSP，
// 或用 -video-source 预置。分类里的交通/监控类公共 HLS 很少长期可用，故内置
// 以稳定演示流 + 公开电视直播打底。
var BuiltinSources = []Source{
	// ---- 演示流（稳定） ----
	{
		ID: "bbb-rtsp", Name: "Big Buck Bunny（RTSP，Wowza 演示）",
		Kind: "rtsp", Category: CatDemo,
		URL:  "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
		Note: "稳定的公开 RTSP 点播流，用于验证拉流管线",
	},
	{
		ID: "bipbop-hls", Name: "Apple BipBop（HLS 测试流）",
		Kind: "hls", Category: CatDemo, BestEffort: true,
		URL:  "https://devstreaming-cdn.apple.com/videos/streaming/examples/bipbop_4x3/bipbop_4x3_variant.m3u8",
		Note: "Apple 官方 HLS 示例",
	},
	{
		ID: "bipbop-adv", Name: "Apple Advanced HLS（fMP4）",
		Kind: "hls", Category: CatDemo, BestEffort: true,
		URL:  "https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_fmp4/master.m3u8",
		Note: "Apple fMP4 HLS 示例",
	},
	{
		ID: "mux-lowlatency", Name: "Mux 低延迟 HLS 示例",
		Kind: "hls", Category: CatDemo, BestEffort: true,
		URL:  "https://stream.mux.com/v69RSHhFelSm4701snP22dYz2jICy4E4FUyk02rW4gxRM.m3u8",
		Note: "Mux 公共低延迟 HLS 测试流",
	},
	{
		ID: "tears-of-steel", Name: "Tears of Steel（HTTP mp4）",
		Kind: "http", Category: CatDemo, BestEffort: true,
		URL:  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
		Note: "Google 公共示例短片",
	},

	// ---- 电视直播 ----
	{
		ID: "aljazeera-en", Name: "Al Jazeera English（半岛电视台英文）",
		Kind: "hls", Category: CatTV, BestEffort: true,
		URL:  "https://live-hls-web-aje.getaj.net/AJE/01.m3u8",
		Note: "公开国际新闻直播",
	},
	{
		ID: "france24-en", Name: "France 24 English",
		Kind: "hls", Category: CatTV, BestEffort: true,
		URL:  "https://static.france24.com/live/F24_EN_LO_HLS/live_web.m3u8",
		Note: "法国 24 英文直播",
	},
	{
		ID: "france24-fr", Name: "France 24 Français",
		Kind: "hls", Category: CatTV, BestEffort: true,
		URL:  "https://static.france24.com/live/F24_FR_LO_HLS/live_web.m3u8",
		Note: "法国 24 法语直播",
	},
	{
		ID: "dw-en", Name: "DW English（德国之声）",
		Kind: "hls", Category: CatTV, BestEffort: true,
		URL:  "https://dwamdstream102.akamaized.net/hls/live/2015525/dwstream102/index.m3u8",
		Note: "德国之声英文直播",
	},
	{
		ID: "nhk-world", Name: "NHK World Japan",
		Kind: "hls", Category: CatTV, BestEffort: true,
		URL:  "https://nhkwlive-ojp.akamaized.net/hls/live/2003459/nhkwlive-ojp-en/index.m3u8",
		Note: "NHK 世界台英文直播",
	},
	{
		ID: "redbull-tv", Name: "Red Bull TV",
		Kind: "hls", Category: CatTV, BestEffort: true,
		URL:  "https://rbmn-live.akamaized.net/hls/live/590964/BoRB-AT/master.m3u8",
		Note: "极限运动/活动直播",
	},

	// ---- 自然风光 ----
	{
		ID: "nasa-public", Name: "NASA Public Channel",
		Kind: "hls", Category: CatNature, BestEffort: true,
		URL:  "https://ntv1.akamaized.net/hls/live/2014075/NASA-NTV1-HLS/master.m3u8",
		Note: "NASA 公共频道（任务/空间站画面）",
	},
	{
		ID: "nasa-media", Name: "NASA Media Channel",
		Kind: "hls", Category: CatNature, BestEffort: true,
		URL:  "https://ntv1.akamaized.net/hls/live/2014076/NASA-NTV2-HLS/master.m3u8",
		Note: "NASA 媒体频道（新闻/纪录片）",
	},
}

// MergeSources 把内置源与用户通过 -video-source 追加的自定义源合并，自定义源
// 的 ID 若与内置重名则覆盖。返回顺序：内置源在前，自定义源按追加顺序在后。
func MergeSources(extra []Source) []Source {
	if len(extra) == 0 {
		out := make([]Source, len(BuiltinSources))
		copy(out, BuiltinSources)
		return out
	}
	byID := make(map[string]Source, len(BuiltinSources)+len(extra))
	for _, s := range BuiltinSources {
		byID[s.ID] = s
	}
	for _, s := range extra {
		byID[s.ID] = s
	}
	out := make([]Source, 0, len(byID))
	seen := make(map[string]bool, len(byID))
	for _, s := range BuiltinSources {
		if v, ok := byID[s.ID]; ok {
			out = append(out, v)
			seen[s.ID] = true
		}
	}
	for _, s := range extra {
		if !seen[s.ID] {
			out = append(out, s)
		}
	}
	return out
}
