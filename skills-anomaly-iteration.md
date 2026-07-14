---
name: screen-anomaly-detection
description: 车机屏幕异常检测算法的标注驱动迭代方法。当用户要为黑屏、白屏、花屏、闪屏、冻屏等屏幕异常开发相机检测算法、按人工标注(contact sheet蓝圈)迭代规则阈值、标定屏幕ROI、或将检测接入事件告警闭环(event gate/截图证据/飞书Webhook)时使用。
---

# 屏幕异常检测算法迭代方法（黑屏 v3 已验证）

## 核心架构（三层，新异常类型照此扩展）

```
① 屏幕ROI标定  calibrate_screen_roi(video)     ← 时序最大亮度，排除车内环境
② 单帧检测     detect(frame, screen_roi) → {abnormal, region{bbox,mean,dark_pct,std,edge_pct}, ...}
③ 事件闭环     连续N帧命中 → event(截图+json) → 冷却 → 飞书告警
```

参考实现：`Camera-Algorithm/analyze_black_screens.py`（算法）、`camera_diag.py`（预览/批量/事件）、`notify/feishu_notifier.py`（告警）。

## 屏幕ROI标定（先界定屏幕位置，防止把车内环境当屏幕）

- 均匀抽 40 帧，累计逐像素最大亮度（屏幕总有点亮时刻，含结尾全屏UI）。
- 最大亮度图 >90 → 形态学 close/open(23) → 最大连通域 bbox（须≥5%画面）。
- 局限：夜间车灯扫过会使 ROI 偏大 → 台架固定机位时标定一次后写死 `--roi x,y,w,h`。
- 相机实时模式用最近 20s 滚动缓存做同样计算。

## 特征库（新异常类型从这里选可分特征）

| 特征 | 计算 | 黑屏 | 白屏 | 花屏 | 闪屏 | 冻屏 |
|---|---|---|---|---|---|---|
| mean | 区域灰度均值 | <50 | >200 | - | 帧间突变 | - |
| dark_pct / bright_pct | (gray<60) / (gray>200) 占比 | ≥85% | ≥85% | - | - | - |
| std | 区域标准差 | 低(平) | 低(平) | 高 | - | - |
| edge_pct | Canny(50,150) 非零占比 | <2.5%(无纹理) | <2.5% | 异常高/块状 | - | - |
| 列/行暗度剖面 | (gray<60).mean(axis=0/1) | 整列≥90%才是pane | 同理用亮 | - | - | - |
| 帧间差 | absdiff 连续帧 | - | - | - | 亮度突变序列 | 差值≈0 持续 |
| 色彩块 | HSV 饱和度/色调块状异常 | - | - | 主特征 | - | - |

关键经验：**边缘密度区分"异常纯色"与"暗色正常内容"**（黑屏 pane edge≈0.1-1%，暗色相机画面 edge≈4%+）；**列/行剖面区分 pane 与形态学连通的假大区域**（异常 pane 整列命中，正常内容的列总有亮像素）。

## 规则分层模式（黑屏 v3 实测阈值，供参考起点）

- P1 深黑 pane：列/行剖面 run（≥90% 暗、长度≥15% 屏宽/高，防状态栏窄条FP）+ mean<50 + dark_pct≥85 + edge<2.5
- P2 泛光异常（玻璃反光/glare 抬高均值）：轮廓候选，面积 15%~85% 屏（防整屏假候选）+ 宽≥40%屏宽 + mean<95 + dark_pct≥50 + edge<5.5
- C 兜底 整屏异常（无点亮屏幕时）：面积/长宽比/均值规则
- 判定 abnormal=bool(best)，候选按 (dark_pct, area) 排序取最优

## 标注驱动迭代流程（每轮 15-30 分钟）

1. **基线全量跑** → 输出红框视频 + 逐帧 CSV + contact sheet(0.5s抽帧) + summary.json
2. **人工标注**：在 contact sheet 上蓝圈圈出真实异常帧 → `*_contact_sheet-labelled.jpg`
3. **诊断脚本**：在标注帧和正常对照帧上打印候选区域全部特征（mean/dark/std/edge/剖面），找出可分特征和间隔（例：GT dark60≥53% vs 正常≤42%）
4. **定阈值**：取分布间隔中点，记下判据依据
5. **抽帧验证**（tune）：每0.5s抽帧对照标注算 MISS/FP，目标 MISS=0
6. **全量重跑 + 其他视频回归**——修 A 必查 B（本次 0202 曾被 v2 规则弄丢主区间）
7. **结果版本化**：`detected_results_vN`，算法留 `_vN_backup.py`，不覆盖旧版

## 常见坑（全部踩过）

- **自动曝光**：屏幕变黑时相机自动提亮 → 阈值漂移。实测前必须手动曝光+固定增益+锁白平衡。
- **__pycache__ 陈旧字节码**：文件 mtime 未变时 Python 用旧缓存 → 改了算法没生效。
- **暗背景连通**：形态学 close 把黑屏 pane 和车内暗环境连成整帧候选 → 必须先界定屏幕 ROI。
- **窄边条 FP**：屏幕黑色边框/状态栏满足暗+平 → 用 run 长度下限(≥15%)排除。
- **位置过滤误删**：v1 的 center_x<0.32w 直接丢左半屏候选 → 位置先验要慎用。
- **反光 glare**：把黑屏均值抬到 80-91 → 需要 P2 这类放宽均值、用剖面/宽度补偿的分支。

## 事件闭环约定

- 门控：连续 3 帧命中 → 事件；10s 冷却；放在检测函数外（event_gate），闪屏/白屏复用。
- 证据：`output/<日期>/<异常类型>/<event_id>/{screenshot.jpg, event.json}`；截图标注 Source/Video Time/Capture Time/Frame/Score。
- event.json 必备字段：event_id, event_type, source, frame_index, source_timestamp_ms, capture_time, score, bbox, screenshot。视频时间与真实检测时间两个都保留，不混用。
- 告警：飞书 Webhook 从环境变量 FEISHU_WEBHOOK 读取（可选 FEISHU_WEBHOOK_SECRET 签名），红色卡片，失败重试2次且不中断检测。

## 新异常类型生成步骤（例：白屏）

1. 复制 detect_dark_region 为 detect_white_region：dark_pct→bright_pct(gray>200)，剖面阈值同构，edge 判据保留（白屏也无纹理）。
2. abnormal_type 写 "white_screen"，事件复用 event_gate（不改）。
3. 收集 3-5 个白屏问题单视频 → 走上面迭代流程 1-7。
4. 验收：样本级+片段级 TP/FP/FN，起止时间误差≤0.5s，必须有可复查红框输出。

## 验证运行方式（沙盒/本机通用）

- 45s 超时环境下：视频分段处理（每段150-330帧，逐段写 records.jsonl + seg mp4，最后 ffmpeg concat + 汇总），单帧接口天然支持。
- 本机全帧率：`py -3.12 camera_diag.py --batch data_source --notify`。
