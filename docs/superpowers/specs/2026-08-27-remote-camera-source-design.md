# 远端相机接入设计（网络视频流源）

日期：2026-08-27 · 状态：**已废弃**（2026-08-27 改为在远端主机本地部署）

> **本方案已停用。** 远端主机现在直接跑完整服务（`git clone` + venv + `deploy/screen-monitor.service`，
> 与采集端同一套方法），本地 V4L2 直采，不再经 ustreamer 与 ssh 隧道。
> 改用本地部署的收益：多一层解码/重编码没了；`adb` 与相机同机，黑屏归因能拿到设备端 `screencap`
> 亮度，`panel_black` 与 `device_black` 才分得开。
> 代价：远端主机上会有本项目源码（原方案的核心约束之一被放弃）。
>
> **不要重新启用 `remote-camera-stream.service`** —— ustreamer 会独占 `/dev/video0`，
> 与本地监控服务抢设备。下文仅作历史记录保留。

## 目标

让同一套屏幕异常监控服务接管另一台 Ubuntu host 上的 USB 相机，且**该主机上不出现任何本项目源码**。
源码与全部检测算法只保留在采集端主机。

## 约束

| 约束 | 来源 | 影响 |
| --- | --- | --- |
| 远端主机不得持有本项目源码 | 需求 | 算法必须留在采集端，远端只能跑通用现成软件 |
| 检测阈值不可重新标定 | `JUMP_CELL_T` / `GRAD_T` / `DYN_T` 等均按本地 UVC MJPEG 画质调出 | 传输链路**不得重新编码**，否则压缩 artifact 会移动阈值 |
| 链路吞吐 26 Mbps（实测） | 两站点间 RTT 68ms | 见下方「传输方案实测对比」，实际占用 3.10 Mbps，余量 8 倍 |
| 现有本地实例不得受影响 | 已标定 ROI 与飞书开关在用 | 持久化文件需按实例隔离 |

## 架构

```
远端 Ubuntu host                          采集端 Ubuntu host
┌────────────────────────┐               ┌──────────────────────────────┐
│ USB UVC 相机           │               │ web_monitor.py               │
│   ↓ 原生 MJPEG         │               │  --source http://127.0.0.1   │
│ ustreamer (apt 包)     │               │           :18080/stream      │
│  --format=MJPEG 直通   │               │  --instance remote           │
│  绑 127.0.0.1:8080     │               │  --port 8001                 │
└──────────┬─────────────┘               └──────────────▲───────────────┘
           │                                            │
           └────── ssh -L 18080:127.0.0.1:8080 ─────────┘
                   （加密隧道，公网零开放端口）
```

**关键点：全链路零转码。** UVC 相机原生输出 MJPEG，`ustreamer --format=MJPEG` 只做转发不重编码
（`/state` 接口自报 `encoder type: HW` 可验证）。到达采集端的每一帧与本地直连采集的画质一致，
现有检测阈值直接可用，无需重新标定。

**落到远端主机的东西**：一个 apt 包（`ustreamer`）+ 一个 systemd 单元文件。**没有一行本项目代码。**

## 配置矩阵

| 项 | 远端主机 | 采集端主机 |
| --- | --- | --- |
| 地址 | `10.79.20.48` | `10.78.20.5` |
| 系统 | Ubuntu 22.04.5 LTS | — |
| 相机 | `/dev/video0`，UVC，原生 MJPG，最高 3840x2160 | 本地 USB 相机（原有） |
| 推流 | `ustreamer 4.9`，1280x720@15fps，绑 `127.0.0.1:8080` | — |
| 隧道 | — | `remote-camera-tunnel.service`，本地 `18080` |
| 监控实例 | — | `screen-monitor-remote.service`，`--instance remote`，端口 `8001` |
| 认证 | ssh 公钥（`BatchMode` 免密） | 私钥在采集端，**不入库、不入文档** |

分辨率与帧率的选择依据：720p@15fps，实测占用 **3.10 Mbps**，对 26 Mbps 链路有 8 倍余量。

> 该数字明显低于 720p MJPEG 的常规估算（15–25 Mbps），原因是曝光校正后房间背景全黑，
> JPEG 压缩黑区效率极高。**曝光设置直接决定了带宽占用**，两者不可分开调。

## 相机参数

远端相机为定焦 UVC 模组（`v4l2-ctl --list-ctrls` 中**无任何 focus 控制项**），对焦只能现场旋转镜筒。
曝光则必须软件校正：

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `auto_exposure` | `1`（手动） | 默认的光圈优先会被自发光屏幕带偏 |
| `exposure_time_absolute` | `90` | 实测该模组把曝光量化到离散档，90/100/120 结果相同 |
| `sharpness` | `7` | 相机侧锐化，实测清晰度 +40%，零代价 |

校正效果（屏幕区域，非整幅画面）：

| | 左屏死白 | 右屏死白 |
| --- | --- | --- |
| 默认（光圈优先 + 背光补偿 64） | 9.6% | **67.8%** |
| 校正后 | **0%** | **0%** |

**这是检测层面的问题，不是观感问题。** 花屏检测依赖相邻网格的 Lab 色块跳变；像素一旦被钳到 255，
色块信息从物理上就不存在了，`JUMP_CELL_T` 再怎么调都无效。而且它**静默失效**——画面看着只是"有点亮"，
检测照跑不报错，只是再也测不出花屏。

参数通过单元的 `ExecStartPost` 写入：内核在设备 open 时会把控制项复位成默认，
而 ustreamer 自身没有 exposure 选项，必须等它打开设备之后再写。

> 已知缺口：**代码里没有任何自动曝光标定**。`set_auto_exposure()` 只接在网页的手动开关上，
> 没有任何逻辑去测量画面再反过来调相机。上述数值是手工扫档得出的，换一台相机或换一个现场光照就得重来。

## 传输方案实测对比

在同一现场、同一相机参数下实测两种传输方案：

| 指标 | 方案 B：MJPEG 直通 | 方案 A：H.264 |
| --- | --- | --- |
| 带宽 | 3.10 Mbps | 2.35 Mbps |
| 花屏判据·邻格均跳（3 屏） | 20.6 / 12.0 / 11.6 | 20.7 / 11.8 / 11.6 |
| 远端 CPU | ≈0（直通不转码） | libx264 软编开销 |
| 服务形态 | ustreamer 常驻，支持多客户端 | ffmpeg `?listen=1` **单次监听**，客户端断开即退出 |

**结论：选 B。** H.264 只省 24% 带宽（0.75 Mbps），而链路余量本就有 8 倍；代价是单点监听的脆弱服务
与额外 CPU 开销。

两点值得记录的实测发现：

1. **H.264 在正常画面上没有移动花屏判据**（差异 <2%）。原因是画面本身失焦、缺少高频细节，
   且邻格跳变算的是 24px 网格的平均色，对压缩天然鲁棒。
   **但该测试未覆盖花屏画面** —— 花屏是高熵内容，正是 H.264 码率最吃紧、块效应最重的场景，
   风险在那一侧尚未验证。若日后需切到 A，必须在真实花屏场景下重测邻格均跳。
2. **ffmpeg 4.4.2 的 RTSP listen 模式不可用**：`-rtsp_flags listen` 被 muxer 忽略，
   ffmpeg 退回成客户端并报 `Connection refused`。H.264 只能走 MPEG-TS over TCP。

## 代码改动

全部改动位于采集端。检测算法（`analyze_*.py`）**一行未动**。

| # | 文件 | 改动 | 理由 |
| --- | --- | --- | --- |
| 1 | `camera_diag.py` | 新增 `open_stream(url)` | 与 `open_device(idx)` 并列的网络流入口。强制 `CAP_FFMPEG`（`CAP_ANY` 会选到 GStreamer 并静默失败）；`CAP_PROP_BUFFERSIZE=1` 防解码器堆积导致画面滞后；不套用亮度校验（远端息屏是待检异常，不该判成"打不开"） |
| 2 | `web_monitor.py` | 新增 `--source` / `--instance` | `--source` 给定时跳过本地设备枚举；网络流不做分辨率协商（`set` 对推流端无效） |
| 3 | `web_monitor.py` | `configure_instance(name)` | 按实例名重绑 `ROI_STORE` / `STATE_STORE` / 事件目录。默认实例 `local` 沿用原文件名，现有部署不受影响 |
| 4 | `web_monitor.py` | `_reopen_camera()` 增加网络流分支 | URL 不会变，掉线后重开同一 URL 即可，不走 USB 重新枚举 |
| 5 | `web_monitor.py` | 网络流源自动关闭 adb 投屏 | `DeviceScreenManager` 探测的是**采集端本机** USB 设备，会把本地车机画面投到远端相机页面上，表现为串台且极难排查 |
| 6 | `live_detectors.py` | `_rebuild` 未标定 ROI 时传整幅矩形而非 `None` | **既有缺陷**。检测器 `process()` 首行即 `sx,sy,sw,sh = self.roi`，收到 `None` 会每帧抛 `cannot unpack non-iterable NoneType`。本地实例因始终存在 `screen_rois.json` 从未暴露；新实例首次启动必然触发，现象是"只有黑屏检测正常"（黑屏走 `detect_dark_region` 另一条路径） |

## 服务清单

| 单元 | 主机 | 作用 |
| --- | --- | --- |
| `remote-camera-stream.service` | 远端（system） | ustreamer 推流，开机自启 |
| `remote-camera-tunnel.service` | 采集端（user） | ssh -L 隧道，`Restart=always` |
| `screen-monitor-remote.service` | 采集端（user） | 远端相机监控实例，`Requires=` 隧道 |
| `screen-monitor.service` | 采集端（user） | 本地相机实例（原有，未改动） |

单元文件归档在 `deploy/`。

## 验证方法

```bash
# 1. 远端推流存活与编码方式（应为 encoder type "HW" = 直通未转码）
ssh <remote> 'curl -s http://127.0.0.1:8080/state'

# 2. 隧道吞吐与实际帧率
timeout 10 curl -s -o /tmp/s.bin http://127.0.0.1:18080/stream
stat -c%s /tmp/s.bin                              # ÷10÷1048576 = MB/s
grep -ac 'Content-Type: image/jpeg' /tmp/s.bin    # ÷10 = fps

# 3. 端到端
curl -s http://127.0.0.1:8001/api/status          # fps 字段应接近 15
```

验收结果：端到端 **15.4 fps**、**6.8 Mbps**、检测器零报错。

## 故障定位

| 观察 | 判断 | 操作 |
| --- | --- | --- |
| 页面打不开，`8001` 无响应 | 监控实例未起 | `systemctl --user status screen-monitor-remote` |
| 实例日志"打不开视频流" | 隧道断了 | `systemctl --user restart remote-camera-tunnel` |
| 隧道 active 但画面打不开 | 端口被占，ssh 保持连接却未转发 | 单元已设 `ExitOnForwardFailure=yes`；检查 `ss -lntp | grep 18080` |
| 画面滞后数秒 | 解码缓冲堆积 | 确认 `open_stream` 里 `CAP_PROP_BUFFERSIZE=1` 生效；链路带宽不足时降 `--desired-fps` |
| fps 明显低于 15 | 链路吞吐不够 | 降分辨率或帧率；实在不够再考虑 H.264（需重新标定阈值） |
| 检测每帧报 unpack 错误 | 该实例未标定 ROI 且改动 6 未生效 | 在网页上标定 ROI，或确认 `live_detectors.py` 已修 |
| 远端相机画面全黑 | 隐私挡片或设备被占用 | `ssh <remote> 'fuser -v /dev/video0'` |

## 限制与后续

1. **ROI 需在网页上单独标定。** 新实例的 `screen_rois_remote.json` 初始为空，此时整幅画面（含房间背景）参与判定，
   必然误报 —— 背景的线缆、椅子本身就满足花屏的"互不相干色块"特征。
2. **帧率 15fps 对闪屏检测偏低。** `ScreenFlickingDetector` 的滑窗 `WIN=13` 帧，15fps 下覆盖 0.87 秒；
   高频闪变可能漏检。花屏与黑屏是空间特征，不受影响。
3. **两路相机是两个网页、两个端口。** 合并成单页面多路需要重构全局单例（ROI / 状态 / 事件目录 / 投屏），
   改动面大，另开一轮。
4. **飞书告警共用同一个群。** 两路实例的告警会混在一起，事件标题里的 `source` 字段带实例名前缀可区分。
5. **H.264 已实测并否决**，见「传输方案实测对比」。若日后链路劣化需重新评估，
   切换只需换 `--source` 的 URL，但必须先在**真实花屏场景**下验证邻格均跳未漂移。
6. **失焦无法用软件解决。** 归一化后各分辨率清晰度均为 11–13，上 4K 也救不回来。
   对检测影响有限（黑屏/白屏/闪屏/冻屏几乎不受影响；花屏因网格边长 24px 远大于模糊尺度，
   影响也很小），主要影响**人工复核证据图**时屏上文字不可读。
