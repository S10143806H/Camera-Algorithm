# Camera-Algorithm

屏幕异常检测算法集：从视频文件或摄像头输入检测**黑屏、白屏、闪屏、花屏、卡顿**。

支持 **Linux / Windows / macOS**，采集后端与设备枚举由 `platform_compat.py` 按平台分派。

| 入口 | 用途 |
|---|---|
| `web_monitor.py` | **Web 版实时监控** —— 局域网浏览器看画面、框选 ROI、调相机参数、看事件 |
| `camera_diag.py` | 本机 GUI 预览 + 摄像头诊断；`--batch` 批量检测视频目录 |
| `analyze_screen_*.py` | 单类异常离线检测（黑/白/闪/花/冻屏） |

多屏台架下每块屏幕**独立编号 S1/S2/S3、独立判定、独立告警**。

---

## 从零搭一套（Ubuntu，10 分钟）

面向"手上有一台 Ubuntu、一个 USB 相机、一台 Android 车机"的人，照着抄就能跑起来。

### 0. 需要的东西

| | 说明 |
|---|---|
| 主机 | Ubuntu 22.04 / 24.04，x86_64；本文在 **Ubuntu 24.04.4 + Python 3.12.3** 上验证 |
| 相机 | 任意 UVC USB 相机即可。本文用的是 `USB2.0 Camera RGB`（`lsusb` 显示 `15aa:1555`），1280x720 MJPG 30fps |
| 被测设备 | Android 设备，开好 USB 调试；投屏与黑屏归因要用 `adb`（没有也能跑，只是少这两个功能） |
| 摆位 | 相机固定，能一次拍全所有待测屏幕；机位定下来后别再动 |

### 1. 装系统依赖

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git \
                    libgl1 libglib2.0-0 \
                    fonts-noto-cjk v4l-utils ffmpeg adb
```

| 包 | 少了会怎样 |
|---|---|
| `libgl1` `libglib2.0-0` | `import cv2` 报 `ImportError: libGL.so.1` |
| `fonts-noto-cjk` | 截图上的中文变方块 |
| `v4l-utils` | 排查相机时用不了 `v4l2-ctl` |
| `ffmpeg` | 只有 `--finalize` 合并分段视频要用 |
| `adb` | 没有设备投屏和黑屏归因 |

### 2. 拿代码、建虚拟环境

```bash
git clone https://github.com/S10143806H/Camera-Algorithm.git
cd Camera-Algorithm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 给权限（这一步最容易漏）

```bash
sudo usermod -aG video "$USER"     # 相机：否则打开 /dev/video* 会失败
sudo usermod -aG plugdev "$USER"   # adb：否则设备显示 no permissions
# 两条都需要重新登录（或重启）才生效
```

重新登录后自检：

```bash
id -nG | grep -o video               # 应输出 video
ls -l /dev/video*                    # 应能看到设备
adb devices                          # 应列出设备且状态为 device
```

> `adb devices` 显示 `no permissions` 时，加一条 udev 规则（`idVendor` 用 `lsusb` 里查到的值）：
> ```bash
> echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="18d1", MODE="0666", GROUP="plugdev"' \
>   | sudo tee /etc/udev/rules.d/51-android.rules
> sudo udevadm control --reload-rules && sudo udevadm trigger
> adb kill-server && adb devices
> ```

### 4. 先确认相机能出画

```bash
python3 camera_diag.py
```

会遍历 index × 后端 × 格式 × 分辨率，打印类似：

```
📋 系统摄像头设备 (1 个):
  📷 USB2.0 Camera RGB: USB2.0 Camer
     Sym: /dev/video0
  🟢 idx=0 V4L2  fourcc=MJPG→MJPG 1280x720@30fps mean=108.6
  ✅ 画面可用: idx=0 V4L2 MJPG 1280x720
```

**记下这个 `idx`**，下一步要用。全是 `⚫`（mean<5）说明没出画，翻最后的[故障定位](#故障定位)表。

### 5. 起 Web 服务

```bash
python3 web_monitor.py --device 0 --screens 3
```

`--device` 填上一步的 idx，`--screens` 填画面里有几块屏。终端会打印访问地址：

```
✅ 相机就绪: idx=0 V4L2 1280x720
📱 设备投屏: 已连接 <机型>，路数跟随相机屏数（当前 3）
🌐 打开浏览器访问:  http://192.168.x.x:8000/
```

局域网内任意 PC 打开这个地址即可，被访问端不用装任何东西。

### 6. 标定屏幕位置

浏览器里二选一：

- 点 **「自动标定」**——启动后等约 20 秒（标定缓存要攒满）再点，成功率最高
- 点 **「框选 ROI」** 手动拖框，一块屏拖一次，最后点「应用」

标定结果会存进 `screen_rois.json`，**重启自动载入，只需做一次**。网页上会给出等效命令行参数：

```
--roi 492,31,294,185;532,231,246,60;388,402,472,296
```

### 7. 验收清单

| 检查项 | 期望 |
|---|---|
| 顶部相机标签 | 显示分辨率与 fps，**不是**红色的"相机掉线" |
| 实时画面 | 每块屏有蓝框 + `S1/S2/S3` 编号 |
| 屏幕状态表 | 各屏 `ok`，暗像素 0% |
| 设备投屏 | 每块屏一格，画面与上方相机看到的对得上 |
| 遮住某块屏 | 该屏转 `BLACK`，几秒后事件列表出现证据截图 |

### 8. 常驻运行

```bash
# 后台跑，日志写文件
setsid nohup python3 -u web_monitor.py --device 0 --screens 3 > web.log 2>&1 &

# 停止（方括号防止 pkill 杀掉自己所在的 shell）
pkill -f "[w]eb_monitor.py"
```

要开机自启就做成 systemd user 服务。仓库里带了模板 `deploy/screen-monitor.service`
（用 `%h` 表示家目录，默认克隆到 `~/Camera-Algorithm`）：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/screen-monitor.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now screen-monitor
loginctl enable-linger "$USER"      # 未登录时也保持运行
```

路径不同就手写（`REPO`/`PY` 换成实际路径）：

```bash
REPO=$HOME/Camera-Algorithm
PY=$REPO/.venv/bin/python3
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/screen-monitor.service <<EOF
[Unit]
Description=Screen anomaly monitor
[Service]
WorkingDirectory=$REPO
ExecStart=$PY -u $REPO/web_monitor.py --screens 3 --port 8000 --types black_screen,screen_distorted,screen_flicking
Restart=always
RestartSec=5
[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now screen-monitor
loginctl enable-linger "$USER"      # 未登录时也保持运行

systemctl --user status screen-monitor    # 看状态
journalctl --user -u screen-monitor -f    # 跟日志
```

### 新手常踩的 5 个坑

1. **加完 `video` 组没重新登录** —— 权限不生效，相机打不开。`id -nG` 里看不到 `video` 就是还没生效。
2. **相机被别的进程占着** —— Web 服务和 `camera_diag.py` 预览不能同时开，相机只能被一个进程持有。`fuser -v /dev/video0` 查是谁。
3. **`--device` 填错** —— USB 重新插拔后 `/dev/videoN` 会换号。服务已能自动找回（约 5 秒），但首次启动仍要填对，用 `camera_diag.py` 查。
4. **自动标定圈成整幅画面** —— 环境太亮时白墙也会被当成屏幕。改用手动框选，或把屏幕内容切到亮画面后重标。
5. **另一台 PC 打不开网页** —— 检查防火墙放行 8000 端口：`sudo ufw allow 8000/tcp`。先用 `curl http://<IP>:8000/api/status` 验证。

---

## 命令速查

### 装环境（一次）

```bash
sudo apt install -y python3-venv ffmpeg libgl1 libglib2.0-0 fonts-noto-cjk v4l-utils
sudo usermod -aG video "$USER"          # 加入 video 组，需重新登录生效
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Web 版实时监控（远程访问，推荐）

```bash
python3 web_monitor.py --device 0 --screens 3      # 启动，默认 0.0.0.0:8000
python3 web_monitor.py --port 8080                 # 换端口
python3 web_monitor.py --device 0 --notify         # 事件推飞书
pkill -f "[w]eb_monitor.py"                        # 停止（方括号防 pkill 自匹配）
curl http://127.0.0.1:8000/api/status              # 健康检查
```

启动后终端打印访问地址，另一台 PC 浏览器直接打开即可框选 ROI、调参数、看事件。

### 本机 GUI 预览

```bash
python3 camera_diag.py                             # 诊断：遍历后端/格式/分辨率找可用画面
python3 camera_diag.py --device 0 --detect --screens 3        # 直接开相机 + 多屏检测
python3 camera_diag.py --detect --roi "433,95,307,182;472,285,248,72;301,467,520,253"
python3 camera_diag.py --video ticket.mp4 --detect  # 回放问题单视频
```

窗口内：`C` 自动标定 · `R` 框选 ROI · `D` 检测开关 · `X` 清除 · `S` 存图 · `Q` **连按两次**退出。

### 批量 / 离线检测（无 GUI，服务器可跑）

```bash
python3 camera_diag.py --batch data_source --step 3 --screens 3   # 目录内全部视频
python3 camera_diag.py --batch one.mp4 --out out/                 # 单个视频
python3 analyze_black_screens.py --root ./data_source --recursive # 黑屏专项
python3 analyze_white_screen.py     --video x.mp4 --out out/      # 白屏
python3 analyze_screen_flicking_v4.py --video x.mp4 --out out/ --finalize  # 闪屏
python3 analyze_screen_distorted.py --video x.mp4 --out out/      # 花屏
python3 analyze_screen_freeze.py    --video x.mp4 --out out/      # 冻屏
```

### 飞书告警

```bash
cp set_feishu_env.sh.example set_feishu_env.sh     # 填入 webhook 后
source set_feishu_env.sh
python3 notify/feishu_notifier.py --test           # 自测
```

### 排查

```bash
ls -l /dev/video*                                  # 设备与权限
v4l2-ctl --list-devices                            # 相机列表
v4l2-ctl -d /dev/video0 --list-formats-ext         # 支持的格式与分辨率
fuser -v /dev/video0                               # 谁占着相机
export QT_QPA_PLATFORM=xcb                         # Wayland 下预览窗口异常时
```

### 常用参数

| 参数 | 适用 | 说明 |
|---|---|---|
| `--device N` | `camera_diag` `web_monitor` | 直接开第 N 个相机，跳过诊断 |
| `--screens N` | `camera_diag` `web_monitor` | 最多识别几块屏幕，默认 3；设 `1` 为单屏行为 |
| `--roi "x,y,w,h;..."` | `camera_diag` `web_monitor` | 固定 ROI，多块用分号分隔 |
| `--detect` | `camera_diag` | 开启黑屏检测叠加（Web 版默认开，用 `--no-detect` 关） |
| `--batch PATH` | `camera_diag` | 目录或单个视频，无 GUI |
| `--step N` | `camera_diag --batch` | 每 N 帧检测一次，加速 |
| `--notify` | `camera_diag` `web_monitor` | 生成事件后推飞书 |
| `--out DIR` | `camera_diag --batch`、`analyze_*` | 输出目录 |
| `--port` / `--host` | `web_monitor` | 监听端口 / 地址，默认 `0.0.0.0:8000` |
| `--quality N` | `web_monitor` | MJPEG 质量 1–100，默认 75；带宽紧张可调低 |
| `--width` / `--height` | `web_monitor` | 固定采集分辨率，默认 `1280x720`。**ROI 是像素坐标，与分辨率绑定，别随意改** |
| `--types` | `web_monitor` | 启用的异常类型，逗号分隔或 `all`。默认只开 `black_screen`，原因见下 |
| `--no-device-screen` | `web_monitor` | 关掉设备投屏（默认自动探测 adb 设备） |
| `--adb-serial S` | `web_monitor` | 多台设备时指定序列号 |
| `--device-displays` | `web_monitor` | 手动指定投屏的 display-id 及顺序（逗号分隔），缺省按 port 自动枚举 |
| `--device-bitrate` | `web_monitor` | 投屏码率，默认 `4M`；分辨率按各屏原始宽高比自动缩放（长边 ≤1280） |
| `--device-stream-mode` | `web_monitor` | `auto`（默认，串台自动降级）或 `screencap`（直接轮询，慢但从头就对） |
| `--finalize` | `analyze_screen_*` | 合并分段视频 + 生成证据图（需 ffmpeg） |
| `--recursive` | `analyze_black_screens` | 递归扫描 `--root` 子目录 |

---

## 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| Python | ≥ 3.8 | 已在 3.12 验证 |
| OpenCV | ≥ 4.8 | `opencv-python`；无 GUI 环境用 `opencv-python-headless` |
| NumPy | ≥ 1.24, < 2 | |
| Pillow | ≥ 10.0 | 仅 `camera/camera_diag.py` 中文叠字用 |
| ffmpeg | 可选 | 仅 `--finalize` 合并分段视频时需要 |
| FastAPI + uvicorn | 可选 | 仅 Web 版 `web_monitor.py` 需要 |

### Linux 安装

```bash
# 系统依赖（GUI 预览 + 视频编解码 + 中文字体 + 摄像头工具）
sudo apt update
sudo apt install -y python3-venv ffmpeg libgl1 libglib2.0-0 \
                    fonts-noto-cjk v4l-utils

# Python 依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

摄像头访问权限（当前用户需在 `video` 组）：

```bash
ls -l /dev/video*
sudo usermod -aG video "$USER"   # 修改后需重新登录生效
```

---

## Web 版实时监控（局域网浏览器访问）

`web_monitor.py` 把采集 + 多屏检测跑成后台服务，另一台 PC 用浏览器直接看和操作，无需安装任何东西。

```bash
pip install fastapi "uvicorn[standard]"
python3 web_monitor.py --device 0 --screens 3        # 默认 0.0.0.0:8000
```

启动后终端打印访问地址，例如 `http://10.0.0.5:8000/`。

网页可完成的操作：

| 功能 | 说明 |
|---|---|
| 实时画面 | MJPEG 流，带 S1/S2/S3 标注，异常屏幕红框高亮 |
| 框选 ROI | 点「框选 ROI」后在画面上拖框，每拖一次加一块，「应用」生效 |
| 自动标定 | 一键调用同一套多屏标定算法 |
| 相机参数 | 亮度/对比度/饱和度/增益/曝光/自动曝光，滑动即生效 |
| **设备投屏** | 相机画面下方并排显示 Android 设备的每块物理屏，**路数跟随相机能检测的屏幕数**；对照"设备以为在显示什么"与"相机实际看到什么" |
| 事件列表 | 每 5 秒刷新，含证据截图、屏幕编号、score；**点击截图看大图**（ESC 关闭，可新标签页打开原图） |
| 面板折叠 | 四个面板（实时画面/屏幕状态/相机参数/事件）点标题栏即可折叠，状态存 localStorage，刷新后保持 |

ROI 会存到 `screen_rois.json`，服务重启后自动载入。

### 实时可检测的异常类型

原本只有黑屏跑在实时链路上，其余四类只能离线跑视频文件。现在五类共用同一套
实时检测组（`live_detectors.py`），**同一块屏可同时命中多种异常**，事件按
`screen_N/<类型>/` 分目录归档，画面上按类型配色。

```bash
python3 web_monitor.py --types black_screen                    # 默认
python3 web_monitor.py --types black_screen,screen_distorted   # 加花屏
python3 web_monitor.py --types all                             # 全开
```

| 类型 | 代号 | 画面配色 | 说明 |
|---|---|---|---|
| `black_screen` | BLACK | 红 | 默认开启 |
| `white_screen` | WHITE | 棕 | 屏幕整片发白 |
| `screen_flicking` | FLICK | 黄 | 亮度反复跳变 |
| `screen_distorted` | GARBLE | 品红 | 花屏 / 马赛克 / 高频噪点 |
| `screen_freeze` | FREEZE | 青 | 画面长时间完全静止 |

> **默认只开黑屏是有原因的。** 黑屏 / 花屏 / 闪屏可放心常开（本仓库的
> `deploy/screen-monitor.service` 即这三类）；**白屏与冻屏建议按需开**。
> 实测在待机的车机台架上全开会刷屏误报：
> `screen_freeze` 在三块屏上 **100% 命中**（待机画面本来就静止），
> `white_screen` 在显示白底页面的那块屏上常驻命中 —— 一分钟就生成 26 个事件、
> 344MB 证据。这两类的语义依赖"这块屏此刻本该在动 / 本该有内容"，
> 需按台架实际用途按需开启。花屏与闪屏没有这个问题。

**花屏判据改过两轮**：

1. **放宽**——原先要求同一网格"梯度达标 **且** 高饱和度达标"，灰度噪点型花屏
   整段漏检（240/240 网格梯度超阈，但只有 3 个网格饱和度达标 → 一帧不报）。
   改为"梯度达标，且（颜色异常 **或** 梯度极强）"，该样本命中 60/60 帧。
2. **加时间判据**——放宽后在实机上误报：一块显示**彩色方块 UI** 的屏被当成马赛克
   花屏，5 分钟刷了 7 个事件（`cells=8~10`、score 0.3~0.45，而真花屏是 cells=240、
   score 1.0）。真花屏的花纹**逐帧随机重排**，静态彩色 UI 不会，于是要求网格
   帧间平均变化量 > `DYN_T`。实测该屏 500 帧误报归零，两段合成花屏命中数不变。

| 样本 | 放宽前 | 放宽后 | 加时间判据后 |
|---|---|---|---|
| 噪点型花屏（真值 60 帧） | 0 | 60 | 60 |
| 彩色型花屏 | 19 | 19 | 19 |
| 正常三屏视频 | 0 | 0 | 0 |
| 实机彩色 UI 屏（500 帧） | — | 19.3% 误报 | **0** |

### 设备投屏

相机画面下方显示 Android 设备自身屏幕，用于对照**设备以为在显示什么**与**相机实际看到什么**。

只依赖 `adb`（`adb exec-out screenrecord` 出 H.264 裸流，用 OpenCV 自带的 FFmpeg 解码），
**不需要装 scrcpy、系统 ffmpeg 或 v4l2loopback，也不需要 root**。

```bash
adb devices                                        # 先确认设备在线
python3 web_monitor.py --device 0 --screens 3      # 自动探测并开启投屏
python3 web_monitor.py --adb-serial A41AEC42       # 多台设备时指定
python3 web_monitor.py --no-device-screen          # 不需要投屏时关掉
```

**多屏**：车机常有中控 / 仪表 / 后排多块屏。服务按 `dumpsys SurfaceFlinger --display-id`
枚举物理屏（排除虚拟录屏屏），按 port 排序编号 D1/D2/D3，
**开几路由相机当前能检测的屏幕数决定** —— 相机侧重新标定 ROI 后，投屏路数自动增减。

每路按该屏原始宽高比缩放（长边 ≤1280），仪表屏这类 1920x480 超宽比例不会被拉变形。

| 行为 | 说明 |
|---|---|
| 首帧延迟 | 约 5 秒（FFmpeg 探测裸流），之后追平实时 |
| 路数同步 | 相机 ROI 数变化后 1–6 秒内跟随 |
| 长时录制 | `--time-limit 0` 去掉 180 秒上限，不再有分段重启的画面停顿 |
| 设备掉线 | 退到"未连接"并每 3 秒重试，插回后自动恢复，不影响相机检测 |
| 看大图 | 点任一路投屏画面可放大查看 |
| 手动重连 | 网页上点「重连投屏」（会重新做一次回落校验） |
| 远程操作 | 打开「控制」后可直接在网页上操作设备，见下 |

**远程操作**：投屏面板的「控制」按钮默认**关**——关闭时点画面是看大图，避免看监控时误触设备。
打开后：

| 操作 | 效果 |
|---|---|
| 单击画面 | `input tap` 点按 |
| 拖动画面 | `input swipe` 滑动（位移超过画面 2% 才算滑动） |
| 键盘打字 | `input text`；回车 / 退格发 `KEYCODE_ENTER` / `KEYCODE_DEL` |
| HOME / BACK / POWER 按钮 | 对应 keyevent，作用于最后点击过的那块屏 |

坐标以归一化 0~1 传给服务端，再乘该屏**原生分辨率** —— 浏览器里画面是缩放显示的，
直传像素会错位。输入用 `input -d <逻辑displayId>` 定向到指定屏；
逻辑 id 与 SurfaceFlinger 的物理 id 不同，从 `DisplayViewport` 解析对应关系，
否则输入会全部落到主屏。

> **多路 screenrecord 串台**：实测部分车机并发跑多个 `screenrecord --display-id` 时各路会互相串台，
> 某一路拿到的其实是另一块屏的画面（并发时甚至输出字节完全相同）。
> 后果不只是显示错乱——一块真正黑屏的副屏会被显示成完好，**直接掩盖故障**。
>
> 服务用 `screencap -d` 做基准做**结构相关度**校验（降采样归一化后求相关，同屏应接近 1，
> 串台会掉到 0.3 以下）。只比亮度抓不住这种情况：两块屏都亮时看不出差别。
> 串台是**运行中才发生**的，所以首帧校验之外每 15 秒复检一次，连续 2 次不过才降级，
> 避免画面变化引起误判。判定串台的那一路自动切到 **screencap 轮询**（约 1 fps，
> 慢但每块屏都读得对），网页上标注 `screencap`，`/api/device_screen` 里带 `corr` 字段。
>
> 用 `--device-stream-mode screencap` 可跳过 screenrecord 直接轮询：帧率低，
> 但从第一帧起就保证对得上，不给串台留窗口。

### HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/stream` | MJPEG 实时流 |
| GET | `/api/snapshot` | 当前帧单张 JPEG（可做健康检查） |
| GET | `/api/status` | 分辨率、fps、各屏状态、当前 ROI |
| GET/POST/DELETE | `/api/rois` | 读取 / 设置 / 清除 ROI |
| POST | `/api/rois/calibrate` | 自动标定多屏 |
| GET/POST | `/api/camera` | 读取 / 设置相机参数 |
| POST | `/api/detect` | 检测开关 |
| GET | `/api/events` | 事件列表 |
| GET | `/api/events/{id}/screenshot` | 事件证据截图 |
| GET | `/device_stream/{i}` | 第 i 块设备屏的 MJPEG 流（i 从 0 起） |
| GET | `/api/device_screen` | 投屏状态（设备、物理屏列表、每路分辨率与 fps） |
| POST | `/api/device_screen/restart` | 重连投屏 |
| POST | `/api/device_screen/{i}/input` | 向第 i 块设备屏注入 tap / swipe / key / text |

停止服务：

```bash
pkill -f "[w]eb_monitor.py"
```

> 方括号写法是为了避免 `pkill` 匹配到自己所在的 shell —— 直接写 `pkill -f web_monitor.py`
> 会因为该字符串出现在自身命令行里而先把发起命令的 shell 杀掉。

> **相机独占**：同一时刻只能有一个进程打开相机。Web 服务运行期间不要再开 `camera_diag.py` 预览，反之亦然。

---

## 快速开始（本机 / 离线）

### 1. 批量检测视频目录（无 GUI，服务器可用）

```bash
python3 camera_diag.py --batch data_source --step 3
```

输出写入 `diag_captures/batch_<日期>/black_screen/`，含 `events_summary.json`。

### 2. 单类异常检测

```bash
python3 analyze_black_screens.py    --video sample.mp4 --out out/
python3 analyze_white_screen.py     --video sample.mp4 --out out/
python3 analyze_screen_flicking.py  --video sample.mp4 --out out/
python3 analyze_screen_flicking_v4.py --video sample.mp4 --out out/ --finalize
python3 analyze_screen_distorted.py --video sample.mp4 --out out/
python3 analyze_screen_freeze.py    --video sample.mp4 --out out/
```

`--finalize` 会调用 ffmpeg 合并分段视频，未安装时报错并给出安装命令。

### 3. 摄像头诊断 + 实时预览（需要图形界面）

```bash
python3 camera_diag.py                       # 遍历 backend/fourcc/分辨率找可用画面
python3 camera_diag.py --device 0 --detect   # 直接开 /dev/video0 并叠加黑屏检测
python3 camera_diag.py --detect --roi 64,0,896,516
```

预览快捷键：

| 键 | 作用 |
|---|---|
| `Q` | 退出 —— **需 3 秒内连按两次**确认（防长时间监控被误触中断） |
| `ESC` | **不退出程序**；仅在框选模式下作"取消框选" |
| `SPACE` | 暂停 / 继续 |
| `D` | 检测开关 |
| `R` | 进入框选模式（不阻塞，画面持续刷新） |
| `C` | 自动标定多屏 ROI，**按下立即生效，无需确认** |
| `X` | 清除 ROI，回到整幅画面检测 |
| `S` | 保存当前帧截图到 `diag_captures/` |

误点窗口关闭按钮不会结束监控 —— 窗口会自动重建，检测循环不中断。

### 4. 多屏台架（画面内同时有多块屏幕）

每块屏幕**独立编号、独立判定、独立事件计数与冷却**，S2 黑屏不会被 S1 的冷却窗口吞掉。
编号按阅读顺序固定：**上→下、左→右**，与框选先后无关，可跨次复现。

```bash
python3 camera_diag.py --device 0 --detect --screens 3          # 自动标定最多 3 块屏
python3 camera_diag.py --device 0 --detect \
  --roi "20,60,300,200;340,40,320,220;680,70,280,190"           # 固定 3 块 ROI（分号分隔）
python3 camera_diag.py --batch data_source --screens 3          # 批量模式同样逐屏统计
```

标定 ROI 的两种方式（预览窗口内）：

| 按键 | 操作 | 适用场景 |
|---|---|---|
| `C` | 自动标定，按下立即生效 | 首选。相机模式下**启动后等约 20 秒**再按，标定缓存填满后最准 |
| `R` | 进入框选模式：鼠标拖框加一块 → 继续拖下一块 → `ENTER` 应用 / `BACKSPACE` 撤销 / `ESC` 取消 | 自动标定漏检、或只想监控其中几块 |
| `X` | 清除 ROI，回到整幅画面检测 | 标定错了想重来 |

> 框选模式**不阻塞**：画面持续刷新、检测继续运行，其它快捷键在退出框选后恢复。

标定后终端会打印可直接复制的参数，例如：

```
🖥️ 自动标定出 3 块屏幕:
   S1: (60, 180, 221, 166)
   S2: (330, 90, 301, 226)
   S3: (700, 160, 211, 161)
可写入 --roi 60,180,221,166;330,90,301,226;700,160,211,161
```

**输出结构**：事件按屏幕分目录归档，`event.json` 含 `screen_no` / `screen_total` / `screen_roi`。

```
diag_captures/live_<日期>/black_screen/
├── screen_1/CAM_<时间>_S1_001/{screenshot.jpg, event.json}
└── screen_3/CAM_<时间>_S3_002/{screenshot.jpg, event.json}
```

> `--screens 1` 回到单屏行为。画面里只有一块屏幕时无需改动，自动标定会只返回一块。

---

## 平台差异

`platform_compat.py` 统一封装，业务脚本不再含平台分支：

| 能力 | Linux | Windows | macOS |
|---|---|---|---|
| 采集后端 | `CAP_V4L2` → `CAP_ANY` | `CAP_MSMF` → `CAP_DSHOW` | `CAP_AVFOUNDATION` → `CAP_ANY` |
| 设备枚举 | 扫描 `/dev/video*`，名称取自 sysfs | PowerShell `Get-PnpDevice` | 按 index 探测 |
| 自动曝光 | `3`=自动 / `1`=手动 | `0.75/1`=自动 / `0.25/0`=手动 | `1`/`0` |
| 中文字体 | Noto CJK / 文泉驿，`fc-match` 兜底 | 微软雅黑 / 黑体 | PingFang / STHeiti |

---

## 黑屏归因（区分正常重启与真故障）

GTMP 跑测试时设备会正常重启、正常灭屏，屏幕本来就该黑。相机只能看到"黑了"，
判不出是不是问题。`device_context.py` 在后台采集设备状态与 logcat，
事件发生时回溯归因，把正常行为和真故障分开。

| 判定 | 含义 | 正常? | 下一步 |
|---|---|---|---|
| `reboot` | uptime 回退 / adb 断连 / `sys.boot_completed=0` | ✅ 正常 | 无需处理 |
| `screen_off` | 该屏电源态不是 ON（灭屏 / DOZE） | ✅ 正常 | 无需处理 |
| `device_black` | 屏电源 ON，但设备端 framebuffer 就是黑的 | ❌ 故障 | 查合成器 / 应用侧，看随事件保存的 logcat |
| `panel_black` | 设备端 framebuffer 正常，相机却看到黑 | ❌ 故障 | 查屏体 / 背光 / 传输链路，**设备侧日志看不出来** |
| `unknown` | adb 不可用 | — | 检查 adb 连接 |

> `device_black` 与 `panel_black` 的区分是关键：前者设备自己就没出画面，后者设备
> 出了画面但没显示出来。两者排查方向完全不同，只看 logcat 无法区分。

采集的信号：`uptime`、`sys.boot_completed`、`ro.boot.bootreason`、各屏
`mScreenState`、该屏 `screencap` 亮度，以及按关键词过滤的滚动 logcat
（ShutdownThread / boot_progress / DisplayPowerController / Watchdog / ANR 等）。

归因结果写进 `event.json` 的 `device_context` 字段，随证据截图一起归档：

```json
{
  "screen_no": 3,
  "device_context": {
    "verdict": "device_black",
    "is_normal": false,
    "reason": "屏电源 ON 但设备端画面就是黑的（screencap 亮度 0.0），合成器/应用侧未出画面",
    "screen_state": "ON",
    "device_mean": 0.0,
    "logs": ["..."]
  },
  "device_state": {"uptime": 67322.0, "boot_completed": true, "boot_reason": "reboot"}
}
```

`GET /api/device_context` 可实时查看设备状态与最近关键日志。用
`--no-device-context` 关闭采集。

### GTMP 关联（可选）

事件里带上当时正在跑的 GTMP 任务，回答"哪个任务、哪个版本、跑到第几步时黑的"。

```bash
export GTMP_TOKEN="<access token>"      # 未设置则自动跳过
python3 web_monitor.py --gtmp-task 96671    # 关联指定任务
python3 web_monitor.py --gtmp-bench 12      # 自动跟踪该台架运行中的任务
```

只做只读查询，不创建/修改/删除任何 GTMP 数据。任务信息写进 `event.json` 的
`gtmp` 字段：任务 ID、名称、状态、进度、版本、台架、创建人。

---

## 飞书告警（可选）

```bash
cp set_feishu_env.sh.example set_feishu_env.sh
# 编辑填入 webhook 后
source set_feishu_env.sh
python3 notify/feishu_notifier.py --test
```

检测时加 `--notify` 即在生成事件后推送卡片。`set_feishu_env.sh` 已被 `.gitignore` 忽略。

---

## 故障定位

| 现象 | 判断 | 操作 |
|---|---|---|
| 所有组合 `mean` 均 < 5（全黑） | 权限或占用 | `ls -l /dev/video*`；`sudo usermod -aG video $USER` 后重新登录 |
| `VIDEOIO ERROR: V4L2: can't open` | 设备被占用或不存在 | `fuser -v /dev/video0`；`v4l2-ctl --list-devices` |
| `ImportError: libGL.so.1` | 缺 GUI 运行库 | `sudo apt install libgl1`，或改装 `opencv-python-headless` |
| `cv2.imshow` 报 `not implemented` | 装的是 headless 版 | 预览需 `opencv-python`；批量检测不受影响 |
| 中文叠字显示为方块 | 缺 CJK 字体 | `sudo apt install fonts-noto-cjk` |
| `C` 标定出整幅画面 `(0,0,W,H)` | 环境过亮，白墙也被判为屏幕 | 已由自适应阈值 + 边框对比修正；仍失败时用 `R` 手动框选并把打印的 `--roi` 固定下来 |
| `C` 漏掉某块屏 | 该屏 bezel 太浅或亮度接近背景 | 用 `R` 手动补框；或把该屏内容切到亮画面后重标定 |
| `--finalize` 报未找到 ffmpeg | 未安装 ffmpeg | `sudo apt install ffmpeg`；无 sudo 时 `pip install imageio-ffmpeg` 后将其二进制软链到 PATH |
| Wayland 下预览窗口异常 | Qt 后端不匹配 | `export QT_QPA_PLATFORM=xcb` 后重跑 |
| 网页画面不动、顶部标签变红 | 相机掉线（USB 重新枚举 `/dev/videoN` 换号） | 服务会自动重新枚举并找回（实测约 5 秒），无需干预；顶部标签会显示重连次数 |
| 重启后突然满屏误报黑屏 | 相机协商到了别的分辨率（实测 1280x720 → 640x480），ROI 整体错位 | 已修：启动固定 `--width/--height`，且 ROI 存盘时记录标定分辨率、载入时按比例换算 |
| 判定在 BLACK / ok 之间快速闪烁 | 自动曝光在追光，画面亮度反复漂移 | 网页相机参数里把 `AutoExp 0/1` 调 0（手动），再固定 `Exposure`；这是误报第一来源 |
| 所有 MJPEG 流掉到 ~1fps | 相机掉线后旧版本会空转拖垮进程 | 已修（掉线自愈）；旧版本需手动重启并用正确的 `--device` |
| `createTrackbar` 报 `NULL window handler` | 窗口句柄名含非 ASCII 字符 | Linux 的 Qt highgui 后端不支持，句柄名保持 ASCII，中文用 `cv2.setWindowTitle` 设置 |
| Web 版报相机打不开 | 相机被 `camera_diag.py` 预览占用 | 同一时刻只能一个进程持有相机，先 `pkill -f camera_diag.py` |
| 网页打得开但画面不动 | 防火墙放行了 80/443 未放行服务端口 | 放行该端口，或换 `--port`；`curl http://<IP>:8000/api/status` 先验证 |

查看相机支持的格式与分辨率：

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```
