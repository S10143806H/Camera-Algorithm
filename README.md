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
| 相机参数 | 亮度/对比度/饱和度/增益/曝光/对焦/自动曝光，滑动即生效 |
| 事件列表 | 每 5 秒刷新，含证据截图、屏幕编号、score |

ROI 会存到 `screen_rois.json`，服务重启后自动载入。

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
| `createTrackbar` 报 `NULL window handler` | 窗口句柄名含非 ASCII 字符 | Linux 的 Qt highgui 后端不支持，句柄名保持 ASCII，中文用 `cv2.setWindowTitle` 设置 |
| Web 版报相机打不开 | 相机被 `camera_diag.py` 预览占用 | 同一时刻只能一个进程持有相机，先 `pkill -f camera_diag.py` |
| 网页打得开但画面不动 | 防火墙放行了 80/443 未放行服务端口 | 放行该端口，或换 `--port`；`curl http://<IP>:8000/api/status` 先验证 |

查看相机支持的格式与分辨率：

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```
