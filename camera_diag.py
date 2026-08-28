"""
USB2.0 摄像头诊断 + 黑屏检测预览工具
====================================
1. 摄像头诊断：强制 MJPG 四字节码 + 遍历 backend/fourcc/分辨率，解决"打开了却黑屏"。
2. 参数调节：预览窗口带滑动条（亮度/对比度/饱和度/增益/曝光/自动曝光）。
3. 黑屏检测：--detect 开启，复用 analyze_black_screens.detect_dark_region，
   在预览中用红色高亮框圈出检测到的黑屏区域，橙框为标定的屏幕 ROI。

跨平台: Linux(V4L2) / Windows(MSMF,DSHOW) / macOS(AVFoundation)，
后端选择与设备枚举见 platform_compat.py；环境准备见 README.md。

用法:
  python3 camera_diag.py                              # 诊断 + 纯预览
  python3 camera_diag.py --detect                     # 摄像头预览 + 黑屏检测
  python3 camera_diag.py --video ticket.mp4 --detect  # 手动播放问题单视频 + 检测
  python3 camera_diag.py --device 0 --detect          # 跳过诊断直接开指定相机
  python3 camera_diag.py --detect --roi 64,0,896,516  # 固定屏幕 ROI（推荐台架用）
  python3 camera_diag.py --batch data_source --step 3   # 批量自动检测目录内全部视频

多屏台架（画面里同时有多块屏幕）:
  每块屏幕独立编号 S1/S2/S3（阅读顺序：上→下、左→右），独立判定、
  独立事件计数与冷却，事件按 screen_N/ 分目录归档。
  python3 camera_diag.py --detect --screens 3          # 自动标定最多3块屏
  python3 camera_diag.py --detect --roi 20,60,300,200;340,40,320,220;680,70,280,190

预览快捷键:
  Q 退出(3秒内连按两次确认)   SPACE 暂停/继续   . 单帧步进(视频)   , 单帧步退(视频)
  D 检测开/关   R 进入框选模式(非阻塞)  C 自动标定多屏ROI  X 清除ROI
  S 保存当前帧截图(diag_captures/)
  R 用法: 鼠标拖框添加一块 → 继续拖下一块 → ENTER 应用 / BACKSPACE 撤销 / ESC 取消
          框选期间预览不卡住，画面持续刷新，其它快捷键在退出框选后恢复

黑屏检测重要相机参数（按影响排序）:
  1. AutoExposure=0(手动) —— 屏幕变黑时自动曝光会拉亮整个画面，
     使黑屏帧亮度漂移，是误报/漏报的第一来源，务必关掉后固定 Exposure。
  2. Gain 固定低增益 —— 高增益噪点会抬高暗区灰度和边缘密度，干扰 P1 规则。
  3. 白平衡锁定（部分相机需在厂商工具中关 AWB）。
  4. ROI 固定 —— 台架机位固定后用 C 标定一次，抄下坐标写入 --roi。
  5. 分辨率/FOURCC：MJPG ≥720p；Brightness/Contrast 保持驱动默认居中值。
"""

import argparse
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from platform_compat import (
    capture_backends,
    list_cameras,
    privacy_hint,
    set_auto_exposure,
)

try:
    from analyze_black_screens import (
        detect_dark_region,
        draw_annotation,
        calibrate_screen_roi,
        calibrate_screen_rois,
        rois_from_maxbright,
    )
    HAVE_DETECTOR = True
except ImportError:
    HAVE_DETECTOR = False


# ---------------------------------------------------------------- 检测封装
def run_detect(frame, roi):
    """兼容新旧算法签名。"""
    try:
        return detect_dark_region(frame, screen_roi=roi)
    except TypeError:
        return detect_dark_region(frame)


def run_detect_multi(frame, rois):
    """对每块屏幕独立检测。rois 为空时退化为整幅画面单次检测。

    返回 [(screen_no, roi, result), ...]，screen_no 从 1 起，整幅画面时为 0。
    """
    if not rois:
        return [(0, None, run_detect(frame, None))]
    return [(i + 1, roi, run_detect(frame, roi)) for i, roi in enumerate(rois)]


def parse_rois(text):
    """解析 --roi：单屏 'x,y,w,h'，多屏用分号分隔 'x,y,w,h;x,y,w,h'。"""
    rois = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(v) for v in chunk.split(",")]
        if len(parts) != 4:
            raise ValueError(f"ROI 需为 x,y,w,h 四个整数: {chunk!r}")
        rois.append(tuple(parts))
    return rois


def format_rois(rois):
    """反向格式化，便于直接抄进 --roi 参数。"""
    return ";".join(",".join(str(int(v)) for v in r) for r in rois)


def roi_from_maxbright(max_bright):
    """由逐像素最大亮度图求单块屏幕 bbox（保留旧签名）。"""
    rois = rois_from_maxbright(max_bright, max_screens=1)
    return rois[0] if rois else None


# ---------------------------------------------------------------- 相机参数
# (滑条名, 属性, 滑条最大值, 换算函数)   曝光滑条值 n 实际设置 -n (DSHOW 对数刻度)
CAM_PARAMS = [
    ("Brightness",  cv2.CAP_PROP_BRIGHTNESS,    255, lambda v: v),
    ("Contrast",    cv2.CAP_PROP_CONTRAST,      255, lambda v: v),
    ("Saturation",  cv2.CAP_PROP_SATURATION,    255, lambda v: v),
    ("Gain",        cv2.CAP_PROP_GAIN,          255, lambda v: v),
    ("Exposure -n", cv2.CAP_PROP_EXPOSURE,       13, lambda v: -v),
]


def _attainable(cap, prop, v):
    """v 能否真正写进驱动：写完读回还是 v 才算。"""
    cap.set(prop, v)
    return abs(cap.get(prop) - v) < 0.5


def probe_param_range(cap, prop, limit=100000):
    """探出驱动实际支持的取值范围，探不出返回 None。

    OpenCV 没有查询控制项范围的接口，而各后端的量纲并不一致：同一个
    CAP_PROP_BRIGHTNESS，V4L2 上是 -64..64，DSHOW 上是 0..255。写死一个
    0..255 的后果是滑条上半段**静默失效**——超界值进不到驱动，回读拿到的
    还是旧值，前端照着显示，界面上完全看不出没生效。

    只写一个超大值再读回是不够的：实测同一台相机上 brightness 会被钳到
    边界（读得到真范围），contrast 却直接拒绝写入（读回原值，看着像范围
    只有一个点）。所以改成以"写完读回是否等于写入值"为判据二分找边界，
    两种驱动行为都能覆盖。

    整数控制项的可写区间是连续的，二分成立。步长不为 1 的控制项会探不出
    （读回值对不上），返回 None 由调用方退回写死量程。
    """
    try:
        orig = int(cap.get(prop))
        if not _attainable(cap, prop, orig):    # 只读或当前 inactive
            cap.set(prop, orig)
            return None
        lo, hi = orig, limit                    # 上界：最大的可写值
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _attainable(cap, prop, mid):
                lo = mid
            else:
                hi = mid
        vmax = lo
        lo, hi = -limit, orig                   # 下界：最小的可写值
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if _attainable(cap, prop, mid):
                hi = mid
            else:
                lo = mid
        vmin = hi
        cap.set(prop, orig)                     # 探测真的改到了相机，必须还原
        if vmin < vmax:
            return vmin, vmax
    except Exception:                           # noqa: BLE001  后端不支持该属性
        pass
    return None


# ---- 开机自动标定 ----
# 目标是"检测器能用"，不是"好看"。任何通道被钳到 255，该处色彩信息物理上就没了，
# 花屏检测赖以判定的相邻网格 Lab 色跳算不出来，而且**静默失效**：不报错，只是再
# 也测不出。灰度指标查不出这种丢失——实测某台架 ROI 灰度死白 0.0% 时蓝通道已有
# 20% 钳死，所以全程按逐通道统计。
CALIB_CLIP_MAX = 2.0        # ROI 内任一通道死白(>=250)占比的容忍上限，百分比
CALIB_MEAN_MIN = 50.0       # ROI 灰度均值下限，再暗对比度不够检测器用
CALIB_WB_STEPS = 8          # 白平衡粗扫点数


CALIB_WARMUP_S = 2.0        # 开机预热：自动曝光收敛前的帧是过曝的，测了会误判
CALIB_SETTLE_S = 0.6        # 每次改完控制项等画面稳定的时间


def _grab(cap, settle=CALIB_SETTLE_S):
    """持续丢帧到画面稳定再取一帧。

    按时间丢而不是按帧数：控制项写下去要几帧才反映到画面，而光圈优先模式下
    相机还会为了维持整体亮度反过来调曝光，这个回路要几百毫秒才收敛。丢得
    不够就会拿到过渡态的帧——实测按 6 帧丢时整段标定 2 秒跑完，增益那步测到
    98.8% 死白（其实是开机瞬间的过曝帧），直接把结论带偏。
    """
    end = time.time() + settle
    frame = None
    while time.time() < end:
        ok, f = cap.read()
        if ok:
            frame = f
    if frame is None:
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def _roi_stats(frame, rois):
    """每块 ROI 的 (任一通道死白占比, 灰度均值)。无 ROI 时整幅画面算一块。"""
    boxes = rois or []
    if not boxes:
        h, w = frame.shape[:2]
        boxes = [(0, 0, w, h)]
    out = []
    for x, y, w, h in boxes:
        sub = frame[int(y):int(y + h), int(x):int(x + w)]
        if sub.size == 0:
            continue
        clip = max(float((sub[:, :, k] >= 250).mean()) * 100 for k in range(3))
        out.append((clip, float(sub.mean())))
    return out


def _typical(vals):
    """取中位数而非极值：三块屏里常有一块正在显示纯白页面（那是真内容，不是
    过曝，压不下去），也常有一块是真黑屏。用极值会被这两种屏各自带偏。"""
    if not vals:
        return 0.0
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _color_cast(frame, rois):
    """背景（画面里扣掉各屏 ROI 的部分）的偏色程度，0 为中性。

    只拿背景评白平衡：屏幕是自发光的，内容本身就有色彩倾向，把它算进去等于
    让 WB 去追画面内容。台架背景是机柜/台面这类中性面，才是可用的白参照。
    未钳死、也不太暗的像素才计入，两端的像素已经没有色彩信息。
    """
    m = np.ones(frame.shape[:2], dtype=bool)
    for x, y, w, h in (rois or []):
        m[int(y):int(y + h), int(x):int(x + w)] = False
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m &= (g > 60) & (g < 240)
    if m.sum() < 500:
        return None
    b, gr, r = (float(frame[:, :, k][m].mean()) for k in range(3))
    if gr <= 1:
        return None
    return abs(b / gr - 1) + abs(r / gr - 1)


def calibrate_camera(cap, rois=None, log=print):
    """开机自动标定：先定白平衡，再压增益。返回选中的参数字典。

    分两步而不是联合搜索：增益等比缩放全部通道，不改变通道比例，所以它不影响
    白平衡的评分；反过来白平衡会改亮度，必须先定。顺序反了要来回迭代。
    """
    picked = {}
    _grab(cap, CALIB_WARMUP_S)          # 先让开机后的自动曝光收敛

    # --- 白平衡：关掉自动，扫温度，取背景最中性的一档 ---
    # 不能用"把蓝通道压到不死白"来选 WB：那是拿偏色换死白。实测某模组 WB 拉到
    # 上限确实能把蓝死白从 20% 压到 0，但背景 B/G 掉到 0.838，画面明显偏暖。
    # 死白该由增益去压（等比缩放，不动色彩），WB 只负责还原正确颜色。
    # 自动白平衡开着时色温是只读的，必须先关掉才探得到范围
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    wb_rng = probe_param_range(cap, cv2.CAP_PROP_WB_TEMPERATURE)
    if wb_rng:
        lo, hi = wb_rng
        best = None
        for i in range(CALIB_WB_STEPS):
            t = int(lo + (hi - lo) * i / (CALIB_WB_STEPS - 1))
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, t)
            frame = _grab(cap)
            cast = _color_cast(frame, rois) if frame is not None else None
            if cast is not None and (best is None or cast < best[1]):
                best = (t, cast)
        if best:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, best[0])
            picked["wb_temperature"] = best[0]
            log(f"   白平衡 {best[0]}K（背景偏色 {best[1]:.3f}，0 为中性）")
    if "wb_temperature" not in picked:
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)          # 探不到就还给自动，别停在半途
        log("   白平衡：该后端不支持手动色温，保持自动")

    # --- 增益：在不死白的前提下取最大，画面尽量亮 ---
    # 死白占比随增益单调，可以二分。判据取各屏中位数，容忍一块屏在显示纯白页面。
    g_rng = probe_param_range(cap, cv2.CAP_PROP_GAIN)
    if not g_rng:
        log("   增益：该后端不可调，跳过")
        return picked
    lo, hi = g_rng

    def ok_at(g):
        cap.set(cv2.CAP_PROP_GAIN, g)
        frame = _grab(cap)
        if frame is None:
            return False, 0.0, 0.0
        st = _roi_stats(frame, rois)
        clip = _typical([c for c, _ in st])
        mean = _typical([m for _, m in st])
        return clip <= CALIB_CLIP_MAX, clip, mean

    good, clip_lo, _ = ok_at(lo)
    if not good:
        # 最小增益都压不住：说明是曝光太高，增益已经无能为力
        cap.set(cv2.CAP_PROP_GAIN, lo)
        picked["gain"] = lo
        log(f"   增益 {lo}（已到最小，死白仍有 {clip_lo:.1f}%，需要降曝光）")
        return picked
    while hi - lo > 1:                      # 找满足死白上限的最大增益
        mid = (lo + hi) // 2
        good, _, _ = ok_at(mid)
        if good:
            lo = mid
        else:
            hi = mid
    _, clip, mean = ok_at(lo)
    cap.set(cv2.CAP_PROP_GAIN, lo)
    picked["gain"] = lo
    log(f"   增益 {lo}（各屏死白中位 {clip:.1f}% ≤ {CALIB_CLIP_MAX}%，亮度中位 {mean:.0f}）")
    return picked


def build_param_specs(cap):
    """把 CAM_PARAMS 与实际探到的范围合成运行时参数表。

    探到真实范围时用真实范围、且不再做量纲换算（CAM_PARAMS 里的 conv 是为
    写死量程配的补偿，真范围下反而会错）；探不到就整条退回写死的老行为。
    """
    specs = []
    for name, prop, vmax, conv in CAM_PARAMS:
        rng = probe_param_range(cap, prop)
        if rng:
            lo, hi = rng
            # "Exposure -n" 的 -n 是写死量程时的取负约定，真范围下不再成立
            specs.append({"name": name.replace(" -n", ""), "prop": prop,
                          "min": lo, "max": hi, "conv": None})
        else:
            specs.append({"name": name, "prop": prop,
                          "min": 0, "max": vmax, "conv": conv})
    return specs


def setup_trackbars(win, cap):
    """为相机创建参数滑动条。返回同步函数。"""
    state = {}

    def make_cb(prop, conv, name):
        def cb(v):
            cap.set(prop, conv(v))
            state[name] = v
        return cb

    for name, prop, vmax, conv in CAM_PARAMS:
        cur = cap.get(prop)
        init = int(min(max(cur if conv(1) >= 0 else -cur, 0), vmax))
        cv2.createTrackbar(name, win, init, vmax, make_cb(prop, conv, name))

    def auto_exp_cb(v):
        # 取值随后端而异（DSHOW/MSMF/V4L2），交由 platform_compat 分派
        set_auto_exposure(cap, bool(v))
        print(f"  AutoExposure -> {'自动' if v else '手动(建议)'}")

    cv2.createTrackbar("AutoExp 0/1", win, 1, 1, auto_exp_cb)


# ---------------------------------------------------------------- 设备枚举
def get_camera_symlinks():
    """列出系统摄像头设备（Windows: PowerShell / Linux: /dev/video* / macOS: 探测）。"""
    return list_cameras()


def open_device(idx):
    """跳过完整诊断，按常用组合快速打开指定相机。"""
    for be_code, be_name in capture_backends():
        cap = cv2.VideoCapture(idx, be_code)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(30):
            cap.read()
        ret, frame = cap.read()
        if ret and frame is not None and frame.mean() > 5:
            # 带上设备名：USB 重新枚举后 index 会变，靠名字才能找回同一台相机
            name = next((c["name"] for c in list_cameras()
                         if str(idx) in str(c.get("instance_id", ""))), "")
            return cap, {"idx": idx, "backend": be_name, "fourcc": "MJPG",
                         "name": name,
                         "w": int(cap.get(3)), "h": int(cap.get(4)),
                         "fps": cap.get(cv2.CAP_PROP_FPS), "mean": frame.mean()}
        cap.release()
    return None


def open_stream(url):
    """打开网络视频流（MJPEG over HTTP / RTSP），用于远端相机。

    与 open_device 的三点不同：
      1. 分辨率与帧率由推流端决定，这里 set 无效，只能读回实际值；
      2. 不做设备名匹配，掉线后按同一个 URL 重连即可；
      3. 强制 CAP_FFMPEG —— CAP_ANY 在装了 GStreamer 的机器上会优先选它，
         而 GStreamer 拿到裸 URL 而非 pipeline 时会静默失败，表现为
         "isOpened() 为真但永远读不到帧"。
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    # MJPEG over HTTP 的帧会在解码器里排队：缓冲一大就等于看几秒前的画面，
    # 检测出的异常与飞书告警也跟着延迟。压到 1 帧，宁可丢帧也要实时。
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # noqa: BLE001  某些后端不支持该属性，忽略即可
        pass
    for _ in range(5):                      # 丢掉握手阶段的残帧
        cap.read()
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None
    # 这里不套用 open_device 的 mean() > 5 亮度校验：远端屏幕本来就可能全黑
    # （待机/息屏），那正是要检测的异常，不该在打开阶段就判成"打不开"。
    return cap, {"idx": url, "url": url, "backend": "FFMPEG", "fourcc": "MJPG",
                 "name": url,
                 "w": int(cap.get(3)), "h": int(cap.get(4)),
                 "fps": cap.get(cv2.CAP_PROP_FPS), "mean": float(frame.mean())}


def diagnose_all():
    """遍历 index × backend × fourcc × 分辨率组合，找可用画面。"""
    print("=" * 60)
    print("  🔍 摄像头诊断 — 遍历所有组合")
    print("=" * 60)
    devices = get_camera_symlinks()
    print(f"\n📋 系统摄像头设备 ({len(devices)} 个):")
    for d in devices:
        print(f"  📷 {d['name']}\n     ID:  {d['instance_id']}\n     Sym: {d['symlink'][:80]}")

    backends = capture_backends()
    fourccs = [("MJPG", "MJPG"), ("YUY2", "YUY2"), (None, "默认")]
    resolutions = [(None, None), (640, 480), (1280, 720), (1920, 1080)]

    for w0, h0 in resolutions:
        for idx in range(4):
            for be_code, be_name in backends:
                for fc_str, fc_label in fourccs:
                    cap = cv2.VideoCapture(idx, be_code)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    if fc_str:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fc_str))
                    if w0:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w0)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h0)
                    actual_fc = int(cap.get(cv2.CAP_PROP_FOURCC))
                    actual_fc_str = "".join(chr((actual_fc >> 8 * i) & 0xFF) for i in range(4))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    for _ in range(30):
                        cap.read()
                    means = []
                    for _ in range(10):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            means.append(frame.mean())
                    avg = sum(means) / len(means) if means else -1
                    print(f"  {'🟢' if avg > 5 else '⚫'} idx={idx} {be_name:5s} "
                          f"fourcc={fc_label:4s}→{actual_fc_str:4s} {w}x{h}@{fps:.0f}fps mean={avg:.1f}")
                    if avg > 5:
                        return cap, {"idx": idx, "backend": be_name, "fourcc": actual_fc_str,
                                     "w": w, "h": h, "fps": fps, "mean": avg}
                    cap.release()
    return None


# ---------------------------------------------------------------- 预览主循环
def preview_loop(cap, info, args):
    is_video = args.video is not None
    detect_on = args.detect and HAVE_DETECTOR
    if args.detect and not HAVE_DETECTOR:
        print("  ⚠️ 未找到 analyze_black_screens.py，检测功能不可用")

    max_screens = max(1, getattr(args, "screens", 3) or 3)
    screen_rois = []
    if args.roi:
        screen_rois = parse_rois(args.roi)
        print(f"  🖥️ 使用固定屏幕ROI ({len(screen_rois)} 块): {format_rois(screen_rois)}")
    elif is_video and HAVE_DETECTOR:
        print(f"  🖥️ 视频模式自动标定屏幕ROI (最多 {max_screens} 块) ...")
        screen_rois = calibrate_screen_rois(args.video, max_screens=max_screens)
        for i, r in enumerate(screen_rois, 1):
            print(f"     S{i}: {r}")
        if not screen_rois:
            print("     标定失败，按整幅画面检测")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 窗口句柄名必须为纯 ASCII：Linux 的 Qt highgui 后端用非 ASCII 名建窗后，
    # createTrackbar / selectROI 反查会失败（NULL window handler）。
    # 中文标题改用 setWindowTitle 设置，两端都正常显示。
    win = "camera_diag_video" if is_video else "camera_diag_camera"
    title = ("视频回放" if is_video else f"USB Camera [{info.get('backend','')}]") + \
            " | Q退出 SPACE暂停 D检测 R框选多屏ROI C自动标定 X清除 S截图"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(win, title)
    if not is_video:
        setup_trackbars(win, cap)

    # 手动框选：自己在主循环里做，不用 cv2.selectROIs。
    # selectROIs 是阻塞调用，进去后主循环停摆、其它快捷键全失效，
    # 且它把 c 键定义为"取消当前框"，会吃掉 C（自动标定），造成"按 C 没反应"。
    sel = {"on": False, "drag": False, "p0": None, "p1": None, "boxes": []}

    def on_mouse(event, x, y, flags, _param):
        if not sel["on"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            sel["drag"] = True
            sel["p0"] = sel["p1"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and sel["drag"]:
            sel["p1"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and sel["drag"]:
            sel["drag"] = False
            sel["p1"] = (x, y)
            (x0, y0), (x1, y1) = sel["p0"], sel["p1"]
            box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            if box[2] > 10 and box[3] > 10 and len(sel["boxes"]) < max_screens:
                sel["boxes"].append(box)
                print(f"  ➕ 已框选第 {len(sel['boxes'])}/{max_screens} 块: {box}"
                      f"   (ENTER 应用 / BACKSPACE 撤销 / ESC 取消)")
            sel["p0"] = sel["p1"] = None

    cv2.setMouseCallback(win, on_mouse)

    calib_buf = deque(maxlen=40)      # 相机模式滚动标定缓存
    last_calib_sample = 0.0
    save_dir = Path(__file__).resolve().parent / "diag_captures"
    paused = False
    # 防误退：ESC 不再退出，Q 需要在 QUIT_CONFIRM_S 秒内连按两次
    QUIT_CONFIRM_S = 3.0
    quit_armed_until = 0.0
    step = 0
    frame = None
    result = None

    # 事件门控：连续3帧命中 -> 事件 -> 10s冷却
    notifier = None
    if getattr(args, "notify", False):
        try:
            from notify.feishu_notifier import send_event
            notifier = send_event
            print("  📨 飞书告警: 开 (FEISHU_WEBHOOK)")
        except Exception as e:
            print(f"  ⚠️ 飞书告警不可用: {e}")
    event_seq = 0
    EVIDENCE_N = max(3, getattr(args, "evidence_n", 10) or 10)
    EVIDENCE_MIN = 3
    EVIDENCE_GAP = 0.4        # 证据帧最小间隔(秒)，去掉重复相似帧
    # 每块屏幕独立计数：S2 黑屏不应被 S1 的冷却窗口吞掉
    screen_state = {}

    def state_of(no):
        return screen_state.setdefault(
            no, {"streak": 0, "cooldown_until": 0.0, "evidence": [], "last_evi_t": -1e9})

    event_root = save_dir / f"live_{datetime.now().strftime('%Y%m%d')}" / "black_screen"
    source_name = Path(args.video).name if is_video else f"camera_{info.get('idx', '?')}"
    results = []              # [(screen_no, roi, result), ...]

    while True:
        advance = (not paused) or step != 0
        if advance:
            if step < 0 and is_video:
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
            ret, new_frame = cap.read()
            step = 0
            if not ret:
                if is_video:
                    paused = True          # 播完停在末帧
                    if frame is None:
                        break
                else:
                    print("  ❌ 相机断流"); break
            else:
                frame = new_frame

        if frame is None:
            continue

        # 相机模式滚动收集标定样本（每0.5s一帧）
        if not is_video and time.time() - last_calib_sample > 0.5:
            calib_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            last_calib_sample = time.time()

        # ---- 检测（逐屏独立） ----
        if detect_on and advance:
            results = run_detect_multi(frame, screen_rois)
            now_wc = datetime.now()
            vt = (cap.get(cv2.CAP_PROP_POS_FRAMES) / fps) if is_video else None
            fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if is_video else None
            ts_text = (_fmt_ts(vt) if is_video
                       else now_wc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            cur_t = vt if is_video else time.time()
            for no, roi, result in results:
                st = state_of(no)
                if result["abnormal"]:
                    st["streak"] += 1
                    if (time.time() >= st["cooldown_until"]
                            and cur_t - st["last_evi_t"] >= EVIDENCE_GAP):
                        st["evidence"].append((frame.copy(), result, ts_text, vt, fi, now_wc))
                        st["last_evi_t"] = cur_t
                        if len(st["evidence"]) >= EVIDENCE_N:
                            event_seq += 1
                            threading.Thread(
                                target=emit_merged_event,
                                args=(st["evidence"], source_name, event_root, event_seq),
                                kwargs={"notifier": notifier, "screen_no": no,
                                        "screen_roi": roi, "screen_total": len(results)},
                                daemon=True).start()
                            st["evidence"] = []
                            st["cooldown_until"] = time.time() + 10.0
                else:
                    if (len(st["evidence"]) >= EVIDENCE_MIN
                            and time.time() >= st["cooldown_until"]):
                        event_seq += 1
                        threading.Thread(
                            target=emit_merged_event,
                            args=(st["evidence"], source_name, event_root, event_seq),
                            kwargs={"notifier": notifier, "screen_no": no,
                                    "screen_roi": roi, "screen_total": len(results)},
                            daemon=True).start()
                        st["cooldown_until"] = time.time() + 10.0
                    st["evidence"] = []
                    st["streak"] = 0

        if detect_on and results:
            ts_disp = (_fmt_ts(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps) if is_video
                       else datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            display = annotate_live_multi(frame, results, ts_disp)
        else:
            display = frame.copy()
            draw_screen_boxes(display, screen_rois)

        # ---- 框选模式叠加层 ----
        if sel["on"]:
            for i, (bx, by, bw, bh) in enumerate(sel["boxes"], 1):
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                cv2.putText(display, f"S{i}", (bx + 5, by + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            if sel["drag"] and sel["p0"] and sel["p1"]:
                cv2.rectangle(display, sel["p0"], sel["p1"], (0, 255, 255), 2)
            _put_text_clamped(display,
                              f"SELECT MODE {len(sel['boxes'])}/{max_screens}  "
                              f"drag=add  ENTER=apply  BACKSPACE=undo  ESC=cancel",
                              12, 68, 0.62, (0, 255, 255))

        # ---- 状态栏 ----
        gray_mean = frame.mean()
        if is_video:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            status = f"frame {pos}/{total}  t={pos/fps:6.2f}s"
        else:
            status = f"{info.get('w','?')}x{info.get('h','?')} {info.get('backend','')}"
        status += f"  mean={gray_mean:.0f}  detect={'ON' if detect_on else 'off'}"
        status += f"  screens={len(screen_rois) if screen_rois else 'full-frame'}"
        if detect_on and results:
            bad = [f"S{no}" for no, _, r in results if r["abnormal"] and no]
            status += f"  ALERT={','.join(bad)}" if bad else "  ALERT=-"
        if paused:
            status += "  [PAUSED]"
        cv2.putText(display, status, (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if quit_armed_until > time.time():
            _put_text_clamped(display, "QUIT?  press Q again to confirm  |  any other key = cancel",
                              12, display.shape[0] // 2, 0.85, (0, 0, 255), 3)

        cv2.imshow(win, display)

        # 误点窗口关闭按钮不应结束监控：重建窗口(连同滑条与鼠标回调)继续跑
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                print("  ⚠️ 预览窗口被关闭，已重新打开（退出请连按两次 Q）")
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.setWindowTitle(win, title)
                if not is_video:
                    setup_trackbars(win, cap)
                cv2.setMouseCallback(win, on_mouse)
        except cv2.error:
            pass

        # ---- 按键 ----
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        # 框选模式独占按键：此时 ESC 是"取消框选"而非退出程序
        if sel["on"]:
            if key == 13:                       # ENTER 应用
                if sel["boxes"]:
                    screen_rois = order_rois(sel["boxes"])[:max_screens]
                    screen_state.clear(); results = []
                    for i, r in enumerate(screen_rois, 1):
                        print(f"     S{i}: {r}")
                    print(f"  可写入 --roi {format_rois(screen_rois)}")
                else:
                    print("  未框选任何区域，ROI 保持不变")
                sel.update(on=False, drag=False, boxes=[], p0=None, p1=None)
            elif key == 27:                     # ESC 取消
                sel.update(on=False, drag=False, boxes=[], p0=None, p1=None)
                print("  已退出框选模式，ROI 保持不变")
            elif key == 8:                      # BACKSPACE 撤销
                if sel["boxes"]:
                    print(f"  ➖ 撤销第 {len(sel['boxes'])} 块")
                    sel["boxes"].pop()
            continue

        # 任意其它按键都取消待确认的退出
        if quit_armed_until and key != 255 and key not in (ord('q'), ord('Q')):
            quit_armed_until = 0.0
            print("  已取消退出")

        if key in (ord('q'), ord('Q')):
            if time.time() < quit_armed_until:
                break
            quit_armed_until = time.time() + QUIT_CONFIRM_S
            print(f"  ⚠️ 确认退出？{QUIT_CONFIRM_S:.0f} 秒内再按一次 Q 退出，按其它任意键取消")
        elif key == 27:
            # ESC 最容易误触，不再绑定退出
            print("  ESC 不退出程序；退出请在 3 秒内连按两次 Q")
        elif key == ord(' '):
            paused = not paused
        elif key == ord('.'):
            paused = True; step = 1
        elif key == ord(','):
            paused = True; step = -1
        elif key in (ord('d'), ord('D')):
            if HAVE_DETECTOR:
                detect_on = not detect_on
                print(f"  检测 -> {'开' if detect_on else '关'}")
        elif key in (ord('r'), ord('R')):
            sel.update(on=True, drag=False, boxes=[], p0=None, p1=None)
            print(f"  🖥️ 进入框选模式（最多 {max_screens} 块）：鼠标拖框添加一块 → "
                  f"继续拖下一块 → ENTER 应用 / BACKSPACE 撤销 / ESC 取消。"
                  f"预览不会卡住，画面持续刷新。")
        elif key in (ord('c'), ord('C')):
            if is_video and HAVE_DETECTOR:
                screen_rois = calibrate_screen_rois(args.video, max_screens=max_screens)
            elif calib_buf:
                mb = calib_buf[0].copy()
                for g in calib_buf:
                    mb = np.maximum(mb, g)
                screen_rois = rois_from_maxbright(mb, max_screens=max_screens)
            screen_state.clear(); results = []
            if screen_rois:
                print(f"  🖥️ 自动标定出 {len(screen_rois)} 块屏幕:")
                for i, r in enumerate(screen_rois, 1):
                    print(f"     S{i}: {r}")
                print(f"  可写入 --roi {format_rois(screen_rois)}")
            else:
                print("  🖥️ 自动标定失败：屏幕未点亮或亮区过小，改用 R 手动框选")
        elif key in (ord('x'), ord('X')):
            screen_rois = []
            screen_state.clear(); results = []
            print("  ROI 已清除，回到整幅画面检测")
        elif key in (ord('s'), ord('S')):
            save_dir.mkdir(exist_ok=True)
            p = save_dir / f"cap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            cv2.imwrite(str(p), display)
            print(f"  💾 已保存 {p}")

    cap.release()
    cv2.destroyAllWindows()





def order_rois(rois):
    """按阅读顺序排序，使屏幕编号 S1/S2/S3 与框选先后无关、可复现。"""
    if HAVE_DETECTOR:
        from analyze_black_screens import _reading_order
        return _reading_order(list(rois))
    return sorted(rois, key=lambda r: (r[1], r[0]))


def _put_text_clamped(img, text, x, y, scale, color, thick=2):
    """在图上写字并把起点夹回画面内，避免长标签被右/上边缘截断。"""
    (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(min(max(12, x), max(12, img.shape[1] - tw - 12)))
    y = int(min(max(th_ + 8, y), img.shape[0] - 8))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _put_text_segments(img, segments, x, y, scale, thick=2, gap=18):
    """一行里逐段用不同颜色写字。segments: [(文本, BGR), ...]

    抬头 "S1:BLACK  S2:ok  S3:GARBLE" 里每块屏的颜色要跟它的框一致，
    整行一个颜色就分不出哪块是哪种异常。超宽时换行，避免右侧被截断。
    """
    max_x = img.shape[1] - 12
    line_h = int(cv2.getTextSize("Sg", cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][1] * 1.8)
    cx, cy = int(x), int(y)
    for text, color in segments:
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if cx > x and cx + tw > max_x:          # 放不下就换行
            cx, cy = int(x), cy + line_h
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)
        cx += tw + gap


def draw_screen_boxes(img, numbered_rois, active=None, tags=None, colors=None):
    """画每块屏幕的标定框 + 编号。

    numbered_rois: [(screen_no, roi), ...] —— 编号由调用方给定，
    不能按列表下标重排，否则单屏事件图会把 S3 画成 S1。
    active: 异常屏幕编号集合。
    tags: {屏号: 简写}，异常时附在编号后；缺省不写死"BLACK"，
          否则花屏等其它类型的事件图上会叠出"S1 BLACK"的错误标签。
    colors: {屏号: BGR}，按命中的异常类型给屏幕框上色；缺省退回红色。
            一眼要能分清是黑屏还是花屏，全画成红框就得去读文字。
    """
    active = active or set()
    tags = tags or {}
    colors = colors or {}
    for no, roi in numbered_rois:
        rx, ry, rw, rh = roi
        bad = no in active
        color = (colors.get(no) or (0, 0, 255)) if bad else (255, 160, 0)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), color, 3 if bad else 2)
        label = f"S{no}" + (f" {tags.get(no, '')}".rstrip() if bad else "")
        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lx = int(min(rx, img.shape[1] - tw - 12))
        ly = ry + th_ + 6 if ry < th_ + 8 else ry - 6
        cv2.rectangle(img, (lx, ly - th_ - 4), (lx + tw + 8, ly + 4), color, -1)
        cv2.putText(img, label, (lx + 4, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _res_parts(result):
    """把两种结果形状统一成 (异常, bbox, 短标签, 颜色)。

    黑屏检测返回 {abnormal, region:{bbox, dark_pct}}；anomaly_core 系检测器
    经 live_detectors 适配后返回 {abnormal, bbox, score, info, label, color}。
    """
    if not result or not result.get("abnormal"):
        return False, None, "", (0, 0, 255), ""
    region = result.get("region")
    short = result.get("short") or ""
    if region and region.get("bbox"):
        return (True, region["bbox"],
                f"dark={region.get('dark_pct', 0):.0f}%",
                tuple(result.get("color") or (0, 0, 255)), short or "BLACK")
    return (True, result.get("bbox"), result.get("info") or "",
            tuple(result.get("color") or (0, 0, 255)), short or "ABNORMAL")


def annotate_live_multi(frame, results, ts_text, summary=True):
    """多屏标注：每块屏独立编号与判定，异常处按类型配色高亮，正常屏橙框。

    results: [(screen_no, roi, result), ...]；screen_no==0 表示整幅画面模式。
    result 可以是黑屏形状，也可以是 live_detectors 适配后的通用形状。
    summary=False 供批量/事件图使用（那边自己写抬头，避免文字叠字）。
    """
    annotated = frame.copy()
    parsed = [(no, roi) + _res_parts(res) for no, roi, res in results]

    overlay = None
    for no, roi, abn, bbox, label, color, short in parsed:
        if not (abn and bbox):
            continue
        x, y, w_, h_ = bbox
        if overlay is None:
            overlay = annotated.copy()
        cv2.rectangle(overlay, (x, y), (x + w_, y + h_), color, -1)
    if overlay is not None:
        annotated = cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0)

    bad, bad_label, bad_color = set(), {}, {}
    for no, roi, abn, bbox, label, color, short in parsed:
        if not abn:
            continue
        bad.add(no)
        bad_label.setdefault(no, []).append(short)
        bad_color.setdefault(no, color)     # 一屏多类型时用先命中的那种颜色
        if not bbox:
            continue
        x, y, w_, h_ = bbox
        cv2.rectangle(annotated, (x, y), (x + w_, y + h_), color, 5)
        # bbox 与 ROI 基本重合时，屏幕框上的短代号已经说明类型，
        # 再画一次细标签只会叠字，这里只在异常区明显小于整屏时才画
        if label and (not roi or abs(x - roi[0]) > 8 or abs(y - roi[1]) > 8):
            _put_text_clamped(annotated, f"{short} {label}", x, y - 12, 0.6, color)

    numbered = []
    seen = set()
    for no, roi, *_ in parsed:
        if roi and no not in seen:
            seen.add(no)
            numbered.append((no, roi))
    if numbered:
        tags = {no: "/".join(dict.fromkeys(v)) for no, v in bad_label.items()}
        draw_screen_boxes(annotated, numbered, active=bad, tags=tags,
                          colors=bad_color)
    if summary:
        if numbered:
            # 抬头逐屏上色：同一帧上黑屏与花屏并存时，颜色比读文字快
            segs = [(f"S{no}:{'/'.join(dict.fromkeys(bad_label[no])) if no in bad else 'ok'}",
                     bad_color.get(no, (0, 180, 0)) if no in bad else (0, 180, 0))
                    for no, _ in numbered]
            _put_text_segments(annotated, segs, 12, 34, 0.8)
        else:
            _put_text_clamped(annotated, "ABNORMAL" if bad else "normal/dim screen",
                              12, 34, 0.8, (0, 0, 255) if bad else (0, 180, 0))
    _put_text_clamped(annotated, ts_text, 12, annotated.shape[0] - 40, 0.7, (0, 215, 255))
    return annotated


def annotate_live(frame, result, ts_text):
    """单屏标注（保留旧签名，供合并证据图逐帧复用）。"""
    return annotate_live_multi(frame, [(0, result.get("screen_roi"), result)], ts_text)


def emit_merged_event(evidence, source_name, out_root, seq, notifier=None,
                      screen_no=0, screen_roi=None, screen_total=1,
                      event_type="black_screen", label="BLACK SCREEN"):
    """把缓存的多帧证据合并成一次完整事件：拼图 + event.json + 飞书(带图)。

    evidence: [(frame, result, ts_text, video_time_s, frame_index, wallclock), ...]
    screen_no: 屏幕编号(1 起)，0 表示整幅画面模式；事件按屏幕分目录归档。
    """
    import json as _json
    now = datetime.now()
    tag = f"_S{screen_no}" if screen_no else ""
    eid = f"CAM_{now.strftime('%Y%m%d_%H%M%S')}{tag}_{seq:03d}"
    ev_dir = (out_root / (f"screen_{screen_no}" if screen_no else "full_frame")
              / event_type / eid)
    ev_dir.mkdir(parents=True, exist_ok=True)

    # 逐帧标注 -> 缩略 -> 拼图 (5列)
    thumbs = []
    for frame, result, ts_text, vt, fi, wc in evidence:
        # 带上真实屏幕编号，否则缩略图里会统一标成 S0
        ann = annotate_live_multi(frame, [(screen_no, screen_roi, result)], ts_text)
        thumbs.append(cv2.resize(ann, (384, 216), interpolation=cv2.INTER_AREA))
    cols = 5 if len(thumbs) <= 10 else 8
    tw, th_ = (384, 216) if cols == 5 else (240, 135)
    thumbs = [cv2.resize(t, (tw, th_), interpolation=cv2.INTER_AREA) for t in thumbs]
    rows = (len(thumbs) + cols - 1) // cols
    header_h = 96
    sheet = np.full((header_h + rows * th_, cols * tw, 3), 255, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[header_h + r * th_: header_h + (r + 1) * th_, c * tw:(c + 1) * tw] = t

    # 分数来源随类型而异：黑屏用暗像素占比，其余检测器自带 score
    def _score_of(res):
        if isinstance(res, dict):
            if res.get("region"):
                return min(1.0, float(res["region"].get("dark_pct", 0.0)) / 100.0)
            if res.get("score") is not None:
                return min(1.0, float(res["score"]))
        return 0.0

    best = max(evidence, key=lambda e: _score_of(e[1]))
    score = round(_score_of(best[1]), 3)
    t0, t1 = evidence[0], evidence[-1]
    span = (f"video {_fmt_ts(t0[3])} ~ {_fmt_ts(t1[3])}" if t0[3] is not None
            else f"{t0[5].strftime('%H:%M:%S.%f')[:-3]} ~ {t1[5].strftime('%H:%M:%S.%f')[:-3]}")
    scr = f"SCREEN {screen_no}/{screen_total} - " if screen_no else ""
    head_lines = [f"{scr}{label} - MERGED EVIDENCE ({len(evidence)} frames)",
                  f"Source: {source_name[:60]}   Span: {span}   Score: {score}"]
    for li, txt in enumerate(head_lines):
        cv2.putText(sheet, txt, (14, 36 + 34 * li), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 0, 255) if li == 0 else (60, 60, 60), 2, cv2.LINE_AA)
    shot = ev_dir / "screenshot.jpg"
    cv2.imwrite(str(shot), sheet)

    event = {
        "event_id": eid,
        "event_type": event_type,
        "event_label": label,
        "source": source_name,
        "screen_no": screen_no or None,
        "screen_total": screen_total,
        "screen_roi": list(screen_roi) if screen_roi else None,
        "frame_count": len(evidence),
        "frame_index": t0[4],
        "frame_index_end": t1[4],
        "source_timestamp_ms": int(round(t0[3] * 1000)) if t0[3] is not None else None,
        "source_timestamp_end_ms": int(round(t1[3] * 1000)) if t1[3] is not None else None,
        "capture_time": t0[5].astimezone().isoformat(timespec="milliseconds"),
        "capture_time_end": t1[5].astimezone().isoformat(timespec="milliseconds"),
        "score": score,
        "bbox": (best[1].get("region") or {}).get("bbox") or best[1].get("bbox"),
        "screenshot": str(shot),
    }
    (ev_dir / "event.json").write_text(_json.dumps(event, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"  🚨 {'S' + str(screen_no) + ' ' if screen_no else ''}{event_type.upper()} event "
          f"({len(evidence)} frames merged) span={span} score={score} -> {eid}")
    if notifier:
        try:
            notifier(event, image_path=str(shot))
            print("  📨 feishu notification sent (merged evidence)")
        except TypeError:
            notifier(event)
            print("  📨 feishu notification sent")
        except Exception as e:
            print(f"  ⚠️ 飞书告警失败: {e}")
    return event


def emit_live_event(frame, result, source_name, out_root, seq,
                    video_time_s=None, frame_index=None, notifier=None):
    """实时/回放模式生成一次黑屏事件：标注截图 + event.json + 可选飞书告警。"""
    import json as _json
    now = datetime.now()
    eid = f"CAM_{now.strftime('%Y%m%d_%H%M%S')}_{seq:03d}"
    ev_dir = out_root / eid
    ev_dir.mkdir(parents=True, exist_ok=True)
    ann = draw_annotation(frame, result, video_time_s or 0.0)
    lines = ["BLACK SCREEN",
             f"Source: {source_name[:44]}",
             f"Video Time: {_fmt_ts(video_time_s) if video_time_s is not None else '-'}",
             f"Capture Time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
             f"Frame: {frame_index if frame_index is not None else '-'}",
             f"Score: {min(1.0, result['region']['dark_pct']/100.0):.3f}"]
    for li, txt in enumerate(lines):
        cv2.putText(ann, txt, (12, 60 + 26*li), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 0, 255), 2, cv2.LINE_AA)
    shot = ev_dir / "screenshot.jpg"
    cv2.imwrite(str(shot), ann)
    event = {
        "event_id": eid,
        "event_type": "black_screen",
        "source": source_name,
        "frame_index": frame_index,
        "source_timestamp_ms": int(round(video_time_s * 1000)) if video_time_s is not None else None,
        "capture_time": now.astimezone().isoformat(timespec="milliseconds"),
        "score": round(min(1.0, result["region"]["dark_pct"] / 100.0), 3),
        "bbox": result["region"]["bbox"],
        "screenshot": str(shot),
    }
    (ev_dir / "event.json").write_text(_json.dumps(event, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"  🚨 BLACK_SCREEN detected {'video_timestamp=' + _fmt_ts(video_time_s) if video_time_s is not None else ''} "
          f"screenshot saved event.json saved -> {eid}")
    if notifier:
        try:
            notifier(event)
            print("  📨 feishu notification sent")
        except Exception as e:
            print(f"  ⚠️ 飞书告警失败: {e}")
    return event


# ---------------------------------------------------------------- 批量自动检测
def _fmt_ts(seconds):
    ms = int(round(seconds * 1000))
    return f"{ms//3600000:02d}:{ms%3600000//60000:02d}:{ms%60000//1000:02d}.{ms%1000:03d}"


def batch_detect_video(video_path, out_root, step=1, hit_frames=3, cooldown_s=10.0,
                       notifier=None, max_screens=3, fixed_rois=None):
    """对单个视频自动运行黑屏检测：连续 hit_frames 个采样命中 → 生成事件，
    进入 cooldown_s 冷却；每块屏幕独立计数与冷却，事件按屏幕分目录归档。"""
    video_path = Path(video_path)
    print(f"\n▶ {video_path.name}")
    if fixed_rois:
        rois = list(fixed_rois)
    else:
        rois = calibrate_screen_rois(str(video_path), max_screens=max_screens) \
            if HAVE_DETECTOR else []
    if rois:
        print(f"  屏幕ROI ({len(rois)} 块): " +
              "  ".join(f"S{i}={r}" for i, r in enumerate(rois, 1)))
    else:
        print("  屏幕ROI: 未标定，按整幅画面检测")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  ❌ 无法打开"); return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    events, idx = [], 0
    per_screen = {}           # 每块屏幕独立的 streak / cooldown
    import json as _json
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            dets = run_detect_multi(frame, rois)
            for no, roi, r in dets:
                st = per_screen.setdefault(no, {"streak": 0, "cooldown_until": -1.0})
                if not r["abnormal"]:
                    st["streak"] = 0
                    continue
                st["streak"] += 1
                if st["streak"] < hit_frames or t < st["cooldown_until"]:
                    continue
                tag = f"_S{no}" if no else ""
                eid = f"{video_path.stem[:22]}{tag}_{len(events)+1:03d}"
                ev_dir = (out_root / video_path.stem[:40] /
                          (f"screen_{no}" if no else "full_frame") / eid)
                ev_dir.mkdir(parents=True, exist_ok=True)
                # 用全部屏幕的检测结果标注：编号才是真实编号，且保留其他屏幕作对照
                ann = annotate_live_multi(frame, dets, _fmt_ts(t), summary=False)
                now = datetime.now()
                # 抬头压缩到 3 行：行数越少，越不容易盖住画面里的屏幕
                lines = [f"{'SCREEN ' + str(no) + ' - ' if no else ''}BLACK SCREEN"
                         f"   Score: {min(1.0, r['region']['dark_pct']/100.0):.3f}",
                         f"Source: {video_path.name[:44]}"
                         f"   Frame: {idx}   Video Time: {_fmt_ts(t)}",
                         f"Capture: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"]
                y0 = 40
                # 抬头压半透明底板，避免文字与屏幕画面叠在一起看不清
                pw = max(cv2.getTextSize(t_, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
                         for t_ in lines) + 24
                panel = ann.copy()
                cv2.rectangle(panel, (0, 0), (pw, y0 + 26 * len(lines)), (0, 0, 0), -1)
                ann = cv2.addWeighted(panel, 0.55, ann, 0.45, 0)
                for li, txt in enumerate(lines):
                    cv2.putText(ann, txt, (12, y0 + 26*li), cv2.FONT_HERSHEY_SIMPLEX,
                                0.62, (0, 0, 255), 2, cv2.LINE_AA)
                shot = ev_dir / "screenshot.jpg"
                cv2.imwrite(str(shot), ann)
                event = {
                    "event_id": eid,
                    "event_type": "black_screen",
                    "source": video_path.name,
                    "screen_no": no or None,
                    "screen_total": max(1, len(rois)),
                    "frame_index": idx,
                    "source_timestamp_ms": int(round(t * 1000)),
                    "capture_time": now.astimezone().isoformat(timespec="milliseconds"),
                    "score": round(min(1.0, r["region"]["dark_pct"] / 100.0), 3),
                    "bbox": r["region"]["bbox"],
                    "screen_roi": list(roi) if roi else None,
                    "screenshot": str(shot),
                }
                (ev_dir / "event.json").write_text(
                    _json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
                events.append(event)
                if notifier:
                    try:
                        notifier(event)
                        print("  📨 飞书告警已发送")
                    except Exception as e:
                        print(f"  ⚠️ 飞书告警失败: {e}")
                st["cooldown_until"] = t + cooldown_s
                print(f"  🚨 {'S' + str(no) + ' ' if no else ''}BLACK_SCREEN detected "
                      f"video_timestamp={_fmt_ts(t)} frame={idx} "
                      f"score={event['score']} -> {ev_dir.name}")
        idx += 1
    cap.release()
    by_screen = {}
    for e in events:
        by_screen[e["screen_no"] or 0] = by_screen.get(e["screen_no"] or 0, 0) + 1
    detail = "  ".join(f"S{k or '-'}={v}" for k, v in sorted(by_screen.items()))
    print(f"  完成: {idx}/{total} 帧, 事件数={len(events)}" + (f"  ({detail})" if detail else ""))
    return events


def run_batch(args):
    if not HAVE_DETECTOR:
        print("❌ 未找到 analyze_black_screens.py，无法批量检测"); return
    target = Path(args.batch)
    videos = sorted(target.glob("*.mp4")) if target.is_dir() else [target]
    videos = [v for v in videos if v.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
    out_root = Path(args.out) if args.out else \
        Path(__file__).resolve().parent / "diag_captures" / f"batch_{datetime.now().strftime('%Y%m%d')}" / "black_screen"
    out_root.mkdir(parents=True, exist_ok=True)
    notifier = None
    if getattr(args, "notify", False):
        try:
            from notify.feishu_notifier import send_event
            notifier = send_event
            print("📨 飞书告警: 开 (FEISHU_WEBHOOK)")
        except Exception as e:
            print(f"⚠️ 飞书告警不可用: {e}")
    max_screens = max(1, getattr(args, "screens", 3) or 3)
    fixed_rois = parse_rois(args.roi) if args.roi else None
    print(f"批量检测 {len(videos)} 个视频, 采样步长={args.step}, "
          f"最多 {max_screens} 块屏幕, 输出: {out_root}")
    all_events = []
    for v in videos:
        all_events += batch_detect_video(v, out_root, step=args.step, notifier=notifier,
                                         max_screens=max_screens, fixed_rois=fixed_rois)
    import json as _json
    (out_root / "events_summary.json").write_text(
        _json.dumps(all_events, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n合计事件 {len(all_events)} 个, 汇总: {out_root / 'events_summary.json'}")


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="摄像头诊断 + 黑屏检测预览")
    parser.add_argument("--video", help="播放视频文件而非相机（手动复核问题单视频）")
    parser.add_argument("--device", type=int, default=None, help="跳过诊断直接打开指定相机 index")
    parser.add_argument("--detect", action="store_true", help="开启黑屏检测红框叠加")
    parser.add_argument("--roi",
                        help="固定屏幕ROI: 单屏 x,y,w,h；多屏用分号分隔 "
                             "x,y,w,h;x,y,w,h;x,y,w,h（台架机位固定后推荐）")
    parser.add_argument("--screens", type=int, default=3,
                        help="画面内最多识别几块屏幕（默认3，设1为旧的单屏行为）")
    parser.add_argument("--batch", help="批量自动检测：目录或单个视频文件（无GUI）")
    parser.add_argument("--step", type=int, default=1, help="批量模式采样步长（每N帧检测1帧）")
    parser.add_argument("--out", help="批量模式输出目录（默认 diag_captures/batch_日期/black_screen）")
    parser.add_argument("--notify", action="store_true",
                        help="事件生成后发送飞书告警（需设置环境变量 FEISHU_WEBHOOK）")
    parser.add_argument("--evidence-n", type=int, default=10, dest="evidence_n",
                        help="合并证据帧数(默认10, 最大40; 帧间隔0.4s去重)")
    args = parser.parse_args()

    if args.batch:
        run_batch(args)
        return

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {args.video}"); return
        preview_loop(cap, {"backend": "FILE"}, args)
        return

    result = open_device(args.device) if args.device is not None else diagnose_all()
    if result:
        cap, info = result
        print(f"\n  ✅ 画面可用: idx={info['idx']} {info['backend']} "
              f"{info.get('fourcc','')} {info.get('w')}x{info.get('h')} mean={info['mean']:.1f}")
        preview_loop(cap, info, args)
    else:
        privacy_hint()


if __name__ == "__main__":
    main()
