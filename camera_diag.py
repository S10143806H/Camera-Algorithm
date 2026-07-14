"""
USB2.0 摄像头诊断 + 黑屏检测预览工具
====================================
1. 摄像头诊断：强制 MJPG 四字节码 + 遍历 backend/fourcc/分辨率，解决"打开了却黑屏"。
2. 参数调节：预览窗口带滑动条（亮度/对比度/饱和度/增益/曝光/自动曝光/对焦）。
3. 黑屏检测：--detect 开启，复用 analyze_black_screens.detect_dark_region，
   在预览中用红色高亮框圈出检测到的黑屏区域，橙框为标定的屏幕 ROI。

用法:
  python camera_diag.py                              # 诊断 + 纯预览
  python camera_diag.py --detect                     # 摄像头预览 + 黑屏检测
  python camera_diag.py --video ticket.mp4 --detect  # 手动播放问题单视频 + 检测
  python camera_diag.py --device 0 --detect          # 跳过诊断直接开指定相机
  python camera_diag.py --detect --roi 64,0,896,516  # 固定屏幕 ROI（推荐台架用）
  python camera_diag.py --batch data_source --step 3   # 批量自动检测目录内全部视频

预览快捷键:
  Q 退出        SPACE 暂停/继续      . 单帧步进(视频)   , 单帧步退(视频)
  D 检测开/关   R 鼠标框选屏幕ROI    C 自动标定ROI      X 清除ROI
  S 保存当前帧截图(diag_captures/)

黑屏检测重要相机参数（按影响排序）:
  1. AutoExposure=0(手动) —— 屏幕变黑时自动曝光会拉亮整个画面，
     使黑屏帧亮度漂移，是误报/漏报的第一来源，务必关掉后固定 Exposure。
  2. Gain 固定低增益 —— 高增益噪点会抬高暗区灰度和边缘密度，干扰 P1 规则。
  3. 白平衡锁定（部分相机需在厂商工具中关 AWB）。
  4. Focus 手动锁定 —— 失焦会模糊 pane 边界。
  5. ROI 固定 —— 台架机位固定后用 C 标定一次，抄下坐标写入 --roi。
  6. 分辨率/FOURCC：MJPG ≥720p；Brightness/Contrast 保持驱动默认居中值。
"""

import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from analyze_black_screens import (
        detect_dark_region,
        draw_annotation,
        calibrate_screen_roi,
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


def roi_from_maxbright(max_bright):
    """由逐像素最大亮度图求屏幕 bbox（与 calibrate_screen_roi 同规则）。"""
    h, w = max_bright.shape
    mask = (max_bright > 90).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh > best_area:
            best_area = bw * bh
            best = (int(x), int(y), int(bw), int(bh))
    if best is not None and best_area >= 0.05 * w * h:
        return best
    return None


# ---------------------------------------------------------------- 相机参数
# (滑条名, 属性, 滑条最大值, 换算函数)   曝光滑条值 n 实际设置 -n (DSHOW 对数刻度)
CAM_PARAMS = [
    ("Brightness",  cv2.CAP_PROP_BRIGHTNESS,    255, lambda v: v),
    ("Contrast",    cv2.CAP_PROP_CONTRAST,      255, lambda v: v),
    ("Saturation",  cv2.CAP_PROP_SATURATION,    255, lambda v: v),
    ("Gain",        cv2.CAP_PROP_GAIN,          255, lambda v: v),
    ("Exposure -n", cv2.CAP_PROP_EXPOSURE,       13, lambda v: -v),
    ("Focus",       cv2.CAP_PROP_FOCUS,         255, lambda v: v),
]


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
        # DSHOW: 0.75=自动 0.25=手动;  MSMF: 1=自动 0=手动
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if v else 0.25)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if v else 0)
        print(f"  AutoExposure -> {'自动' if v else '手动(建议)'}")

    cv2.createTrackbar("AutoExp 0/1", win, 1, 1, auto_exp_cb)


# ---------------------------------------------------------------- 设备枚举
def get_camera_symlinks():
    """列出系统摄像头设备（PowerShell）。"""
    ps = r"""
$devs = Get-PnpDevice -Class Camera -Status OK
foreach ($d in $devs) {
    $name = $d.FriendlyName
    $iid  = $d.InstanceId
    $props = Get-PnpDeviceProperty -InstanceId $iid -KeyName 'DEVPKEY_Device_SymbolicLink' -ErrorAction SilentlyContinue
    $sym = if ($props) { $props.Data } else { "" }
    if (-not $sym) {
        $ifaces = Get-PnpDeviceInterface -InstanceId $iid -ErrorAction SilentlyContinue
        $sym = ($ifaces | Select-Object -First 1).SymbolicLink
    }
    "$name||$iid||$sym"
}
"""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps], text=True, timeout=20)
    except Exception as e:
        print(f"  ❌ PowerShell 查询失败: {e}")
        return []
    devices = []
    for line in out.strip().splitlines():
        if "||" in line:
            parts = line.strip().split("||")
            devices.append({"name": parts[0].strip(),
                            "instance_id": parts[1].strip() if len(parts) > 1 else "",
                            "symlink": parts[2].strip() if len(parts) > 2 else ""})
    return devices


def open_device(idx):
    """跳过完整诊断，按常用组合快速打开指定相机。"""
    for be_code, be_name in [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW")]:
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
            return cap, {"idx": idx, "backend": be_name, "fourcc": "MJPG",
                         "w": int(cap.get(3)), "h": int(cap.get(4)),
                         "fps": cap.get(cv2.CAP_PROP_FPS), "mean": frame.mean()}
        cap.release()
    return None


def diagnose_all():
    """遍历 index × backend × fourcc × 分辨率组合，找可用画面。"""
    print("=" * 60)
    print("  🔍 摄像头诊断 — 遍历所有组合")
    print("=" * 60)
    devices = get_camera_symlinks()
    print(f"\n📋 系统摄像头设备 ({len(devices)} 个):")
    for d in devices:
        print(f"  📷 {d['name']}\n     ID:  {d['instance_id']}\n     Sym: {d['symlink'][:80]}")

    backends = [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW")]
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

    screen_roi = None
    if args.roi:
        screen_roi = tuple(int(v) for v in args.roi.split(","))
        print(f"  🖥️ 使用固定屏幕ROI: {screen_roi}")
    elif is_video and HAVE_DETECTOR:
        print("  🖥️ 视频模式自动标定屏幕ROI ...")
        screen_roi = calibrate_screen_roi(args.video)
        print(f"     标定结果: {screen_roi}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    win = ("视频回放" if is_video else f"USB Camera [{info.get('backend','')}]") + \
          " | Q退出 SPACE暂停 D检测 R框选ROI C标定 S截图"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if not is_video:
        setup_trackbars(win, cap)

    calib_buf = deque(maxlen=40)      # 相机模式滚动标定缓存
    last_calib_sample = 0.0
    save_dir = Path(__file__).resolve().parent / "diag_captures"
    paused = False
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
    streak, cooldown_until, event_seq = 0, 0.0, 0
    EVIDENCE_N = max(3, getattr(args, "evidence_n", 10) or 10)
    EVIDENCE_MIN = 3
    EVIDENCE_GAP = 0.4        # 证据帧最小间隔(秒)，去掉重复相似帧
    evidence = []
    last_evi_t = -1e9         # 上一条证据的时间(视频秒或墙钟秒)
    event_root = save_dir / f"live_{datetime.now().strftime('%Y%m%d')}" / "black_screen"
    source_name = Path(args.video).name if is_video else f"camera_{info.get('idx', '?')}" 

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

        # ---- 检测 ----
        if detect_on and advance:
            result = run_detect(frame, screen_roi)
            now_wc = datetime.now()
            vt = (cap.get(cv2.CAP_PROP_POS_FRAMES) / fps) if is_video else None
            fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if is_video else None
            ts_text = (_fmt_ts(vt) if is_video
                       else now_wc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            cur_t = vt if is_video else time.time()
            if result["abnormal"]:
                streak += 1
                if time.time() >= cooldown_until and cur_t - last_evi_t >= EVIDENCE_GAP:
                    evidence.append((frame.copy(), result, ts_text, vt, fi, now_wc))
                    last_evi_t = cur_t
                    if len(evidence) >= EVIDENCE_N:
                        event_seq += 1
                        threading.Thread(
                            target=emit_merged_event,
                            args=(evidence, source_name, event_root, event_seq),
                            kwargs={"notifier": notifier}, daemon=True).start()
                        evidence = []
                        cooldown_until = time.time() + 10.0
            else:
                if len(evidence) >= EVIDENCE_MIN and time.time() >= cooldown_until:
                    event_seq += 1
                    threading.Thread(
                        target=emit_merged_event,
                        args=(evidence, source_name, event_root, event_seq),
                        kwargs={"notifier": notifier}, daemon=True).start()
                    cooldown_until = time.time() + 10.0
                evidence = []
                streak = 0

        if detect_on and result is not None:
            ts_disp = (_fmt_ts(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps) if is_video
                       else datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            display = annotate_live(frame, result, ts_disp)
        else:
            display = frame.copy()
            if screen_roi:
                rx, ry, rw, rh = screen_roi
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 160, 0), 2)

        # ---- 状态栏 ----
        gray_mean = frame.mean()
        if is_video:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            status = f"frame {pos}/{total}  t={pos/fps:6.2f}s"
        else:
            status = f"{info.get('w','?')}x{info.get('h','?')} {info.get('backend','')}"
        status += f"  mean={gray_mean:.0f}  detect={'ON' if detect_on else 'off'}"
        status += f"  ROI={'set' if screen_roi else 'none'}"
        if paused:
            status += "  [PAUSED]"
        cv2.putText(display, status, (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, display)

        # ---- 按键 ----
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
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
            sel = cv2.selectROI(win, frame, showCrosshair=True)
            if sel[2] > 10 and sel[3] > 10:
                screen_roi = tuple(int(v) for v in sel)
                print(f"  🖥️ 手动ROI = {screen_roi}  (可写入 --roi {','.join(map(str,screen_roi))})")
        elif key in (ord('c'), ord('C')):
            if is_video and HAVE_DETECTOR:
                screen_roi = calibrate_screen_roi(args.video)
            elif calib_buf:
                mb = calib_buf[0].copy()
                for g in calib_buf:
                    mb = np.maximum(mb, g)
                screen_roi = roi_from_maxbright(mb)
            print(f"  🖥️ 自动标定ROI = {screen_roi}")
        elif key in (ord('x'), ord('X')):
            screen_roi = None
            print("  ROI 已清除")
        elif key in (ord('s'), ord('S')):
            save_dir.mkdir(exist_ok=True)
            p = save_dir / f"cap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            cv2.imwrite(str(p), display)
            print(f"  💾 已保存 {p}")

    cap.release()
    cv2.destroyAllWindows()





def annotate_live(frame, result, ts_text):
    """实时/回放标注：红框 + 检测文字，左下角为真实时间戳(取代读秒)。"""
    annotated = frame.copy()
    if result.get("screen_roi"):
        rx, ry, rw, rh = result["screen_roi"]
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 160, 0), 2)
    if result["abnormal"] and result["region"]:
        x, y, w_, h_ = result["region"]["bbox"]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x, y), (x + w_, y + h_), (0, 0, 255), -1)
        annotated = cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0)
        cv2.rectangle(annotated, (x, y), (x + w_, y + h_), (0, 0, 255), 5)
        cv2.putText(annotated,
                    f"BLACK SCREEN DETECTED  region_dark={result['region']['dark_pct']:.1f}%",
                    (max(12, x), max(34, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(annotated, "normal/dim screen", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated, ts_text, (12, annotated.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
    return annotated


def emit_merged_event(evidence, source_name, out_root, seq, notifier=None):
    """把缓存的多帧证据合并成一次完整事件：拼图 + event.json + 飞书(带图)。

    evidence: [(frame, result, ts_text, video_time_s, frame_index, wallclock), ...]
    """
    import json as _json
    now = datetime.now()
    eid = f"CAM_{now.strftime('%Y%m%d_%H%M%S')}_{seq:03d}"
    ev_dir = out_root / eid
    ev_dir.mkdir(parents=True, exist_ok=True)

    # 逐帧标注 -> 缩略 -> 拼图 (5列)
    thumbs = []
    for frame, result, ts_text, vt, fi, wc in evidence:
        ann = annotate_live(frame, result, ts_text)
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

    best = max(evidence, key=lambda e: e[1]["region"]["dark_pct"])
    score = round(min(1.0, best[1]["region"]["dark_pct"] / 100.0), 3)
    t0, t1 = evidence[0], evidence[-1]
    span = (f"video {_fmt_ts(t0[3])} ~ {_fmt_ts(t1[3])}" if t0[3] is not None
            else f"{t0[5].strftime('%H:%M:%S.%f')[:-3]} ~ {t1[5].strftime('%H:%M:%S.%f')[:-3]}")
    head_lines = [f"BLACK SCREEN - MERGED EVIDENCE ({len(evidence)} frames)",
                  f"Source: {source_name[:60]}   Span: {span}   Score: {score}"]
    for li, txt in enumerate(head_lines):
        cv2.putText(sheet, txt, (14, 36 + 34 * li), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 0, 255) if li == 0 else (60, 60, 60), 2, cv2.LINE_AA)
    shot = ev_dir / "screenshot.jpg"
    cv2.imwrite(str(shot), sheet)

    event = {
        "event_id": eid,
        "event_type": "black_screen",
        "source": source_name,
        "frame_count": len(evidence),
        "frame_index": t0[4],
        "frame_index_end": t1[4],
        "source_timestamp_ms": int(round(t0[3] * 1000)) if t0[3] is not None else None,
        "source_timestamp_end_ms": int(round(t1[3] * 1000)) if t1[3] is not None else None,
        "capture_time": t0[5].astimezone().isoformat(timespec="milliseconds"),
        "capture_time_end": t1[5].astimezone().isoformat(timespec="milliseconds"),
        "score": score,
        "bbox": best[1]["region"]["bbox"],
        "screenshot": str(shot),
    }
    (ev_dir / "event.json").write_text(_json.dumps(event, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"  🚨 BLACK_SCREEN event ({len(evidence)} frames merged) span={span} "
          f"score={score} -> {eid}")
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
                       notifier=None):
    """对单个视频自动运行黑屏检测：连续 hit_frames 个采样命中 → 生成事件，
    进入 cooldown_s 冷却；每个事件保存标注截图 + event.json。"""
    video_path = Path(video_path)
    print(f"\n▶ {video_path.name}")
    roi = calibrate_screen_roi(str(video_path)) if HAVE_DETECTOR else None
    print(f"  屏幕ROI: {roi}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  ❌ 无法打开"); return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    events, streak, cooldown_until, idx = [], 0, -1.0, 0
    import json as _json
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            r = run_detect(frame, roi)
            if r["abnormal"]:
                streak += 1
                if streak >= hit_frames and t >= cooldown_until:
                    eid = f"{video_path.stem[:22]}_{len(events)+1:03d}"
                    ev_dir = out_root / video_path.stem[:40] / eid
                    ev_dir.mkdir(parents=True, exist_ok=True)
                    ann = draw_annotation(frame, r, t)
                    now = datetime.now()
                    lines = ["BLACK SCREEN",
                             f"Source: {video_path.name[:44]}",
                             f"Video Time: {_fmt_ts(t)}",
                             f"Capture Time: {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
                             f"Frame: {idx}",
                             f"Score: {min(1.0, r['region']['dark_pct']/100.0):.3f}"]
                    y0 = 60
                    for li, txt in enumerate(lines):
                        cv2.putText(ann, txt, (12, y0 + 26*li), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.62, (0, 0, 255), 2, cv2.LINE_AA)
                    shot = ev_dir / "screenshot.jpg"
                    cv2.imwrite(str(shot), ann)
                    event = {
                        "event_id": eid,
                        "event_type": "black_screen",
                        "source": video_path.name,
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
                    cooldown_until = t + cooldown_s
                    print(f"  🚨 BLACK_SCREEN detected video_timestamp={_fmt_ts(t)} "
                          f"frame={idx} score={event['score']} -> {ev_dir.name}")
            else:
                streak = 0
        idx += 1
    cap.release()
    print(f"  完成: {idx}/{total} 帧, 事件数={len(events)}")
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
    print(f"批量检测 {len(videos)} 个视频, 采样步长={args.step}, 输出: {out_root}")
    all_events = []
    for v in videos:
        all_events += batch_detect_video(v, out_root, step=args.step, notifier=notifier)
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
    parser.add_argument("--roi", help="固定屏幕ROI: x,y,w,h（台架机位固定后推荐）")
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
        print("\n  ❌ 所有组合均黑屏！可能原因：")
        print("     1. 摄像头隐私开关未打开（物理或 Windows 隐私设置）")
        print("     2. 摄像头被其他程序占用")
        print("     3. 驱动问题，需更新固件")
        print("  💡 Windows 设置 → 隐私 → 相机 → 允许应用访问相机 = 开")


if __name__ == "__main__":
    main()
