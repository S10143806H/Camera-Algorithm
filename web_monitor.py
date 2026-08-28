"""
屏幕异常实时监控 —— Web 版（局域网内浏览器访问）
================================================
把采集 + 多屏黑屏检测跑成后台线程，通过 HTTP 对外提供：
  * MJPEG 实时画面（带 S1/S2/S3 标注与红框告警）
  * 网页内鼠标框选 ROI、一键自动标定
  * 网页内调相机参数（亮度/对比度/饱和度/增益/曝光/自动曝光）
  * 设备投屏：adb 拉 Android 设备屏幕显示在相机画面下方作对照；
    同一时刻只投一块，网页上可切换（并发投屏会串台且拖低帧率）
  * 事件列表与证据截图

相机同一时刻只能被一个进程占用：本服务运行期间不要再开 camera_diag.py 预览。

用法:
  python3 web_monitor.py                          # 0.0.0.0:8000, 自动挑相机
  python3 web_monitor.py --device 0 --screens 3
  python3 web_monitor.py --port 8080 --roi 20,60,300,200;340,40,320,220
  python3 web_monitor.py --notify                 # 事件推飞书(需 FEISHU_WEBHOOK)

访问: http://<本机IP>:8000/
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from camera_diag import (  # noqa: E402
    open_stream,
    build_param_specs,
    probe_param_range,
    annotate_live_multi,
    diagnose_all,
    emit_merged_event,
    format_rois,
    open_device,
    order_rois,
    parse_rois,
    run_detect_multi,
)
from device_context import DeviceContext  # noqa: E402
from live_detectors import (ALL_TYPES, BLACK, TYPE_COLORS, TYPE_LABELS,  # noqa: E402
                            TYPE_SHORT, DetectorBank, normalize_types)
from device_screen import DeviceScreenManager, pick_device, send_input  # noqa: E402
from gtmp_link import GtmpLink  # noqa: E402
from platform_compat import list_cameras, set_auto_exposure  # noqa: E402

try:
    from analyze_black_screens import calibrate_screen_rois, rois_from_maxbright
    HAVE_DETECTOR = True
except ImportError:
    HAVE_DETECTOR = False

INSTANCE = "local"                     # 实例名，--instance 覆盖
ROI_STORE = ROOT / "screen_rois.json"
# 网页上的飞书开关要能扛住重启：systemd 单元里没有 --notify，重启后开关会被
# 重置成关，等于"开了但下次告警不发"，比一直关着更坑
STATE_STORE = ROOT / "monitor_state.json"
EVIDENCE_N = 10
EVIDENCE_MIN = 3
EVIDENCE_GAP = 0.4
COOLDOWN_S = 10.0
MAX_EVENTS_KEPT = 200


def configure_instance(name):
    """按实例名重绑持久化文件，让多路相机可以在同一台机器上并存。

    ROI、飞书开关、事件目录原本都是写死的单例文件。第二个实例（远端相机）
    起来后会把第一个实例标定好的 ROI 直接覆盖掉，两边一起误报。默认实例
    仍用原文件名，保证现有部署与已标定的 ROI 不受影响。
    """
    global INSTANCE, ROI_STORE, STATE_STORE
    INSTANCE = name or "local"
    suffix = "" if INSTANCE == "local" else "_" + INSTANCE
    ROI_STORE = ROOT / ("screen_rois%s.json" % suffix)
    STATE_STORE = ROOT / ("monitor_state%s.json" % suffix)
    return INSTANCE


# ---------------------------------------------------------------- ROI 持久化
def load_rois(frame_size=None):
    """读取上次保存的 ROI，使网页里标定过一次后重启仍生效。

    ROI 是像素坐标，和标定时的采集分辨率绑定。相机重新枚举后 OpenCV 可能
    协商到另一档分辨率（实测 1280x720 → 640x480），此时旧 ROI 会整体错位甚至
    越界，检测立刻开始误报黑屏。这里按分辨率比例换算回来。
    """
    try:
        data = json.loads(ROI_STORE.read_text(encoding="utf-8"))
        rois = [tuple(int(v) for v in r) for r in data.get("rois", [])]
    except Exception:
        return []
    if not rois or not frame_size:
        return rois

    saved = data.get("frame_size")
    cw, ch = frame_size
    if not saved or len(saved) != 2 or not all(saved):
        print(f"⚠️ 已存 ROI 未记录标定分辨率，按当前 {cw}x{ch} 直接使用；"
              f"若框位不对请在网页上重新标定")
        return [_clamp_roi(r, cw, ch) for r in rois]

    sw, sh = int(saved[0]), int(saved[1])
    if (sw, sh) == (cw, ch):
        return rois
    fx, fy = cw / sw, ch / sh
    scaled = [_clamp_roi((int(round(x * fx)), int(round(y * fy)),
                          int(round(w * fx)), int(round(h * fy))), cw, ch)
              for x, y, w, h in rois]
    print(f"🔄 采集分辨率由标定时的 {sw}x{sh} 变为 {cw}x{ch}，ROI 已按比例换算")
    return scaled


def load_state():
    try:
        return json.loads(STATE_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(**kv):
    st = load_state()
    st.update(kv)
    try:
        STATE_STORE.write_text(json.dumps(st, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    except OSError as e:
        print(f"⚠️ 状态保存失败: {e}")


def _clamp_roi(roi, w, h):
    """把 ROI 夹回画面内，越界的框会让检测读到空区域而误判黑屏。"""
    x, y, rw, rh = roi
    x = max(0, min(int(x), w - 1)); y = max(0, min(int(y), h - 1))
    rw = max(1, min(int(rw), w - x)); rh = max(1, min(int(rh), h - y))
    return (x, y, rw, rh)


def save_rois(rois, frame_size=None):
    try:
        payload = {"rois": [list(r) for r in rois],
                   "saved_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        if frame_size:
            payload["frame_size"] = [int(frame_size[0]), int(frame_size[1])]
        ROI_STORE.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    except OSError as e:
        print(f"⚠️ ROI 保存失败: {e}")


def _flatten(multi):
    """把 [(屏号, roi, {类型: 结果})] 摊平成标注函数要的 [(屏号, roi, 结果)]。

    每块屏每个异常类型各占一项，且带上类型的标签与配色，便于同一帧上把
    黑屏/花屏/闪屏画成不同颜色。正常的类型保留一项（不画框），以便状态栏
    仍能显示该屏"ok"。
    """
    flat = []
    for no, roi, by_type in multi:
        abnormal = [(t, r) for t, r in by_type.items() if r["abnormal"]]
        if not abnormal:
            flat.append((no, roi, {"abnormal": False, "region": None}))
            continue
        for t, r in abnormal:
            item = dict(r)
            item["type"] = t
            item["label"] = TYPE_LABELS.get(t, t.upper())
            item["short"] = TYPE_SHORT.get(t, t.upper()[:6])
            item["color"] = TYPE_COLORS.get(t, (0, 0, 255))
            if t == BLACK and r.get("raw"):
                item["region"] = (r["raw"] or {}).get("region")
            flat.append((no, roi, item))
    return flat


# ---------------------------------------------------------------- 数字变焦
# 相机参数里的 Zoom 只作用于预览画面：裁一块再放回原尺寸。
# 不动检测输入是有意为之——放大到 2x 后另外两块屏就在取景框外了，
# 若检测也跟着放大，它们的 ROI 会立刻退化成空区域并连报黑屏。
ZOOM_MIN, ZOOM_MAX = 100, 400        # 百分比，1.00x ~ 4.00x


def _apply_zoom(img, zoom, cx, cy):
    """按 zoom 倍率、以 (cx, cy) 归一化中心裁剪并放回原尺寸。"""
    z = max(1.0, float(zoom))
    if z <= 1.0 + 1e-6:
        return img
    h, w = img.shape[:2]
    cw, ch = max(16, int(w / z)), max(16, int(h / z))
    x0 = min(max(int(cx * w - cw / 2), 0), w - cw)
    y0 = min(max(int(cy * h - ch / 2), 0), h - ch)
    crop = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- 监控服务
class MonitorService:
    """唯一的相机持有者：采集 → 逐屏检测 → 标注 → 编码 JPEG 供 HTTP 取用。"""

    def __init__(self, cap, info, max_screens=3, rois=None, detect=True,
                 notifier=None, jpeg_quality=75, types=None, notify_on=False):
        self.cap = cap
        self.info = info
        # 相机控制项的真实取值范围要向驱动探，换了设备就得重探（置 None 触发）
        self._param_specs = None
        self.max_screens = max(1, max_screens)
        self.detect_on = detect and HAVE_DETECTOR
        self.types = normalize_types(types)
        self.bank = DetectorBank(self.types, fps=float(info.get("fps") or 30.0))
        self.bank.set_rois(rois or [])
        # 发送函数始终持有（只要模块导入成功），开关只控制发不发，
        # 这样网页上可以随时开启，不必重启服务改命令行
        self._notify_fn = notifier
        self.notify_on = bool(notifier) and notify_on
        self.jpeg_quality = jpeg_quality

        self.lock = threading.Lock()          # 保护 _jpeg / _raw / results / rois
        self.frame_event = threading.Event()  # 有新帧时唤醒所有 MJPEG 订阅者
        self._jpeg = None
        self._raw = None
        self.rois = list(rois or [])
        self.results = []
        self.multi = []                       # [(屏号, roi, {类型: 结果})]
        self.fps = 0.0
        self.running = True
        self.zoom = 100                       # 预览数字变焦，百分比
        self.pan = [50, 50]                   # 变焦中心，百分比

        self.events = []                      # 最近事件（内存），完整证据在磁盘
        self.event_seq = 0
        self._screen_state = {}
        self._calib_buf = []                  # 滚动标定缓存（每 0.5s 一帧，最多 40）
        self._last_calib_sample = 0.0
        # 事件按 screen_N/<异常类型>/ 分目录，根目录不再写死 black_screen
        self.event_root = (ROOT / "diag_captures"
                           / f"web_{datetime.now().strftime('%Y%m%d')}_{INSTANCE}")
        self.source_name = f"{INSTANCE}_camera_{info.get('idx', '?')}"
        # 网络流源：掉线后不重新枚举 USB，直接按同一个 URL 重连
        self.stream_url = info.get("url") or None
        # 掉线自愈：USB 重新枚举后 /dev/videoN 会换号，旧 cap 只会一直读失败
        self.cam_name = info.get("name") or ""
        self.cam_status = "采集中"
        self.cam_online = True
        self.reconnects = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ---------------------------------------------------------- 飞书告警
    @property
    def notifier(self):
        return self._notify_fn if self.notify_on else None

    def set_notify(self, on):
        if on and not self._notify_fn:
            raise RuntimeError("飞书告警模块不可用，无法开启")
        self.notify_on = bool(on)
        save_state(notify_on=self.notify_on)   # 重启后保持，否则等于静默失效
        print(f"📨 飞书告警 -> {'开' if self.notify_on else '关'}")
        return self.notify_on

    # ---------------------------------------------------------- 生命周期
    def start(self):
        # 探范围要反复写控制项再还原，画面会短暂抖动；放在采集线程起来之前做，
        # 免得运行中探测把抖动喂给检测器，误报一串异常
        self.param_specs()
        self._thread.start()

    def stop(self):
        self.running = False
        self._thread.join(timeout=3)
        self.cap.release()

    # ---------------------------------------------------------- 主循环
    def _state_of(self, no):
        return self._screen_state.setdefault(
            no, {"streak": 0, "cooldown_until": 0.0, "evidence": [], "last_evi_t": -1e9})

    # ---------------------------------------------------------- 掉线自愈
    def _reopen_camera(self):
        """相机掉线后重新枚举并打开。

        USB 重新枚举会让 /dev/videoN 换号（实测 video0 → video1），此时旧 cap
        只会一直读失败：采集线程空转、MJPEG 全部掉到 ~1fps，页面看着像卡死却
        没有任何提示。这里按设备名重新找回，找不到名字再退回逐个 index 探测。
        """
        try:
            self.cap.release()
        except Exception:
            pass

        if self.stream_url:                 # 网络流：URL 不会变，重开即可
            opened = open_stream(self.stream_url)
            if not opened:
                return False
            cap, info = opened
            self.cap = cap
            self._param_specs = None
            with self.lock:
                old = (int(self.info.get("w") or 0), int(self.info.get("h") or 0))
                new = (int(info.get("w") or 0), int(info.get("h") or 0))
                self.info = info
                # 重连时推流端可能还没起稳，OpenCV 会协商到 640x480 之类的其它档位。
                # ROI 是像素坐标，不跟着换算就会整体越界：检测框跑出画面、
                # 分区裁剪出空切片(Mean of empty slice)、所有屏立刻误报黑屏。
                if old != new and all(old) and all(new) and self.rois:
                    fx, fy = new[0] / old[0], new[1] / old[1]
                    self.rois = [_clamp_roi((round(x * fx), round(y * fy),
                                             round(w * fx), round(h * fy)), *new)
                                 for x, y, w, h in self.rois]
                    self._screen_state.clear()
                    print(f"🔄 重连后分辨率 {old[0]}x{old[1]} → {new[0]}x{new[1]}，ROI 已换算")
                self.cam_online, self.cam_status = True, "采集中"
                self.reconnects += 1
            print(f"✅ 视频流已重连: {self.stream_url} "
                  f"{info.get('w')}x{info.get('h')}（第 {self.reconnects} 次）")
            return True

        cams = list_cameras()
        order = []
        if self.cam_name:                       # 优先按设备名找回同一台相机
            order += [c for c in cams if c.get("name") == self.cam_name]
        order += [c for c in cams if c not in order]
        candidates = []
        for c in order:
            digits = "".join(ch for ch in str(c.get("instance_id", "")) if ch.isdigit())
            if digits:
                candidates.append(int(digits))
        candidates += [i for i in range(6) if i not in candidates]

        for idx in candidates:
            opened = open_device(idx)
            if not opened:
                continue
            cap, info = opened
            self.cap = cap
            self._param_specs = None
            with self.lock:
                old = (int(self.info.get("w") or 0), int(self.info.get("h") or 0))
                new = (int(info.get("w") or 0), int(info.get("h") or 0))
                self.info = info
                if self.cam_name:
                    self.info.setdefault("name", self.cam_name)
                # 重连后分辨率若变了，ROI 必须同步换算，否则立刻开始误报黑屏
                if old != new and all(old) and all(new) and self.rois:
                    fx, fy = new[0] / old[0], new[1] / old[1]
                    self.rois = [_clamp_roi((round(x * fx), round(y * fy),
                                             round(w * fx), round(h * fy)), *new)
                                 for x, y, w, h in self.rois]
                    self._screen_state.clear()
                    print(f"🔄 重连后分辨率 {old[0]}x{old[1]} → {new[0]}x{new[1]}，ROI 已换算")
                self.cam_online, self.cam_status = True, "采集中"
                self.reconnects += 1
            print(f"✅ 相机已重连: idx={idx} {info.get('backend')} "
                  f"{info.get('w')}x{info.get('h')}（第 {self.reconnects} 次）")
            return True
        return False

    def _loop(self):
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        t_last, n = time.time(), 0
        fail = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fail += 1
                # 连续 30 次(约 1.5s)读不到帧即判掉线，避免偶发丢帧误触发重连
                if fail == 30:
                    with self.lock:
                        self.cam_online = False
                        self.cam_status = "相机掉线，重连中…"
                    print("⚠️ 相机连续读帧失败，判定掉线，开始重新枚举…")
                if fail >= 30 and fail % 30 == 0:
                    if self._reopen_camera():
                        fail = 0
                        continue
                    with self.lock:
                        self.cam_status = "相机掉线，重连中…（未找到设备）"
                time.sleep(0.05)
                continue
            if fail:
                fail = 0
                with self.lock:
                    self.cam_online, self.cam_status = True, "采集中"

            now = time.time()
            if now - self._last_calib_sample > 0.5:
                self._calib_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                del self._calib_buf[:-40]
                self._last_calib_sample = now

            with self.lock:
                rois = list(self.rois)
                detect_on = self.detect_on

            multi = self.bank.run(frame, now) if detect_on else []
            if detect_on:
                self._gate_events(frame, multi)
            results = _flatten(multi)          # 供标注与状态接口使用

            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            display = (annotate_live_multi(frame, results, ts) if results
                       else self._plain(frame, rois, ts))
            with self.lock:
                zoom, (px, py) = self.zoom, self.pan
            if zoom > 100:
                display = _apply_zoom(display, zoom / 100.0, px / 100.0, py / 100.0)

            ok, buf = cv2.imencode(".jpg", display, enc)
            if not ok:
                continue

            n += 1
            if now - t_last >= 1.0:
                self.fps, n, t_last = n / (now - t_last), 0, now

            with self.lock:
                self._jpeg = buf.tobytes()
                self._raw = frame
                self.results = results
                self.multi = multi         # 逐屏逐类型的原始分数，供状态接口展示
            self.frame_event.set()
            self.frame_event.clear()

    def _plain(self, frame, rois, ts):
        from camera_diag import _put_text_clamped, draw_screen_boxes
        out = frame.copy()
        if rois:
            draw_screen_boxes(out, list(enumerate(rois, 1)))
        _put_text_clamped(out, ts, 12, out.shape[0] - 40, 0.7, (0, 215, 255))
        return out

    def _gate_events(self, frame, multi):
        """逐屏 × 逐类型独立门控：连续命中攒够证据帧 → 落盘事件 → 进入冷却。

        状态按 (屏号, 类型) 分开：同一块屏的花屏和黑屏互不干扰，一种异常的冷却
        窗口不会把另一种吞掉。
        """
        wc = datetime.now()
        ts = wc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        cur = time.time()
        total = len(multi)
        for no, roi, by_type in multi:
            for typ, r in by_type.items():
                st = self._state_of((no, typ))
                # 证据里带上类型标签/配色，事件图才不会一律画成黑屏样式
                payload = dict(r.get("raw") or r)
                payload["type"] = typ
                payload["label"] = TYPE_LABELS.get(typ, typ.upper())
                payload["color"] = TYPE_COLORS.get(typ, (0, 0, 255))
                payload["short"] = TYPE_SHORT.get(typ, typ.upper()[:6])
                if typ != BLACK:
                    payload.pop("region", None)
                if r["abnormal"]:
                    st["streak"] += 1
                    if cur >= st["cooldown_until"] and cur - st["last_evi_t"] >= EVIDENCE_GAP:
                        st["evidence"].append((frame.copy(), payload, ts, None, None, wc))
                        st["last_evi_t"] = cur
                        if len(st["evidence"]) >= EVIDENCE_N:
                            self._emit(st, no, roi, total, typ)
                else:
                    if (len(st["evidence"]) >= EVIDENCE_MIN
                            and cur >= st["cooldown_until"]):
                        self._emit(st, no, roi, total, typ)
                    st["evidence"] = []
                    st["streak"] = 0

    def _emit(self, st, no, roi, total, typ=BLACK):
        self.event_seq += 1
        evidence, seq = st["evidence"], self.event_seq
        st["evidence"] = []
        st["cooldown_until"] = time.time() + COOLDOWN_S

        def work():
            try:
                ev = emit_merged_event(evidence, self.source_name, self.event_root, seq,
                                       notifier=self.notifier, screen_no=no,
                                       screen_roi=roi, screen_total=total,
                                       event_type=typ,
                                       label=TYPE_LABELS.get(typ, typ.upper()))
                enrich_event(ev, no)          # 附加设备归因与 GTMP 任务上下文
                with self.lock:
                    self.events.append(ev)
                    del self.events[:-MAX_EVENTS_KEPT]
            except Exception as e:
                print(f"⚠️ 事件生成失败: {e}")

        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------- 对外操作
    def snapshot_jpeg(self):
        with self.lock:
            return self._jpeg

    def snapshot_raw(self):
        """未标注的原始帧（调阈值用，标注版会被告警蒙版污染）。"""
        with self.lock:
            return None if self._raw is None else self._raw.copy()

    def calibrate(self):
        """用滚动缓存的最大亮度图自动标定多块屏幕。"""
        if not HAVE_DETECTOR:
            raise RuntimeError("未找到 analyze_black_screens.py，无法自动标定")
        if not self._calib_buf:
            raise RuntimeError("标定缓存为空，等几秒再试")
        mb = self._calib_buf[0].copy()
        for g in self._calib_buf:
            mb = np.maximum(mb, g)
        rois = rois_from_maxbright(mb, max_screens=self.max_screens)
        self.set_rois(rois)
        return rois

    def set_rois(self, rois):
        rois = order_rois([tuple(int(v) for v in r) for r in rois])[:self.max_screens]
        with self.lock:
            self.rois = rois
            self.results = []
            self._screen_state.clear()
        # 闪屏/冻屏检测器带历史，换了 ROI 必须重建，否则拿旧屏历史判新屏
        self.bank.set_rois(rois)
        save_rois(rois, self.frame_size())
        return rois

    def frame_size(self):
        """当前采集分辨率，用于把 ROI 与分辨率绑定保存。"""
        with self.lock:
            return (int(self.info.get("w") or 0), int(self.info.get("h") or 0))

    def set_detect(self, on):
        with self.lock:
            self.detect_on = bool(on) and HAVE_DETECTOR
            if not self.detect_on:
                self.results = []
        return self.detect_on

    def _reprobe(self, prop):
        """重探单个控制项的范围，就地更新缓存。"""
        for i, sp in enumerate(self.param_specs()):
            if sp["prop"] != prop:
                continue
            rng = probe_param_range(self.cap, prop)
            if rng:
                lo, hi = rng
                self._param_specs[i] = {"name": sp["name"].replace(" -n", ""),
                                        "prop": prop, "min": lo, "max": hi,
                                        "conv": None}
            return

    def param_specs(self):
        """相机控制项表；首次访问时向驱动探真实范围并缓存。

        探测要写超界值再还原，会瞬时改到相机，所以只在首次访问和换设备后做。
        """
        if self._param_specs is None:
            self._param_specs = build_param_specs(self.cap)
        return self._param_specs

    def camera_params(self):
        out = []
        for sp in self.param_specs():
            cur, conv = self.cap.get(sp["prop"]), sp["conv"]
            if conv is None:
                val = int(min(max(cur, sp["min"]), sp["max"]))
            else:
                val = int(min(max(cur if conv(1) >= 0 else -cur, 0), sp["max"]))
            out.append({"name": sp["name"], "value": val,
                        "min": sp["min"], "max": sp["max"]})
        ae = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        out.append({"name": "AutoExp 0/1", "value": 0 if ae in (0, 1, 0.25) else 1, "max": 1})
        with self.lock:
            zoom, (px, py) = self.zoom, list(self.pan)
        out.append({"name": "Zoom", "value": zoom, "min": ZOOM_MIN, "max": ZOOM_MAX,
                    "step": 10, "div": 100, "digits": 2, "unit": "x"})
        out.append({"name": "Pan X", "value": px, "max": 100, "unit": "%",
                    "needs_zoom": True})
        out.append({"name": "Pan Y", "value": py, "max": 100, "unit": "%",
                    "needs_zoom": True})
        return out

    def set_camera_param(self, name, value):
        if name == "AutoExp 0/1":
            set_auto_exposure(self.cap, bool(int(value)))
            # 自动曝光开着时 Exposure 控制项是 inactive，启动时探不到范围，只能
            # 退回写死量程。关掉自动曝光后它才可写，这里重探一次，否则曝光滑条
            # 一直是错的量程（V4L2 实际 1..10000，写死量程是 0..13 取负）
            self._reprobe(cv2.CAP_PROP_EXPOSURE)
            return True
        if name in ("Zoom", "Pan X", "Pan Y"):
            v = int(value)
            with self.lock:
                if name == "Zoom":
                    self.zoom = min(max(v, ZOOM_MIN), ZOOM_MAX)
                else:
                    self.pan[0 if name == "Pan X" else 1] = min(max(v, 0), 100)
            return True
        for sp in self.param_specs():
            if sp["name"] == name:
                conv = sp["conv"]
                v = int(min(max(int(value), sp["min"]), sp["max"]))
                self.cap.set(sp["prop"], v if conv is None else conv(v))
                return True
        return False

    def status(self):
        with self.lock:
            results, rois = list(self.results), list(self.rois)
            multi = list(self.multi)
            n_events = len(self.events)
        # 每块屏每种类型的实时分数：正常时 _flatten 会把类型丢掉，只剩一个 ok，
        # 光看状态表分不清"检测器在跑但离阈值还远"和"这个检测器压根没跑"
        by_screen = {no: {t: {"score": float(r.get("score") or 0.0),
                              "abnormal": bool(r.get("abnormal")),
                              "info": r.get("info") or "",
                              "short": TYPE_SHORT.get(t, t.upper()[:6])}
                          for t, r in per_type.items()}
                     for no, _roi, per_type in multi}
        screens = []
        for no, roi, res in results:
            hit = res.get("abnormal") and (res.get("label") or "")
            screens.append({
                "no": no,
                "roi": list(roi) if roi else None,
                "type": (res.get("type") or ""),
                "label": res.get("label") or "",
                "abnormal": bool(res.get("abnormal")),
                "score": float(res.get("score") or 0.0),
                "info": res.get("info") or "",
                "scores": by_screen.get(no, {}),
                "dark_pct": round(float((res.get("region") or {}).get("dark_pct", 0.0)), 1),
            })
        return {
            "device": self.info.get("idx"),
            "types": list(self.types),
            "type_labels": {t: TYPE_LABELS.get(t, t) for t in self.types},
            "zoom": self.zoom,
            "camera_online": self.cam_online,
            "camera_status": self.cam_status,
            "camera_reconnects": self.reconnects,
            "backend": self.info.get("backend"),
            "width": int(self.info.get("w") or 0),
            "height": int(self.info.get("h") or 0),
            "fps": round(self.fps, 1),
            "detect_on": self.detect_on,
            "max_screens": self.max_screens,
            "rois": [list(r) for r in rois],
            "roi_arg": format_rois(rois),
            "screens": screens,
            "event_count": n_events,
            "have_detector": HAVE_DETECTOR,
        }


def enrich_event(ev, screen_no):
    """给事件补上设备侧归因与 GTMP 任务上下文，并回写 event.json。

    相机只能看到"黑了"，判不出是 GTMP 正常重启还是真故障；
    这里把设备状态 / logcat / 该屏 framebuffer 一并落进证据里。
    """
    import json as _json
    try:
        if device_ctx:
            # 相机第 N 块屏 -> 设备第 N 块屏（两边都按物理顺序编号）
            did = None
            if device_screen and screen_no:
                with device_screen.lock:
                    ds = list(device_screen.displays)
                if 0 < screen_no <= len(ds):
                    did = ds[screen_no - 1]["display_id"]
            # camera_dark 只对黑屏成立：花屏/闪屏事件里相机看到的是花的不是黑的，
            # 硬传 True 会让归因输出"设备端正常但相机看到黑"这种不成立的结论
            ev["device_context"] = device_ctx.classify(
                display_id=did, camera_dark=(ev.get("event_type") == BLACK))
            ev["device_state"] = device_ctx.status_dict()
        if gtmp:
            ev["gtmp"] = gtmp.snapshot()
        shot = Path(ev["screenshot"])
        (shot.parent / "event.json").write_text(
            _json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
        v = (ev.get("device_context") or {}).get("verdict")
        if v:
            normal = (ev["device_context"] or {}).get("is_normal")
            tag = "正常行为" if normal else ("疑似故障" if normal is False else "无法判断")
            print(f"     归因: {v} ({tag}) — {ev['device_context']['reason']}")
    except Exception as e:
        print(f"⚠️ 事件归因失败: {e}")


# ---------------------------------------------------------------- HTTP
service: MonitorService = None
device_screen: DeviceScreenManager = None      # 设备多屏投屏（无设备时降级为"未连接"）
device_ctx: DeviceContext = None  # 设备状态 + logcat，用于黑屏归因
gtmp: GtmpLink = None             # GTMP 任务上下文（可选）
server = None                     # uvicorn.Server，供 /stream 感知退出信号
app = FastAPI(title="Screen Anomaly Monitor")


def _should_stop():
    """服务要退出了吗。uvicorn 优雅关闭会先等所有连接结束，而 /stream 是
    无限生成器；不看 server.should_exit 的话，浏览器挂着流时 SIGTERM 会永久
    挂住，只能 kill -9。"""
    return (not service) or (not service.running) or (server is not None and server.should_exit)


class RoisIn(BaseModel):
    rois: list


class ParamIn(BaseModel):
    name: str
    value: int


class DetectIn(BaseModel):
    on: bool


class NotifyIn(BaseModel):
    on: bool


class InputIn(BaseModel):
    action: str                       # tap / swipe / key / text
    x: float = None                   # 归一化 0~1，浏览器画面是缩放的
    y: float = None
    x2: float = None
    y2: float = None
    duration_ms: int = 200
    key: str = None
    text: str = None


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/stream")
async def stream():
    """相机画面 MJPEG 流，浏览器 <img src="/stream"> 直接显示。"""
    return _mjpeg(lambda: service.snapshot_jpeg() if service else None)


@app.get("/api/status")
def api_status():
    return service.status()


def _notify_state():
    """飞书告警的可用性与开关状态，供网页按钮渲染。"""
    try:
        from notify.feishu_notifier import config_status
        cfg = config_status()
    except Exception as e:                      # noqa: BLE001
        cfg = {"mode": None, "ready": False, "target": None, "problem": str(e)}
    available = bool(service and service._notify_fn) and cfg["ready"]
    return {"available": available, "on": bool(service and service.notify_on),
            "mode": cfg["mode"], "target": cfg["target"], "problem": cfg["problem"]}


@app.get("/api/notify")
def api_notify_get():
    return _notify_state()


@app.post("/api/notify")
def api_notify_set(body: NotifyIn):
    st = _notify_state()
    if body.on and not st["available"]:
        raise HTTPException(409, st["problem"] or "飞书告警未配置")
    service.set_notify(body.on)
    return _notify_state()


@app.post("/api/notify/test")
def api_notify_test():
    """发一条测试消息，确认配置真的能送达（不依赖开关，也不产生事件）。"""
    st = _notify_state()
    if not st["available"]:
        raise HTTPException(409, st["problem"] or "飞书告警未配置")
    from notify.feishu_notifier import send_text
    try:
        send_text(f"✅ 屏幕异常监控测试消息 "
                  f"{datetime.now():%Y-%m-%d %H:%M:%S} · 相机 {service.info.get('idx')}")
    except Exception as e:                      # noqa: BLE001
        raise HTTPException(502, f"发送失败: {e}")
    return {"ok": True, **st}


@app.get("/api/rois")
def api_get_rois():
    st = service.status()
    return {"rois": st["rois"], "roi_arg": st["roi_arg"], "max_screens": st["max_screens"]}


@app.post("/api/rois")
def api_set_rois(body: RoisIn):
    bad = [r for r in body.rois if len(r) != 4]
    if bad:
        raise HTTPException(400, "每块 ROI 需为 [x, y, w, h]")
    rois = service.set_rois(body.rois)
    return {"rois": [list(r) for r in rois], "roi_arg": format_rois(rois)}


@app.delete("/api/rois")
def api_clear_rois():
    service.set_rois([])
    return {"rois": [], "roi_arg": ""}


@app.post("/api/rois/calibrate")
def api_calibrate():
    try:
        rois = service.calibrate()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    if not rois:
        raise HTTPException(422, "自动标定失败：屏幕未点亮或亮区过小，请手动框选")
    return {"rois": [list(r) for r in rois], "roi_arg": format_rois(rois)}


@app.get("/api/camera")
def api_get_camera():
    return {"params": service.camera_params()}


@app.post("/api/camera")
def api_set_camera(body: ParamIn):
    if not service.set_camera_param(body.name, body.value):
        raise HTTPException(404, f"未知参数: {body.name}")
    return {"ok": True, "params": service.camera_params()}


@app.post("/api/detect")
def api_detect(body: DetectIn):
    return {"detect_on": service.set_detect(body.on)}


@app.get("/api/events")
def api_events(limit: int = 50):
    with service.lock:
        evs = list(service.events)[-max(1, min(limit, MAX_EVENTS_KEPT)):]
    return {"events": list(reversed(evs))}


@app.get("/api/events/{event_id}/screenshot")
def api_event_shot(event_id: str):
    with service.lock:
        match = next((e for e in service.events if e["event_id"] == event_id), None)
    if not match:
        raise HTTPException(404, "事件不存在")
    p = Path(match["screenshot"])
    if not p.exists():
        raise HTTPException(404, "截图文件已被清理")
    return FileResponse(p, media_type="image/jpeg")


# ---------------------------------------------------------------- 设备投屏
def _mjpeg(get_jpeg):
    """把"取一帧 JPEG"的回调包装成 MJPEG 多部件流。"""
    boundary = "frame"

    async def gen():
        last = None
        try:
            while not _should_stop():
                jpg = get_jpeg()
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield (b"--" + boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                           + jpg + b"\r\n")
                await asyncio.sleep(0.04)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


class DeviceSelectIn(BaseModel):
    index: int


@app.post("/api/device_screen/select")
def api_device_select(body: DeviceSelectIn):
    """切换正在投的那块屏。同一时刻只投一路，切换即停旧开新。"""
    if not device_screen:
        raise HTTPException(409, "设备投屏未启用")
    try:
        idx = device_screen.select(body.index)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "active": idx}


@app.get("/device_stream/{index}")
async def device_stream(index: int):
    """第 index 块设备屏的 MJPEG 流（index 从 0 起，与相机 S1/S2/S3 顺序对应）。"""
    if not device_screen:
        raise HTTPException(409, "设备投屏未启用")
    return _mjpeg(lambda: (lambda s: s.snapshot_jpeg() if s else None)(device_screen.get(index)))


@app.get("/api/device_screen")
def api_device_screen():
    if not device_screen:
        return {"enabled": False, "connected": False, "screens": [],
                "status": "未启用（--no-device-screen）"}
    return {"enabled": True, **device_screen.status_dict()}


@app.post("/api/device_screen/restart")
def api_device_restart():
    if not device_screen:
        raise HTTPException(409, "设备投屏未启用")
    return {"ok": True, "restarted": device_screen.restart_all()}


@app.post("/api/device_screen/{index}/input")
def api_device_input(index: int, body: InputIn):
    """把浏览器里的点击/滑动/按键打到对应那块设备屏。"""
    if not (device_screen and device_screen.get(index)):
        raise HTTPException(404, f"第 {index + 1} 路投屏不存在")
    svc = device_screen.get(index)
    d = svc.display
    lid, nw, nh = d.get("logical_id"), d.get("width") or 0, d.get("height") or 0
    if lid is None:
        raise HTTPException(409, f"{d.get('name')} 未解析到逻辑 displayId，无法注入输入")

    def px(v, span):
        return int(max(0, min(1.0, float(v))) * span)

    try:
        if body.action == "tap":
            payload = {"x": px(body.x, nw), "y": px(body.y, nh)}
        elif body.action == "swipe":
            payload = {"x1": px(body.x, nw), "y1": px(body.y, nh),
                       "x2": px(body.x2, nw), "y2": px(body.y2, nh),
                       "duration_ms": body.duration_ms}
        elif body.action == "key":
            payload = {"key": body.key}
        elif body.action == "text":
            payload = {"text": body.text}
        else:
            raise HTTPException(400, f"不支持的操作: {body.action}")
        cmd = send_input(device_screen.active_serial, lid, body.action, payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"注入失败: {e}")
    return {"ok": True, "display": d.get("name"), "logical_id": lid, "sent": cmd}


@app.get("/api/device_context")
def api_device_context():
    """设备实时状态（uptime / 开机完成 / 各屏电源态）与最近关键日志。"""
    if not device_ctx:
        return {"enabled": False}
    return {"enabled": True, **device_ctx.status_dict(),
            "recent_logs": device_ctx.recent_logs(window_s=120, limit=30),
            "gtmp": gtmp.snapshot() if gtmp else None}


@app.get("/api/snapshot")
def api_snapshot(raw: int = 0):
    """当前帧单张 JPEG，便于外部脚本抓图或做健康检查。

    raw=1 取未标注的原始帧：调阈值时必须用它，标注版的红色告警蒙版会把
    ROI 内的亮度整体抬高，拿它量出来的阈值是错的。
    """
    if raw:
        frame = service.snapshot_raw()
        if frame is None:
            raise HTTPException(503, "尚无画面")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise HTTPException(500, "编码失败")
        return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")
    jpg = service.snapshot_jpeg()
    if jpg is None:
        raise HTTPException(503, "尚无画面")
    return StreamingResponse(iter([jpg]), media_type="image/jpeg")


# ---------------------------------------------------------------- main
def main():
    global service, device_screen, device_ctx, gtmp
    ap = argparse.ArgumentParser(description="屏幕异常实时监控 Web 服务")
    ap.add_argument("--device", type=int, default=None, help="相机 index，缺省则自动诊断")
    ap.add_argument("--source",
                    help="网络视频流地址，替代本地相机。例：远端 ustreamer 经 ssh "
                         "隧道映射到本机后 http://127.0.0.1:18080/stream。"
                         "给了 --source 就忽略 --device 与自动诊断")
    ap.add_argument("--instance", default="local",
                    help="实例名（默认 local）。多路相机同机并存时用它隔离 ROI、"
                         "飞书开关与事件目录；默认实例沿用原有文件名")
    ap.add_argument("--screens", type=int, default=3, help="最多识别几块屏幕（默认3）")
    ap.add_argument("--roi", help="启动时固定 ROI: x,y,w,h 多块用分号分隔")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址（默认全网卡）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-detect", action="store_true", help="只看画面，不做检测")
    ap.add_argument("--notify", action="store_true", help="事件推送飞书（需 FEISHU_WEBHOOK）")
    ap.add_argument("--quality", type=int, default=75, help="MJPEG 质量 1-100（默认75）")
    ap.add_argument("--no-device-screen", action="store_true",
                    help="不启用 Android 设备投屏（默认自动探测 adb 设备）")
    ap.add_argument("--adb-serial", help="指定 adb 设备序列号（多台设备时）")
    ap.add_argument("--device-record-max", type=int, default=2,
                    help="同时用 screenrecord 的投屏路数上限（默认2）。实测该类车机"
                         "三路并发会串台，超出的屏自动改用 screencap 轮询（慢但正确）")
    ap.add_argument("--device-displays",
                    help="手动指定投屏的 display-id 及顺序，逗号分隔；"
                         "缺省按设备 port 顺序自动枚举")
    ap.add_argument("--device-bitrate", default="4M", help="投屏码率（默认4M）")
    ap.add_argument("--device-stream-mode", choices=("auto", "screencap"), default="auto",
                    help="auto=先用 screenrecord 并持续校验，串台自动降级；"
                         "screencap=直接用轮询（慢但从第一帧就保证对得上）")
    ap.add_argument("--no-device-context", action="store_true",
                    help="不采集设备状态与 logcat（默认开启，用于黑屏归因）")
    ap.add_argument("--types", default=BLACK,
                    help="启用的异常类型，逗号分隔或 all。默认只开 black_screen——"
                         "实测台架待机时 screen_freeze 会在所有屏 100% 命中"
                         "（待机画面本就静止），white_screen 也会在白底页面上常驻，"
                         "其余类型请按台架实际情况按需开启。可选: "
                         + ", ".join(ALL_TYPES))
    ap.add_argument("--width", type=int, default=1280,
                    help="固定采集宽度（默认1280；ROI 与分辨率绑定，别随意改）")
    ap.add_argument("--height", type=int, default=720, help="固定采集高度（默认720）")
    ap.add_argument("--gtmp-task", type=int, help="关联 GTMP 任务 ID，事件附带任务上下文")
    ap.add_argument("--gtmp-bench", type=int, help="关联 GTMP 台架 ID，自动跟踪其运行中的任务")
    args = ap.parse_args()

    configure_instance(args.instance)      # 必须早于 load_rois / load_state
    is_stream = bool(args.source)
    if is_stream:
        print(f"🌐 视频流源: {args.source}")
        opened = open_stream(args.source)
    elif args.device is not None:
        opened = open_device(args.device)
    else:
        opened = diagnose_all()
    if not opened:
        print("❌ 打不开%s，Web 服务未启动" % ("视频流" if is_stream else "相机"))
        return 1
    cap, info = opened

    # 固定采集分辨率：诊断流程可能协商到 640x480 等其它档位，而 ROI 是像素坐标，
    # 分辨率一变就整体错位并立刻误报黑屏。这里统一钉死，取不到就退回实际值。
    want_w, want_h = (None, None) if is_stream else (args.width, args.height)
    if want_w and want_h and (int(info.get("w") or 0), int(info.get("h") or 0)) != (want_w, want_h):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
        for _ in range(5):
            cap.read()
        info["w"], info["h"] = int(cap.get(3)), int(cap.get(4))
        if (info["w"], info["h"]) != (want_w, want_h):
            print(f"⚠️ 相机不支持 {want_w}x{want_h}，实际使用 {info['w']}x{info['h']}")
    print(f"✅ {'视频流' if is_stream else '相机'}就绪: idx={info.get('idx')} "
          f"{info.get('backend')} {info.get('w')}x{info.get('h')}")
    # 网络流源默认关掉 adb 投屏：DeviceScreenManager 探测的是【本机】USB 上的
    # 设备，会把本地车机的画面投到远端相机的页面上，看着像串台，极难排查。
    if is_stream and not args.no_device_screen:
        args.no_device_screen = True
        print("ℹ️ 视频流源已自动关闭 adb 投屏（投屏只对本机 USB 设备有意义）")

    rois = parse_rois(args.roi) if args.roi else load_rois((info.get("w"), info.get("h")))
    if rois:
        print(f"🖥️ 载入 ROI ({len(rois)} 块): {format_rois(rois)}")

    # 发送函数总是加载，网页上随时可开关；--notify 只决定启动时的初值
    notifier, notify_cfg = None, None
    try:
        from notify.feishu_notifier import config_status, send_event
        notifier, notify_cfg = send_event, config_status()
    except Exception as e:                      # noqa: BLE001
        print(f"⚠️ 飞书告警模块不可用: {e}")
    if notify_cfg and notify_cfg["ready"]:
        on0 = args.notify or load_state().get("notify_on", False)
        print(f"📨 飞书告警: {'开' if on0 else '关（网页上可随时打开）'}"
              f" · {notify_cfg['mode']} → {notify_cfg['target']}")
    elif notify_cfg:
        print(f"📨 飞书告警: 未配置 — {notify_cfg['problem']}")

    service = MonitorService(cap, info, max_screens=args.screens, rois=rois,
                             detect=not args.no_detect, notifier=notifier,
                             jpeg_quality=args.quality, types=args.types,
                             # --notify 强制开；否则沿用网页上次的选择
                             notify_on=args.notify or load_state().get("notify_on", False))
    print("🔎 启用检测: " + ", ".join(
        f"{TYPE_LABELS.get(t, t)}({t})" for t in service.types))
    service.start()

    if not args.no_device_screen:
        def camera_screen_count():
            """相机当前能检测几块屏，就开几路投屏（未标定 ROI 时用 --screens）。"""
            with service.lock:
                return len(service.rois) or service.max_screens

        device_screen = DeviceScreenManager(
            serial=args.adb_serial, bitrate=args.device_bitrate,
            target_count=camera_screen_count,
            display_ids=[d.strip() for d in args.device_displays.split(",") if d.strip()]
                        if args.device_displays else None,
            force_mode=None if args.device_stream_mode == "auto" else args.device_stream_mode,
            record_max=args.device_record_max)
        device_screen.start()
        serial, model = pick_device(args.adb_serial)
        print(f"📱 设备投屏: {'已连接 ' + (model or serial) if serial else '未检测到 adb 设备（插上后会自动重连）'}"
              f"，同一时刻只投一块屏，可在网页上切换")

    if not args.no_device_context:
        device_ctx = DeviceContext(serial=args.adb_serial)
        device_ctx.start()
        print("🩺 设备归因: 开（uptime/开机标志/各屏电源态 + logcat 滚动缓冲）")

    if args.gtmp_task or args.gtmp_bench:
        gtmp = GtmpLink(task_id=args.gtmp_task, bench_id=args.gtmp_bench)
        gtmp.start()
        print(f"🔗 GTMP: 关联 {'任务 ' + str(args.gtmp_task) if args.gtmp_task else '台架 ' + str(args.gtmp_bench)}")

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); lan = s.getsockname()[0]; s.close()
    except OSError:
        lan = "127.0.0.1"
    print(f"\n🌐 打开浏览器访问:  http://{lan}:{args.port}/   (本机 http://127.0.0.1:{args.port}/)")
    print("   相机被本服务独占，运行期间不要再开 camera_diag.py 预览\n")

    import uvicorn
    global server
    # 自建 Server 以便 /stream 读到 should_exit；再加优雅关闭超时兜底
    server = uvicorn.Server(uvicorn.Config(
        app, host=args.host, port=args.port, log_level="warning",
        timeout_graceful_shutdown=5))
    try:
        server.run()
    finally:
        service.stop()
        if device_screen:
            device_screen.stop()
        if device_ctx:
            device_ctx.stop()
        if gtmp:
            gtmp.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
