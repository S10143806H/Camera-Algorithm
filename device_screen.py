"""
设备投屏采集 —— 把 Android 设备的每块物理屏实时取到本地
=======================================================
用 `adb exec-out screenrecord --display-id ID` 把指定显示屏以 H.264 裸流写进
FIFO，再用 OpenCV（自带 FFmpeg）解码，产出与相机同格式的 JPEG 帧。

选这条路而不是 scrcpy：只依赖 adb，无需 `apt install scrcpy`、无需
`modprobe v4l2loopback`、无需系统 ffmpeg，因此不需要 root。

车机常有多块屏（中控 / 仪表 / 后排）。DeviceScreenManager 按相机当前能检测的
屏幕数决定开几路投屏，相机侧重新标定后自动增减，两边编号一一对应。
"""

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

RETRY_S = 3.0            # 设备不在 / 出错时的重试间隔
# 并发跑多个 screenrecord 时，实测该车机各路会互相串台：某一路拿到的其实是
# 另一块屏的画面（甚至字节完全相同）。只比亮度抓不住——两块屏都亮时看不出差别，
# 所以改用结构相关度：把 screenrecord 帧与该屏 screencap 都降采样归一化后求相关，
# 同一块屏应接近 1，串台时会掉到 0.3 以下。
FALLBACK_CORR_MIN = 0.40
FALLBACK_STRIKES = 2          # 连续几次校验不过才切，避免画面变化引起的误判
VERIFY_INTERVAL_S = 15.0      # screenrecord 模式下周期性复检；串台是运行中才发生的，
                              # 间隔越短，显示错误画面的窗口越小（每次仅一张 screencap 开销）
SCREENCAP_INTERVAL_S = 0.45
MAX_LONG_EDGE = 1280     # 投屏长边上限，按原始宽高比缩放后再送编码器
SYNC_S = 2.0             # 管理器检查"该开几路"的间隔


# ---------------------------------------------------------------- adb 查询
def _adb(serial, *args, timeout=15):
    cmd = ["adb"] + (["-s", serial] if serial else []) + list(args)
    return subprocess.check_output(cmd, text=True, timeout=timeout,
                                   stderr=subprocess.DEVNULL)


def list_devices():
    """返回 [(serial, 状态, model), ...]；adb 不可用时返回 []。"""
    if not shutil.which("adb"):
        return []
    try:
        out = _adb(None, "devices", "-l", timeout=10)
    except Exception:
        return []
    devices = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            model = next((p.split(":", 1)[1] for p in parts if p.startswith("model:")), "")
            devices.append((parts[0], parts[1], model))
    return devices


def pick_device(serial=None):
    """挑一台在线设备。指定 serial 时只认它。"""
    for s, state, model in list_devices():
        if state != "device":
            continue
        if serial is None or s == serial:
            return s, model
    return None, ""


def list_displays(serial=None):
    """列出设备的物理显示屏，按 port 排序（与实际接线顺序一致，编号稳定）。

    返回 [{"display_id", "port", "name", "width", "height"}, ...]。
    虚拟屏（录屏器等 type=VIRTUAL）会被排除——它不是真实存在的一块屏。
    """
    try:
        sf = _adb(serial, "shell", "dumpsys", "SurfaceFlinger", "--display-id")
    except Exception:
        return []

    displays = []
    for m in re.finditer(r"^Display (\d+) \(HWC display \d+\): port=(\d+).*?displayName=\"([^\"]*)\"",
                         sf, re.M):
        displays.append({"display_id": m.group(1), "port": int(m.group(2)),
                         "name": m.group(3), "width": 0, "height": 0,
                         "logical_id": None})
    if not displays:
        return []

    # 从 dumpsys display 补上人类可读名与分辨率（按 uniqueId 里的 display_id 关联）
    try:
        dd = _adb(serial, "shell", "dumpsys", "display", timeout=20)
        for m in re.finditer(r'DisplayDeviceInfo\{"([^"]+)": uniqueId="local:(\d+)", (\d+) x (\d+)', dd):
            name, did, w, h = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            for d in displays:
                if d["display_id"] == did:
                    d["name"], d["width"], d["height"] = name, w, h
        # input/am 用的是逻辑 displayId（0/1/2），与 SurfaceFlinger 的物理 id 不同，
        # 从 DisplayViewport 里把两者对上，否则输入会全部落到主屏
        for m in re.finditer(r"displayId=(\d+), uniqueId='local:(\d+)'", dd):
            for d in displays:
                if d["display_id"] == m.group(2):
                    d["logical_id"] = int(m.group(1))
    except Exception:
        pass

    displays.sort(key=lambda d: d["port"])
    return displays


def screencap_frame(serial, display_id, timeout=20):
    """用 screencap 抓某块屏的单帧（BGR ndarray）。失败返回 None。

    screencap 按 display-id 逐屏读取始终正确，是校验 screenrecord 的基准。
    """
    try:
        raw = subprocess.check_output(
            ["adb"] + (["-s", serial] if serial else []) +
            ["exec-out", "screencap", "-d", str(display_id), "-p"],
            timeout=timeout, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    i = raw.find(b"\x89PNG")          # 部分机型会在 PNG 前打印告警
    if i < 0:
        return None
    return cv2.imdecode(np.frombuffer(raw[i:], np.uint8), cv2.IMREAD_COLOR)


def scaled_size(width, height, long_edge=MAX_LONG_EDGE):
    """按原始宽高比缩到长边不超过 long_edge，并对齐到偶数（AVC 编码器要求）。

    车机仪表屏常是 1920x480 这类超宽比例，统一压成固定尺寸会拉变形。
    """
    if width <= 0 or height <= 0:
        return None
    scale = min(1.0, long_edge / max(width, height))
    w = max(16, int(width * scale) // 2 * 2)
    h = max(16, int(height * scale) // 2 * 2)
    return f"{w}x{h}"


# ---------------------------------------------------------------- 输入回传
KEYCODES = {
    "HOME": "KEYCODE_HOME", "BACK": "KEYCODE_BACK", "MENU": "KEYCODE_MENU",
    "POWER": "KEYCODE_POWER", "ENTER": "KEYCODE_ENTER", "DEL": "KEYCODE_DEL",
    "VOLUME_UP": "KEYCODE_VOLUME_UP", "VOLUME_DOWN": "KEYCODE_VOLUME_DOWN",
    "APP_SWITCH": "KEYCODE_APP_SWITCH", "WAKEUP": "KEYCODE_WAKEUP",
}


def send_input(serial, logical_id, action, payload):
    """把浏览器里的操作转成 adb input 打到指定屏。

    坐标以归一化 0~1 传入，再乘该屏原生宽高——浏览器里画面是缩放显示的，
    直接传像素会错位。logical_id 为 None 时不带 -d，落到主屏。
    """
    if logical_id is None and action in ("tap", "swipe"):
        raise ValueError("该屏没有解析到逻辑 displayId，无法定向注入输入")
    base = ["adb"] + (["-s", serial] if serial else []) + ["shell", "input"]
    if logical_id is not None:
        base += ["-d", str(logical_id)]

    if action == "tap":
        cmd = base + ["tap", str(payload["x"]), str(payload["y"])]
    elif action == "swipe":
        cmd = base + ["swipe", str(payload["x1"]), str(payload["y1"]),
                      str(payload["x2"]), str(payload["y2"]),
                      str(int(payload.get("duration_ms", 200)))]
    elif action == "key":
        name = str(payload.get("key", "")).upper()
        cmd = base + ["keyevent", KEYCODES.get(name, name)]
    elif action == "text":
        # input text 不接受空格与部分符号，转义后再发
        txt = str(payload.get("text", "")).replace(" ", "%s")
        cmd = base + ["text", txt]
    else:
        raise ValueError(f"不支持的操作: {action}")

    subprocess.run(cmd, timeout=20, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return " ".join(cmd[-4:])


# ---------------------------------------------------------------- 单屏投屏
class DeviceScreenService:
    """后台线程持续拉某一块显示屏，产出 JPEG 供 HTTP 取用。"""

    def __init__(self, serial, display, bitrate="4M", jpeg_quality=70,
                 force_mode=None):
        self.serial = serial
        self.display = display              # list_displays() 的一项
        self.bitrate = bitrate
        self.jpeg_quality = jpeg_quality

        self.lock = threading.Lock()
        self._jpeg = None
        self.running = True
        self.connected = False
        self.status = "启动中"
        # force_mode="screencap" 时直接用轮询，不给串台留任何窗口
        self.mode = force_mode or "screenrecord"
        self.force_mode = force_mode
        self.last_corr = None           # 最近一次结构校验的相关度
        self._strikes = 0
        self.width = self.height = 0
        self.fps = 0.0
        self._proc = None
        self._tmpdir = Path(tempfile.mkdtemp(prefix="devscr_"))
        self._fifo = self._tmpdir / "screen.h264"
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ------------------------------------------------------------ 生命周期
    def start(self):
        self._thread.start()

    def stop(self):
        self.running = False
        self._kill_proc()
        self._thread.join(timeout=5)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def snapshot_jpeg(self):
        with self.lock:
            return self._jpeg

    def restart(self):
        """外部触发重连：回到 screenrecord 并重新校验，杀掉录制进程后主循环自动重开。"""
        self.mode = self.force_mode or "screenrecord"
        self._strikes = 0
        self._kill_proc()
        return True

    # ------------------------------------------------------------ 内部
    def _kill_proc(self):
        p, self._proc = self._proc, None
        if p and p.poll() is None:
            try:
                p.terminate(); p.wait(timeout=3)
            except Exception:
                try: p.kill()
                except Exception: pass

    def _set_status(self, text, connected=None):
        with self.lock:
            self.status = text
            if connected is not None:
                self.connected = connected

    def _loop(self):
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self.running:
            try:
                if self.mode == "screencap":
                    self._run_screencap(enc)
                else:
                    self._run_once(enc)
            except Exception as e:
                self._set_status(f"投屏出错: {e}", connected=False)
            if self.running:
                time.sleep(RETRY_S)

    @staticmethod
    def _signature(img, n=64):
        """降采样 + 零均值单位方差归一化，用于跨分辨率比对画面结构。"""
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (n, n),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-6)

    def _verify_not_fallback(self, frame):
        """用 screencap 校验 screenrecord 拿到的确实是这块屏。

        比结构而非亮度：并发 screenrecord 串台时，两块屏可能都是亮的，
        只看亮度分不出来；结构相关度能直接判出"这画面根本不是这块屏"。
        """
        ref = screencap_frame(self.serial, self.display["display_id"])
        if ref is None:
            return True                 # 校验不了就不误判
        corr = float((self._signature(frame) * self._signature(ref)).mean())
        with self.lock:
            self.last_corr = round(corr, 3)
        if corr >= FALLBACK_CORR_MIN:
            self._strikes = 0
            return True
        self._strikes += 1
        if self._strikes < FALLBACK_STRIKES:
            return True                 # 可能只是画面变化，再看一次
        print(f"⚠️ display {self.display['display_id']} ({self.display.get('name')}): "
              f"screenrecord 画面与该屏 screencap 结构相关度仅 {corr:.2f}"
              f"（连续 {self._strikes} 次），判定为串台/回落，改用 screencap 轮询")
        return False

    def _run_screencap(self, enc):
        """screencap 轮询模式：帧率低但每块屏都读得对。"""
        self._set_status("投屏中（screencap 模式）", connected=True)
        n, t_fps = 0, time.time()
        while self.running:
            frame = screencap_frame(self.serial, self.display["display_id"])
            if frame is None:
                self._set_status("screencap 抓图失败", connected=False)
                return
            size = scaled_size(frame.shape[1], frame.shape[0])
            if size:
                w, h = (int(v) for v in size.split("x"))
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            with self.lock:
                self.height, self.width = frame.shape[:2]
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if ok:
                with self.lock:
                    self._jpeg = buf.tobytes()
            n += 1
            now = time.time()
            if now - t_fps >= 1.0:
                with self.lock:
                    self.fps = n / (now - t_fps)
                n, t_fps = 0, now
            time.sleep(SCREENCAP_INTERVAL_S)

    def _run_once(self, enc):
        if self._fifo.exists():
            self._fifo.unlink()
        os.mkfifo(self._fifo)

        size = scaled_size(self.display.get("width", 0), self.display.get("height", 0))
        # --time-limit 0 去掉 180 秒硬上限，长时间监控不再有分段重启的画面停顿
        cmd = (f"adb -s {self.serial} exec-out screenrecord --output-format=h264 "
               f"--display-id {self.display['display_id']} "
               + (f"--size {size} " if size else "")
               + f"--bit-rate {self.bitrate} --time-limit 0 - > {self._fifo}")
        self._proc = subprocess.Popen(cmd, shell=True,
                                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        self._set_status("连接中…")
        cap = cv2.VideoCapture(str(self._fifo))
        if not cap.isOpened():
            self._set_status("无法解码该屏画面", connected=False)
            self._kill_proc()
            return

        got_first, n, t_fps = False, 0, time.time()
        next_verify = time.time() + VERIFY_INTERVAL_S
        while self.running:
            ok, frame = cap.read()
            if not ok:
                break
            if not got_first:
                got_first = True
                if not self._verify_not_fallback(frame):
                    self.mode = "screencap"
                    cap.release(); self._kill_proc()
                    return
                with self.lock:
                    self.height, self.width = frame.shape[:2]
                self._set_status("投屏中", connected=True)
                next_verify = time.time() + VERIFY_INTERVAL_S
            elif time.time() >= next_verify:
                # 串台不一定一开始就发生，运行期间也要复检
                next_verify = time.time() + VERIFY_INTERVAL_S
                if not self._verify_not_fallback(frame):
                    self.mode = "screencap"
                    cap.release(); self._kill_proc()
                    return
            ok, buf = cv2.imencode(".jpg", frame, enc)
            if ok:
                with self.lock:
                    self._jpeg = buf.tobytes()
            n += 1
            now = time.time()
            if now - t_fps >= 1.0:
                with self.lock:
                    self.fps = n / (now - t_fps)
                n, t_fps = 0, now

        cap.release()
        self._kill_proc()
        self._set_status("已断开，重连中…", connected=False)

    def status_dict(self, index):
        with self.lock:
            return {
                "index": index,
                "display_id": self.display["display_id"],
                "port": self.display.get("port"),
                "name": self.display.get("name") or f"Display {self.display.get('port')}",
                "native": f"{self.display.get('width')}x{self.display.get('height')}",
                "native_w": self.display.get("width"),
                "native_h": self.display.get("height"),
                "logical_id": self.display.get("logical_id"),
                "can_input": self.display.get("logical_id") is not None,
                "connected": self.connected,
                "status": self.status,
                "mode": self.mode,
                "corr": self.last_corr,
                "width": self.width,
                "height": self.height,
                "fps": round(self.fps, 1),
            }


# ---------------------------------------------------------------- 多屏管理
class DeviceScreenManager:
    """按相机当前能检测的屏幕数决定开几路投屏，设备增减时自动跟随。"""

    def __init__(self, serial=None, bitrate="4M", jpeg_quality=70,
                 target_count=None, display_ids=None, force_mode=None):
        self.serial = serial
        self.bitrate = bitrate
        self.jpeg_quality = jpeg_quality
        self.target_count = target_count or (lambda: 3)   # 由相机侧提供
        self.display_ids = display_ids                    # 手动指定顺序时用
        self.force_mode = force_mode                      # None=自动，"screencap"=强制轮询

        self.lock = threading.Lock()
        self.services = []
        self.displays = []
        self.active_serial = None
        self.model = ""
        self.status = "启动中"
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self.running = False
        with self.lock:
            svcs, self.services = self.services, []
        for s in svcs:
            s.stop()
        self._thread.join(timeout=5)

    def get(self, index):
        with self.lock:
            return self.services[index] if 0 <= index < len(self.services) else None

    def restart_all(self):
        with self.lock:
            svcs = list(self.services)
        for s in svcs:
            s.restart()
        return len(svcs)

    def _stop_all(self, note):
        with self.lock:
            svcs, self.services = self.services, []
            self.status = note
        for s in svcs:
            s.stop()

    def _loop(self):
        while self.running:
            serial, model = pick_device(self.serial)
            if not serial:
                if self.services:
                    self._stop_all("未连接设备（adb devices 为空）")
                else:
                    with self.lock:
                        self.status = "未连接设备（adb devices 为空）"
                with self.lock:
                    self.active_serial, self.model, self.displays = None, "", []
                time.sleep(RETRY_S)
                continue

            if serial != self.active_serial:
                self._stop_all("设备已更换，重新枚举")
                displays = list_displays(serial)
                if self.display_ids:      # 手动指定则按给定顺序过滤
                    order = {d: i for i, d in enumerate(self.display_ids)}
                    displays = sorted((d for d in displays if d["display_id"] in order),
                                      key=lambda d: order[d["display_id"]])
                with self.lock:
                    self.active_serial, self.model, self.displays = serial, model, displays
                if not displays:
                    with self.lock:
                        self.status = "未枚举到物理显示屏"
                    time.sleep(RETRY_S)
                    continue

            self._sync()
            time.sleep(SYNC_S)

    def _sync(self):
        """把投屏路数对齐到 min(设备屏数, 相机能检测的屏数)。"""
        try:
            want = int(self.target_count())
        except Exception:
            want = len(self.displays)
        with self.lock:
            displays = list(self.displays)
            cur = len(self.services)
        want = max(0, min(want, len(displays)))
        if want == cur:
            with self.lock:
                self.status = f"投屏 {cur}/{len(displays)} 块屏"
            return

        if want < cur:                    # 相机侧屏数减少：关掉多余的
            with self.lock:
                extra, self.services = self.services[want:], self.services[:want]
            for s in extra:
                s.stop()
        else:                             # 增加：为新的显示屏开流
            for i in range(cur, want):
                svc = DeviceScreenService(self.active_serial, displays[i],
                                          bitrate=self.bitrate,
                                          jpeg_quality=self.jpeg_quality,
                                          force_mode=self.force_mode)
                svc.start()
                with self.lock:
                    self.services.append(svc)
        with self.lock:
            self.status = f"投屏 {len(self.services)}/{len(displays)} 块屏"

    def status_dict(self):
        with self.lock:
            svcs = list(self.services)
            base = {
                "available": bool(shutil.which("adb")),
                "serial": self.active_serial,
                "model": self.model,
                "status": self.status,
                "display_count": len(self.displays),
                "displays": [dict(d) for d in self.displays],
            }
        base["screens"] = [s.status_dict(i) for i, s in enumerate(svcs)]
        base["connected"] = any(s["connected"] for s in base["screens"])
        return base
