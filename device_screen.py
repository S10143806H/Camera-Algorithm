"""
设备投屏采集 —— 把 Android 设备屏幕实时取到本地
================================================
用 `adb exec-out screenrecord --output-format=h264` 把设备画面以 H.264 裸流
写进 FIFO，再用 OpenCV（自带 FFmpeg）解码，产出与相机同格式的 JPEG 帧。

选这条路而不是 scrcpy 的原因：只依赖 adb，无需 `apt install scrcpy`、
无需 `modprobe v4l2loopback`、无需系统 ffmpeg，因此不需要 root 权限。
效果等价：都是把设备屏幕镜像过来。

screenrecord 单次最长 180 秒，到点自动重启；设备掉线时退到"未连接"
并周期重试，不会让服务崩掉。
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2

SEGMENT_S = 170          # 单段录制时长，留出余量避免撞上 180s 硬上限
RETRY_S = 3.0            # 设备不在时的重试间隔
OPEN_TIMEOUT_S = 20.0    # 等待首帧的上限


def list_devices():
    """返回 [(serial, 状态, model), ...]；adb 不可用时返回 []。"""
    if not shutil.which("adb"):
        return []
    try:
        out = subprocess.check_output(["adb", "devices", "-l"], text=True, timeout=10)
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


class DeviceScreenService:
    """后台线程持续拉设备屏幕，产出 JPEG 供 HTTP 取用。"""

    def __init__(self, serial=None, size="1280x800", bitrate="4M", jpeg_quality=70):
        self.serial = serial
        self.size = size
        self.bitrate = bitrate
        self.jpeg_quality = jpeg_quality

        self.lock = threading.Lock()
        self._jpeg = None
        self.running = True
        self.connected = False
        self.status = "启动中"
        self.model = ""
        self.active_serial = None
        self.width = self.height = 0
        self.fps = 0.0
        self._proc = None
        self._tmpdir = Path(tempfile.mkdtemp(prefix="devscr_"))
        self._fifo = self._tmpdir / "screen.h264"
        self._thread = threading.Thread(target=self._loop, daemon=True)

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
        """外部触发重连：杀掉当前录制进程，主循环会自动重开一段。"""
        self._kill_proc()
        return True

    # ------------------------------------------------------------ 内部
    def _kill_proc(self):
        p, self._proc = self._proc, None
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def _set_status(self, text, connected=None):
        with self.lock:
            self.status = text
            if connected is not None:
                self.connected = connected

    def _loop(self):
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self.running:
            serial, model = pick_device(self.serial)
            if not serial:
                self._set_status("未连接设备（adb devices 为空）", connected=False)
                with self.lock:
                    self.active_serial, self.model, self.fps = None, "", 0.0
                time.sleep(RETRY_S)
                continue

            with self.lock:
                self.active_serial, self.model = serial, model
            self._set_status(f"连接中… ({model or serial})")
            try:
                self._run_segment(serial, enc)
            except Exception as e:
                self._set_status(f"投屏出错: {e}", connected=False)
                time.sleep(RETRY_S)

    def _run_segment(self, serial, enc):
        """录一段并解码。screenrecord 到时限自然结束，返回后外层重开一段。"""
        if self._fifo.exists():
            self._fifo.unlink()
        os.mkfifo(self._fifo)

        cmd = (f"adb -s {serial} exec-out screenrecord --output-format=h264 "
               f"--size {self.size} --bit-rate {self.bitrate} "
               f"--time-limit {SEGMENT_S} - > {self._fifo}")
        # 经 shell 重定向：写端会阻塞到读端打开，父进程不受影响
        self._proc = subprocess.Popen(cmd, shell=True,
                                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        cap = cv2.VideoCapture(str(self._fifo))
        if not cap.isOpened():
            self._set_status("无法解码设备画面（screenrecord 不可用？）", connected=False)
            self._kill_proc()
            time.sleep(RETRY_S)
            return

        t_open, n, t_fps = time.time(), 0, time.time()
        got_first = False
        while self.running:
            ok, frame = cap.read()
            if not ok:
                break                       # 本段结束（到时限或设备断开）
            if not got_first:
                got_first = True
                with self.lock:
                    self.height, self.width = frame.shape[:2]
                self._set_status("投屏中", connected=True)

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
        if not got_first and time.time() - t_open > OPEN_TIMEOUT_S:
            self._set_status("等待设备画面超时", connected=False)
            time.sleep(RETRY_S)

    def status_dict(self):
        with self.lock:
            return {
                "available": bool(shutil.which("adb")),
                "connected": self.connected,
                "status": self.status,
                "serial": self.active_serial,
                "model": self.model,
                "width": self.width,
                "height": self.height,
                "fps": round(self.fps, 1),
                "size": self.size,
                "bitrate": self.bitrate,
            }
