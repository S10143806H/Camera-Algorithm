"""
设备上下文采集与黑屏归因
========================
黑屏事件光有相机画面无法判断是不是问题：GTMP 跑测试时设备会正常重启、正常灭屏，
屏幕本来就该黑。本模块在后台持续采集设备状态与 logcat，事件发生时回溯当时的
上下文，把"正常行为"和"真故障"分开。

判据（按优先级）:
  reboot        设备重启中或刚重启 —— uptime 回退 / adb 断连 / sys.boot_completed=0
  screen_off    该屏电源态不是 ON（灭屏、DOZE）—— 正常灭屏
  device_black  屏电源 ON，但设备端 framebuffer（screencap）就是全黑
                → 合成器/应用侧没出画面，设备自己的问题
  panel_black   设备端 framebuffer 正常，相机却看到黑
                → 屏体 / 背光 / 传输链路故障，设备侧日志看不出来
  unknown       adb 不可用，无法判断

前两类是正常行为，后两类是 bug，且指向完全不同的排查方向。
"""

import collections
import re
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np

LOG_LINES_KEPT = 4000      # 滚动 logcat 缓冲行数
STATE_INTERVAL_S = 1.0     # 设备状态采样间隔
DARK_MEAN = 12.0           # screencap 判黑的平均亮度阈值
REBOOT_WINDOW_S = 90.0     # 事件前后多久内发生过重启就归因为重启

# 与重启 / 灭屏强相关的日志行，事件证据里优先保留这些
KEY_LOG_RE = re.compile(
    r"ShutdownThread|boot_progress|sys\.boot_completed|BootReceiver|"
    r"DisplayPowerController|setDisplayState|PowerManagerService|"
    r"SurfaceFlinger.*(?:blank|power|display)|Watchdog|ANR in|FATAL EXCEPTION|"
    r"system_server.*died|init:.*(?:reboot|shutdown)", re.I)


def _adb(serial, *args, timeout=15, text=True):
    cmd = ["adb"] + (["-s", serial] if serial else []) + list(args)
    return subprocess.check_output(cmd, timeout=timeout, text=text,
                                   stderr=subprocess.DEVNULL)


class DeviceContext:
    """后台采集设备状态 + logcat，供黑屏事件回溯归因。"""

    def __init__(self, serial=None, get_display_ids=None):
        self.serial = serial
        # 由投屏侧提供 {屏号: display_id}，用于按屏取电源态与 screencap
        self.get_display_ids = get_display_ids or (lambda: {})

        self.lock = threading.Lock()
        self.running = True
        self.logs = collections.deque(maxlen=LOG_LINES_KEPT)
        self.state = {}                 # 最近一次采样
        self.history = collections.deque(maxlen=600)   # 约 10 分钟状态历史
        self.last_reboot_at = None      # 检测到 uptime 回退的墙钟时间
        self.available = False

        self._threads = [threading.Thread(target=self._state_loop, daemon=True),
                         threading.Thread(target=self._log_loop, daemon=True)]

    def start(self):
        for t in self._threads:
            t.start()

    def stop(self):
        self.running = False

    # ------------------------------------------------------------ 采集
    def _state_loop(self):
        prev_uptime = None
        while self.running:
            try:
                out = _adb(self.serial, "shell",
                           'echo "$(cut -d. -f1 /proc/uptime)|'
                           '$(getprop sys.boot_completed)|$(getprop ro.boot.bootreason)"',
                           timeout=10).strip()
                uptime_s, boot_completed, boot_reason = (out.split("|") + ["", "", ""])[:3]
                uptime = float(uptime_s or 0)
                screens = self._read_screen_states()
                now = time.time()

                # uptime 回退 = 设备重启过（adb 断连期间也会体现为回退）
                if prev_uptime is not None and uptime < prev_uptime - 5:
                    with self.lock:
                        self.last_reboot_at = now
                prev_uptime = uptime

                snap = {"t": now, "uptime": uptime,
                        "boot_completed": boot_completed.strip() == "1",
                        "boot_reason": boot_reason.strip(),
                        "adb_ok": True, "screens": screens}
                with self.lock:
                    self.state = snap
                    self.history.append(snap)
                    self.available = True
            except Exception:
                now = time.time()
                snap = {"t": now, "adb_ok": False, "screens": {}}
                with self.lock:
                    self.state = snap
                    self.history.append(snap)
                prev_uptime = None       # 断连后重新基线，回连时能测出重启
            time.sleep(STATE_INTERVAL_S)

    def _read_screen_states(self):
        """读每块屏的电源态。返回 {display_id: "ON"/"OFF"/...}。"""
        try:
            dd = _adb(self.serial, "shell", "dumpsys", "display", timeout=15)
        except Exception:
            return {}
        states = {}
        for m in re.finditer(r'uniqueId="local:(\d+)".*?, state ([A-Z_]+)', dd, re.S):
            states.setdefault(m.group(1), m.group(2))
        return states

    def _log_loop(self):
        """常驻 logcat 流，只保留与重启/显示相关的行，避免缓冲被刷爆。"""
        while self.running:
            proc = None
            try:
                cmd = ["adb"] + (["-s", self.serial] if self.serial else []) + \
                      ["logcat", "-v", "threadtime"]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
                for line in proc.stdout:
                    if not self.running:
                        break
                    if KEY_LOG_RE.search(line):
                        with self.lock:
                            self.logs.append((time.time(), line.rstrip()))
            except Exception:
                pass
            finally:
                if proc and proc.poll() is None:
                    try: proc.kill()
                    except Exception: pass
            if self.running:
                time.sleep(3.0)

    # ------------------------------------------------------------ 归因
    def recent_logs(self, window_s=30.0, limit=40):
        cutoff = time.time() - window_s
        with self.lock:
            rows = [l for t, l in self.logs if t >= cutoff]
        return rows[-limit:]

    def rebooted_recently(self, window_s=REBOOT_WINDOW_S):
        with self.lock:
            r = self.last_reboot_at
            hist = list(self.history)
        now = time.time()
        if r and now - r <= window_s:
            return True, f"检测到 uptime 回退（{now - r:.0f} 秒前）"
        # adb 在窗口内断过 = 设备很可能在重启
        lost = [h for h in hist if h["t"] >= now - window_s and not h.get("adb_ok")]
        if lost:
            return True, f"窗口内 adb 断连 {len(lost)} 次（设备重启/掉线）"
        recent = [h for h in hist if h["t"] >= now - window_s and h.get("adb_ok")]
        if recent and not recent[-1].get("boot_completed", True):
            return True, "sys.boot_completed=0（仍在开机过程中）"
        if recent and recent[-1].get("uptime", 1e9) < window_s:
            return True, f"uptime 仅 {recent[-1]['uptime']:.0f} 秒（刚开机）"
        return False, ""

    def classify(self, display_id=None, camera_dark=True):
        """给一次黑屏事件归因。

        display_id 为该相机 ROI 对应的设备屏；给不出时只能判到重启一级。
        返回 {"verdict", "is_normal", "reason", "screen_state", "device_mean", "logs"}。
        """
        with self.lock:
            available = self.available
        if not available:
            return {"verdict": "unknown", "is_normal": None,
                    "reason": "adb 不可用，无法判断是正常行为还是故障",
                    "screen_state": None, "device_mean": None, "logs": []}

        logs = self.recent_logs()
        rebooted, why = self.rebooted_recently()
        if rebooted:
            return {"verdict": "reboot", "is_normal": True,
                    "reason": f"设备重启导致的黑屏（{why}）",
                    "screen_state": None, "device_mean": None, "logs": logs}

        with self.lock:
            screens = (self.state or {}).get("screens", {})
        state = screens.get(str(display_id)) if display_id else None
        if state and state != "ON":
            return {"verdict": "screen_off", "is_normal": True,
                    "reason": f"该屏电源态为 {state}，属正常灭屏",
                    "screen_state": state, "device_mean": None, "logs": logs}

        device_mean = None
        if display_id:
            device_mean = self._screencap_mean(display_id)

        if device_mean is not None and device_mean < DARK_MEAN:
            return {"verdict": "device_black", "is_normal": False,
                    "reason": (f"屏电源 ON 但设备端画面就是黑的"
                               f"（screencap 亮度 {device_mean:.1f}）"
                               f"，合成器/应用侧未出画面"),
                    "screen_state": state, "device_mean": device_mean, "logs": logs}
        if device_mean is not None and camera_dark:
            return {"verdict": "panel_black", "is_normal": False,
                    "reason": (f"设备端画面正常（screencap 亮度 {device_mean:.1f}）"
                               f"但相机看到黑，问题在屏体/背光/传输链路"),
                    "screen_state": state, "device_mean": device_mean, "logs": logs}
        return {"verdict": "unknown", "is_normal": None,
                "reason": "未取到该屏的设备端画面，无法区分设备侧还是显示链路",
                "screen_state": state, "device_mean": device_mean, "logs": logs}

    def _screencap_mean(self, display_id):
        try:
            raw = _adb(self.serial, "exec-out", "screencap", "-d", str(display_id), "-p",
                       timeout=20, text=False)
        except Exception:
            return None
        i = raw.find(b"\x89PNG")
        if i < 0:
            return None
        im = cv2.imdecode(np.frombuffer(raw[i:], np.uint8), cv2.IMREAD_COLOR)
        return None if im is None else float(im.mean())

    def status_dict(self):
        with self.lock:
            st = dict(self.state or {})
            n_logs = len(self.logs)
            reboot = self.last_reboot_at
        return {
            "available": self.available,
            "adb_ok": st.get("adb_ok", False),
            "uptime": st.get("uptime"),
            "boot_completed": st.get("boot_completed"),
            "boot_reason": st.get("boot_reason"),
            "screens": st.get("screens", {}),
            "log_lines": n_logs,
            "last_reboot_ago_s": round(time.time() - reboot, 1) if reboot else None,
        }
