"""闪屏检测（黄色检测框）：屏幕亮度在连续帧间反复突变（振荡）。
按 3x3 分区独立判定, 检测框为闪烁分区的并集。"""
import sys
from collections import deque
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anomaly_core as core

WIN = 13           # 滑窗帧数
EXC_T = 20.0       # 亮度偏移幅值阈值
MIN_ZONES = 3      # 至少多少个分区同时闪变


class ScreenFlickingDetector:
    color = (0, 230, 230)         # 黄
    label = "SCREEN FLICKING"
    abnormal_type = "screen_flicking"

    def __init__(self, roi, scale, fps):
        self.roi = roi
        self.hist = [deque(maxlen=WIN) for _ in range(9)]

    def process(self, small, gray, t):
        """闪屏＝亮度骤变后快速恢复的 V 型闪变（区别于单次界面切换）。"""
        sx, sy, sw, sh = self.roi
        zones, flick, max_exc = [], [], 0.0
        for zi in range(9):
            r, c = divmod(zi, 3)
            zx, zy = sx + c*sw//3, sy + r*sh//3
            zw, zh = sw//3, sh//3
            self.hist[zi].append(float(gray[zy:zy+zh, zx:zx+zw].mean()))
            zones.append((zx, zy, zw, zh))
            h = self.hist[zi]
            if len(h) == WIN:
                s_ = np.array(h)
                base = np.median(s_[:4])
                dev = s_ - base
                pk = int(np.argmax(np.abs(dev))); exc = abs(dev[pk])
                if exc > EXC_T and 2 <= pk <= WIN-3 and abs(dev[-1]) < 0.5*exc:
                    flick.append(zi); max_exc = max(max_exc, exc)
        if len(flick) >= MIN_ZONES:
            xs = [zones[i][0] for i in flick]; ys = [zones[i][1] for i in flick]
            x2 = max(zones[i][0]+zones[i][2] for i in flick)
            y2 = max(zones[i][1]+zones[i][3] for i in flick)
            bb = (min(xs), min(ys), x2-min(xs), y2-min(ys))
            return True, bb, min(1.0, max_exc/60.0), f"zones={len(flick)} exc={max_exc:.0f}"
        return False, None, 0.0, ""


if __name__ == "__main__":
    core.cli(ScreenFlickingDetector, "闪屏检测")
