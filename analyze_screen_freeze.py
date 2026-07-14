"""冻屏检测（青色检测框）：屏幕区域连续帧变化量低于阈值超过时限。
注意：静态菜单会误报，建议结合操作日志复核。"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anomaly_core as core

FREEZE_MIN_S = 3.0     # 持续静止判冻屏的最短时间
ABS_T = 1.5            # 绝对帧差阈值(屏录/三脚架场景)
REL_T = 0.40           # 相对阈值: 低于滚动中位数的该比例视为静止(手持场景)


class ScreenFreezeDetector:
    color = (255, 255, 0)         # 青
    label = "SCREEN FREEZE"
    abnormal_type = "screen_freeze"

    def __init__(self, roi, scale, fps):
        self.roi = roi
        self.fps = fps
        self.prev = None
        self.still_since = None
        from collections import deque
        self.diff_hist = deque(maxlen=120)

    def _comp_diff(self, prev, cur):
        """相位相关补偿全局平移(手持抖动)后的帧差。"""
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(prev, M, (prev.shape[1], prev.shape[0]))
        m = 6
        return float(np.abs(cur - warped)[m:-m, m:-m].mean())

    def process(self, small, gray, t):
        sx, sy, sw, sh = self.roi
        sub = cv2.resize(gray[sy:sy+sh, sx:sx+sw], (240, 135)).astype(np.float32)
        sub = cv2.GaussianBlur(sub, (5, 5), 0)
        if self.prev is None:
            self.prev = sub
            return False, None, 0.0, ""
        diff = self._comp_diff(self.prev, sub)
        self.prev = sub
        self.diff_hist.append(diff)
        med = float(np.median(self.diff_hist)) if len(self.diff_hist) >= 30 else None
        thr = max(ABS_T, REL_T * med) if med is not None else ABS_T
        if diff < thr:
            if self.still_since is None:
                self.still_since = t
            dur = t - self.still_since
            if dur >= FREEZE_MIN_S:
                sc = min(1.0, dur / 10.0)
                return True, (sx, sy, sw, sh), sc, f"still {dur:.1f}s diff={diff:.2f}"
        else:
            self.still_since = None
        return False, None, 0.0, ""


if __name__ == "__main__":
    core.cli(ScreenFreezeDetector, "冻屏检测")
