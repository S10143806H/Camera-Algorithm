"""花屏检测（紫色检测框）：屏幕区域出现高频噪点/块状色彩异常纹理。
按 24px 网格计算梯度能量+高饱和噪点占比, 连片异常网格构成花屏区域。"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anomaly_core as core

CELL = 24
GRAD_T = 55.0       # 网格梯度能量阈值
SAT_T = 0.30        # 高饱和噪点占比阈值
MIN_CELLS = 8       # 触发的最少连片网格数


class ScreenDistortedDetector:
    color = (255, 0, 255)         # 紫
    label = "SCREEN DISTORTED"
    abnormal_type = "screen_distorted"

    def __init__(self, roi, scale, fps):
        self.roi = roi

    def process(self, small, gray, t):
        sx, sy, sw, sh = self.roi
        sub = small[sy:sy+sh, sx:sx+sw]
        g = gray[sy:sy+sh, sx:sx+sw].astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        sat_hi = ((hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 70)).astype(np.float32)
        rows, cols = sh // CELL, sw // CELL
        if rows < 2 or cols < 2:
            return False, None, 0.0, ""
        mask = np.zeros((rows, cols), np.uint8)
        for r in range(rows):
            for c in range(cols):
                cg = grad[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL]
                cs = sat_hi[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL]
                if cg.mean() > GRAD_T and cs.mean() > SAT_T:
                    mask[r, c] = 255
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_CELLS and (best is None or area > best[0]):
                best = (area, stats[i])
        if best:
            area, st = best
            bb = (int(sx + st[cv2.CC_STAT_LEFT]*CELL), int(sy + st[cv2.CC_STAT_TOP]*CELL),
                  int(st[cv2.CC_STAT_WIDTH]*CELL), int(st[cv2.CC_STAT_HEIGHT]*CELL))
            return True, bb, min(1.0, float(area) / 40.0), f"cells={int(area)}"
        return False, None, 0.0, ""


if __name__ == "__main__":
    core.cli(ScreenDistortedDetector, "花屏检测")
