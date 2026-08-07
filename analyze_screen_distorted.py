"""花屏检测（紫色检测框）：屏幕区域出现高频噪点/块状色彩异常纹理。

按 24px 网格计算梯度能量与高饱和噪点占比，连片异常网格构成花屏区域。
判据为"梯度达标"且"颜色异常或梯度极强"二选一——早先要求梯度与饱和度同时
达标，实测会漏掉灰度噪点/撕裂型花屏：合成样本里 240/240 网格梯度超阈，但
只有 3 个网格饱和度达标，整段一帧未报。"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anomaly_core as core

CELL = 24
GRAD_T = 55.0        # 网格梯度能量阈值（判花屏的必要条件）
SAT_T = 0.30         # 高饱和噪点占比阈值（彩色马赛克型花屏走这条）
GRAD_STRONG = 95.0   # 梯度极强：即使几乎无彩色，也判为花屏（灰度噪点/撕裂型）
MIN_CELLS = 8        # 触发的最少连片网格数
DYN_T = 8.0          # 网格帧间平均变化量下限：真花屏逐帧乱跳，静态彩色 UI 不会


class ScreenDistortedDetector:
    color = (255, 0, 255)         # 紫
    label = "SCREEN DISTORTED"
    abnormal_type = "screen_distorted"

    def __init__(self, roi, scale, fps):
        self.roi = roi
        self.prev = None        # 上一帧 ROI 灰度，用于判"这块花纹是不是在动"

    def process(self, small, gray, t):
        sx, sy, sw, sh = self.roi
        sub = small[sy:sy+sh, sx:sx+sw]
        g = gray[sy:sy+sh, sx:sx+sw].astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        sat_hi = ((hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 70)).astype(np.float32)
        # 帧间差：真花屏的花纹逐帧随机重排，静态的彩色方块 UI 几乎不变。
        # 实测台架上一块彩色方块 UI 会被误判为马赛克花屏（cells=8~10、score 0.3~0.45），
        # 加这道判据后可与真花屏（cells=240、score 1.0）分开。
        gi = g.astype(np.uint8)
        dyn = None
        if self.prev is not None and self.prev.shape == gi.shape:
            dyn = cv2.absdiff(gi, self.prev).astype(np.float32)
        self.prev = gi.copy()

        rows, cols = sh // CELL, sw // CELL
        if rows < 2 or cols < 2:
            return False, None, 0.0, ""
        mask = np.zeros((rows, cols), np.uint8)
        for r in range(rows):
            for c in range(cols):
                cg = grad[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL]
                cs = sat_hi[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL]
                gm, sm = cg.mean(), cs.mean()
                if not (gm > GRAD_T and (sm > SAT_T or gm > GRAD_STRONG)):
                    continue
                # 还要求这块在动；拿不到上一帧时（首帧）不拦
                if dyn is not None:
                    if dyn[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL].mean() <= DYN_T:
                        continue
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
