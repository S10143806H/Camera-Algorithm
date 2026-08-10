"""花屏检测（紫色检测框）：屏幕区域出现噪点纹理或大块乱色。

两条互补判据，任一成立即判花屏：

1. **色块跳变**（主判据）——把 ROI 压成网格取每格均色，算相邻格在 Lab 空间的
   色差。真花屏是一片互不相干的色块拼在一起，相邻格色差极大；正常 UI 即使
   花花绿绿，色块之间也有留白/渐变过渡，相邻格色差小一个数量级。
   实测同一台架：花屏「高跳格占比 0.44~0.91、均跳 23~42」，
   正常画面「0.00~0.04、均跳 2.8~7.1」。

2. **梯度能量**（补充判据）——高频噪点/撕裂型花屏颜色可能很淡，靠 Sobel 梯度
   抓。这条沿用旧实现，并保留帧间变化门槛：静态的彩色方块 UI 曾在这条上误报
   （cells=8~10、score 0.3~0.45），要求"这块花纹在动"可以分开。

判据 1 不设帧间门槛：车机上的花屏经常是**定格**的（画面卡死在一帧乱码上），
只认"在动"的会整类漏掉——实测三块屏同时花屏、肉眼明显，旧实现全部报 0.00。

网格边长按 ROI 尺寸自适应：仪表屏那种细长 ROI（缩放后仅 42px 高）用固定 24px
只能切出 1 行 7 格，永远达不到最小连片格数，等于该屏检测被禁用。
"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anomaly_core as core

CELL_MIN, CELL_MAX = 6, 24   # 网格边长上下限，实际按 ROI 短边自适应
CELL_DIV = 6                 # 短边至少切成这么多格

JUMP_CELL_T = 22.0   # 单格判"花"的邻格 Lab 色差阈值
MIN_CELL_FRAC = 0.20 # 连片异常格至少占 ROI 网格数的比例
MIN_CELLS_ABS = 6    # 连片异常格的绝对下限（网格很少的细长 ROI 用）

GRAD_T = 55.0        # 网格梯度能量阈值（补充判据的必要条件）
SAT_T = 0.30         # 高饱和噪点占比阈值（彩色马赛克型花屏走这条）
GRAD_STRONG = 95.0   # 梯度极强：即使几乎无彩色，也判为花屏（灰度噪点/撕裂型）
DYN_T = 8.0          # 仅作用于梯度判据：静态彩色 UI 曾在这条上误报


def _neighbor_jump(cell_bgr):
    """每格与其上下左右邻格的平均 Lab 色差。"""
    lab = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    rows, cols = lab.shape[:2]
    acc = np.zeros((rows, cols), np.float32)
    cnt = np.zeros((rows, cols), np.float32)
    dx = np.linalg.norm(np.diff(lab, axis=1), axis=2)
    acc[:, :-1] += dx; cnt[:, :-1] += 1
    acc[:, 1:] += dx;  cnt[:, 1:] += 1
    dy = np.linalg.norm(np.diff(lab, axis=0), axis=2)
    acc[:-1, :] += dy; cnt[:-1, :] += 1
    acc[1:, :] += dy;  cnt[1:, :] += 1
    return acc / np.maximum(cnt, 1)


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
        if sub.size == 0:
            return False, None, 0.0, ""

        cell_px = int(np.clip(min(sh, sw) // CELL_DIV, CELL_MIN, CELL_MAX))
        rows, cols = sh // cell_px, sw // cell_px
        if rows < 2 or cols < 2:
            return False, None, 0.0, ""
        total = rows * cols

        # ---- 判据 1：相邻网格色块跳变
        cell_bgr = cv2.resize(sub, (cols, rows), interpolation=cv2.INTER_AREA)
        jump = _neighbor_jump(cell_bgr)
        mask = (jump > JUMP_CELL_T).astype(np.uint8) * 255

        # ---- 判据 2：梯度能量（要求在动，避免静态彩色 UI 误报）
        g = gray[sy:sy+sh, sx:sx+sw].astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        sat_hi = ((hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 70)).astype(np.float32)
        gi = g.astype(np.uint8)
        dyn = None
        if self.prev is not None and self.prev.shape == gi.shape:
            dyn = cv2.absdiff(gi, self.prev).astype(np.float32)
        self.prev = gi.copy()

        for r in range(rows):
            for c in range(cols):
                if mask[r, c]:
                    continue
                sl = (slice(r*cell_px, (r+1)*cell_px), slice(c*cell_px, (c+1)*cell_px))
                gm, sm = grad[sl].mean(), sat_hi[sl].mean()
                if not (gm > GRAD_T and (sm > SAT_T or gm > GRAD_STRONG)):
                    continue
                # 首帧没有上一帧可比，无法确认"在动"，这条判据直接跳过。
                # 放行首帧会漏进误报：实测正常画面首帧就从这条报过一次。
                if dyn is None or dyn[sl].mean() <= DYN_T:
                    continue
                mask[r, c] = 255

        need = max(MIN_CELLS_ABS, int(MIN_CELL_FRAC * total))
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = None
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= need and (best is None or area > best[0]):
                best = (area, stats[i])
        if best:
            area, st = best
            bb = (int(sx + st[cv2.CC_STAT_LEFT]*cell_px),
                  int(sy + st[cv2.CC_STAT_TOP]*cell_px),
                  int(st[cv2.CC_STAT_WIDTH]*cell_px),
                  int(st[cv2.CC_STAT_HEIGHT]*cell_px))
            frac = area / float(total)
            return (True, bb, min(1.0, frac / 0.5),
                    f"cells={int(area)}/{total} jump={jump.mean():.0f}")
        return False, None, 0.0, ""


if __name__ == "__main__":
    core.cli(ScreenDistortedDetector, "花屏检测")
