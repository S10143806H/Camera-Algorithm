"""白屏检测（棕色检测框）：屏幕区域大面积高亮、无有效画面。镜像黑屏 v3 规则。"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anomaly_core as core


class WhiteScreenDetector:
    color = (0, 165, 255)         # 橙：原来的深棕压在亮屏上几乎看不见
    label = "WHITE SCREEN"
    abnormal_type = "white_screen"
    bridge_gap_s = 3.5            # 同区域检出间隙桥接(反光等致中段暂时越界)

    def __init__(self, roi, scale, fps):
        self.roi = roi

    def process(self, small, gray, t):
        """白屏=亮+平坦+大面积: 相机自动曝光会把白屏压暗, 不能用绝对高亮阈值。
        flat_white = 亮(>165) 且 无边缘纹理 的像素; 最大连通域够大即白屏 pane。"""
        sx, sy, sw, sh = self.roi
        sub = gray[sy:sy+sh, sx:sx+sw]
        screen_area = sw * sh
        edges = cv2.Canny(sub, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        flat_white = ((sub > 165) & (edges == 0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(flat_white, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if bw * bh < 0.28 * screen_area or bw < 0.30 * sw or bh < 0.35 * sh:
                continue
            roi_g = sub[y:y+bh, x:x+bw]
            mean = float(roi_g.mean())
            std = float(roi_g.std())
            edge = float((cv2.Canny(roi_g, 50, 150) > 0).mean() * 100)
            fill = float(area) / (bw * bh)
            if mean > 175 and std < 45 and edge < 2.5 and fill > 0.55:
                sc = fill
                if best is None or sc > best[1]:
                    best = ((int(sx+x), int(sy+y), int(bw), int(bh)), sc,
                            f"mean={mean:.0f} std={std:.0f} fill={fill:.2f}")
        if best:
            return True, best[0], min(1.0, best[1]), best[2]
        return False, None, 0.0, ""


if __name__ == "__main__":
    core.cli(WhiteScreenDetector, "白屏检测")
