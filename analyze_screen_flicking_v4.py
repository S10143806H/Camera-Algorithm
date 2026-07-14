"""闪屏检测 v4 引擎包装（黄色检测框）。

复用 D:\\flicker-detection 的 FlickerDetectorV4（增益归一化局部能量 +
结构不变性门控 + ABA 脉冲 + 滚动 MAD 基线），比本目录 V 型分区检测器
召回更强（实测命中 60fps 背光闪和低幅仪表闪）。
flicker_detector.py / flicker_detector_v4.py 为副本，源仓库 D:\\flicker-detection。

用法同其他 analyze_screen_*：
  python analyze_screen_flicking_v4.py --video x.mp4 [--out dir] [--finalize]
"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anomaly_core as core
from flicker_detector_v4 import FlickerDetectorV4

THRESHOLD = 0.70


class ScreenFlickingV4Detector:
    color = (0, 230, 230)         # 黄
    label = "SCREEN FLICKING"
    abnormal_type = "screen_flicking"

    def __init__(self, roi, scale, fps):
        self.roi = roi
        self.det = FlickerDetectorV4()

    def process(self, small, gray, t):
        r = self.det.push_frame(small)
        if r.is_flickering and r.raw_confidence >= THRESHOLD:
            return True, tuple(self.roi), float(min(1.0, r.raw_confidence)), \
                f"conf={r.raw_confidence:.2f} aba={r.details.get('aba', 0):.1f}"
        return False, None, 0.0, ""


if __name__ == "__main__":
    core.cli(ScreenFlickingV4Detector, "闪屏检测(v4引擎)")
