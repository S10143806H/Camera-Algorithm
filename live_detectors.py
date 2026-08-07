"""
实时多类型异常检测组
====================
把原本只用于离线视频的 anomaly_core 系检测器（白屏 / 闪屏 / 花屏 / 冻屏）接到
实时相机链路上，与既有的黑屏检测并列，做到"一块屏同时跑多种异常判定"。

两套检测器的接口本来不一样：
  * 黑屏  detect_dark_region(frame, screen_roi) -> dict          按原分辨率
  * 其余  Detector(roi_scaled, scale, fps).process(small, gray, t)
          -> (abnormal, bbox_scaled, score, info)                按 960 宽降采样

本模块统一成一种结果形状，并保持与离线脚本相同的降采样比例（960 宽），
使阈值行为与离线验证过的一致。

检测器是有状态的（闪屏留滚动窗口、冻屏留上一帧），因此每块屏、每种类型各持
一个实例；ROI 或分辨率一变就整体重建，避免拿旧屏的历史去判新屏。
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyze_black_screens import detect_dark_region  # noqa: E402

BLACK = "black_screen"

# 与离线脚本一致：长边压到 960 再判，阈值才和离线验证过的行为相同
TARGET_W = 960.0


def _load_detector_classes():
    """按需导入各类型检测器；某个脚本缺失不影响其它类型。"""
    out = {}
    for mod, cls in (("analyze_white_screen", "WhiteScreenDetector"),
                     ("analyze_screen_flicking", "ScreenFlickingDetector"),
                     ("analyze_screen_distorted", "ScreenDistortedDetector"),
                     ("analyze_screen_freeze", "ScreenFreezeDetector")):
        try:
            m = __import__(mod)
            c = getattr(m, cls)
            out[c.abnormal_type] = c
        except Exception as e:                       # noqa: BLE001
            print(f"⚠️ 未能加载 {cls}: {e}")
    return out


DETECTOR_CLASSES = _load_detector_classes()

# 黑屏沿用自己的实现，其余来自 anomaly_core 系
ALL_TYPES = [BLACK] + list(DETECTOR_CLASSES.keys())

TYPE_LABELS = {BLACK: "BLACK SCREEN"}
TYPE_COLORS = {BLACK: (0, 0, 255)}
for _t, _c in DETECTOR_CLASSES.items():
    TYPE_LABELS[_t] = getattr(_c, "label", _t)
    TYPE_COLORS[_t] = tuple(getattr(_c, "color", (0, 0, 255)))


# 画面上叠字用的短代号：完整 label（如 "SCREEN DISTORTED"）太长，
# 与 bbox 上的细标签并排时会互相压住
TYPE_SHORT = {
    BLACK: "BLACK",
    "white_screen": "WHITE",
    "screen_flicking": "FLICK",
    "screen_distorted": "GARBLE",
    "screen_freeze": "FREEZE",
}


def normalize_types(types):
    """把用户传入的类型名规整成有效列表；'all' 表示全开。"""
    if not types:
        return [BLACK]
    if isinstance(types, str):
        types = [t.strip() for t in types.split(",") if t.strip()]
    if "all" in types:
        return list(ALL_TYPES)
    keep, unknown = [], []
    for t in types:
        (keep if t in ALL_TYPES else unknown).append(t)
    if unknown:
        print(f"⚠️ 未知异常类型 {unknown}，可选: {', '.join(ALL_TYPES)}")
    return keep or [BLACK]


class ScreenDetectors:
    """单块屏幕上的一组检测器（每种类型一个实例）。"""

    def __init__(self, roi, scale, fps, types):
        self.roi = roi                      # 原分辨率坐标
        self.scale = scale
        self.types = types
        self.dets = {}
        rs = tuple(int(v * scale) for v in roi) if roi else None
        for t in types:
            if t == BLACK:
                continue
            cls = DETECTOR_CLASSES.get(t)
            if cls is None:
                continue
            try:
                self.dets[t] = cls(rs, scale, fps)
            except Exception as e:          # noqa: BLE001
                print(f"⚠️ 初始化 {t} 检测器失败: {e}")

    def run(self, frame, small, gray, t):
        """返回 {type: {...}}，形状与黑屏结果对齐。"""
        out = {}
        if BLACK in self.types:
            try:
                r = detect_dark_region(frame, screen_roi=self.roi)
            except TypeError:
                r = detect_dark_region(frame)
            region = r.get("region") or {}
            out[BLACK] = {
                "abnormal": bool(r.get("abnormal")),
                "bbox": region.get("bbox"),
                "score": round(min(1.0, float(region.get("dark_pct", 0.0)) / 100.0), 3),
                "info": f"dark={region.get('dark_pct', 0.0):.1f}%",
                # 保留原始结构，事件与标注沿用既有黑屏逻辑
                "raw": r,
            }
        for name, det in self.dets.items():
            try:
                abnormal, bbox_s, score, info = det.process(small, gray, t)
            except Exception as e:          # noqa: BLE001
                print(f"⚠️ {name} 检测异常: {e}")
                continue
            bbox = ([int(v / self.scale) for v in bbox_s]
                    if (bbox_s and self.scale < 1) else (list(bbox_s) if bbox_s else None))
            out[name] = {"abnormal": bool(abnormal), "bbox": bbox,
                         "score": round(float(score), 3), "info": info or "", "raw": None}
        return out


class DetectorBank:
    """全部屏幕 × 全部类型。ROI / 分辨率 / 类型变化时整体重建。"""

    def __init__(self, types=None, fps=30.0):
        self.types = normalize_types(types)
        self.fps = fps or 30.0
        self._key = None
        self._screens = []
        self.scale = 1.0

    def _rebuild(self, rois, shape):
        h, w = shape[:2]
        self.scale = min(1.0, TARGET_W / w)
        self._screens = [ScreenDetectors(roi, self.scale, self.fps, self.types)
                         for roi in (rois or [None])]
        self._key = (tuple(map(tuple, rois or [])), w, h, tuple(self.types))

    def run(self, frame, t):
        """对每块屏跑全部启用的检测器。

        返回 [(screen_no, roi, {type: result}), ...]，screen_no 从 1 起；
        未标定 ROI 时整幅画面视为一块，screen_no=0。
        """
        rois = getattr(self, "_rois", None) or []
        h, w = frame.shape[:2]
        key = (tuple(map(tuple, rois)), w, h, tuple(self.types))
        if key != self._key:
            self._rebuild(rois, frame.shape)

        small = (cv2.resize(frame, (int(w * self.scale), int(h * self.scale)),
                            interpolation=cv2.INTER_AREA) if self.scale < 1 else frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        out = []
        for i, sd in enumerate(self._screens):
            no = (i + 1) if rois else 0
            out.append((no, sd.roi, sd.run(frame, small, gray, t)))
        return out

    def set_rois(self, rois):
        """ROI 变化后强制重建，避免用旧屏的历史判新屏。"""
        self._rois = list(rois or [])
        self._key = None
