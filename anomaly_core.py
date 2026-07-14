"""
屏幕异常检测公共核心（供 analyze_screen_* 系列复用）
====================================================
- 屏幕ROI标定复用 analyze_black_screens.calibrate_screen_roi
- 分析在降采样图上做(宽≤960)提高速度, bbox 映射回原分辨率标注
- 支持分段处理(--start/--count)与合并(--finalize), 适配受限运行环境
- 每种异常类型用不同颜色检测框

检测器接口:
    class Detector:
        color = (B, G, R); label = "XXX"; abnormal_type = "xxx"
        def __init__(self, roi_scaled, scale, fps): ...
        def process(self, small_bgr, small_gray, t):
            return abnormal(bool), bbox_scaled(x,y,w,h)|None, score(float), info(str)
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_black_screens import calibrate_screen_roi  # noqa: E402


def fmt_ts(seconds):
    ms = int(round(seconds * 1000))
    return f"{ms//3600000:02d}:{ms%3600000//60000:02d}:{ms%60000//1000:02d}.{ms%1000:03d}"


def annotate(frame, abnormal, bbox, score, info, t, detector, screen_roi):
    out = frame.copy()
    if screen_roi:
        rx, ry, rw, rh = screen_roi
        cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (255, 160, 0), 2)
    if abnormal and bbox:
        x, y, w_, h_ = bbox
        overlay = out.copy()
        cv2.rectangle(overlay, (x, y), (x + w_, y + h_), detector.color, -1)
        out = cv2.addWeighted(overlay, 0.20, out, 0.80, 0)
        cv2.rectangle(out, (x, y), (x + w_, y + h_), detector.color, 5)
        cv2.putText(out, f"{detector.label} DETECTED  {info}",
                    (max(12, x), max(34, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, detector.color, 2, cv2.LINE_AA)
    cv2.putText(out, fmt_ts(t), (12, out.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)
    return out


def process_chunk(video, detector_cls, out_dir, start, count, roi=None):
    video = Path(video)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem.encode("ascii", "ignore").decode() or "video"

    if roi is None:
        roi = calibrate_screen_roi(str(video))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 960.0 / w)
    sw_, sh_ = int(w * scale), int(h * scale)
    roi_scaled = tuple(int(v * scale) for v in roi) if roi else (0, 0, sw_, sh_)
    det = detector_cls(roi_scaled, scale, fps)

    for _ in range(start):
        cap.grab()
    seg = out_dir / f"{stem}_seg{start:06d}.mp4"
    writer = cv2.VideoWriter(str(seg), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    recs = []
    idx = start
    eof = False
    while idx < start + count:
        ok, frame = cap.read()
        if not ok:
            eof = True; break
        t = idx / fps
        small = cv2.resize(frame, (sw_, sh_), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        abnormal, bbox_s, score, info = det.process(small, gray, t)
        bbox = ([int(v / scale) for v in bbox_s] if (bbox_s and scale < 1)
                else (list(bbox_s) if bbox_s else None))
        roi_orig = tuple(int(v) for v in roi) if roi else None
        writer.write(annotate(frame, abnormal, bbox, score, info, t, det, roi_orig))
        recs.append({"frame": idx, "time": round(t, 3), "abnormal": bool(abnormal),
                     "score": round(float(score), 3),
                     "bbox": bbox if abnormal else None})
        idx += 1
    cap.release(); writer.release()
    with (out_dir / f"{stem}_records.jsonl").open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"processed_to": idx, "eof": eof, "roi": roi}))



def build_evidence_sheet(video, detector_cls, out_dir, max_frames=40, min_gap=0.4):
    """从 records 挑异常帧(去重), 重读原帧标注后合并成拼图, 供人工复核迭代。"""
    video = Path(video); out_dir = Path(out_dir)
    stem = video.stem.encode("ascii", "ignore").decode() or "video"
    rec_file = out_dir / f"{stem}_records.jsonl"
    if not rec_file.exists():
        return None
    recs = [json.loads(l) for l in rec_file.open(encoding="utf-8")]
    ab = sorted((r for r in recs if r["abnormal"]), key=lambda r: r["frame"])
    picked, last_t = [], -1e9
    for r in ab:
        if r["time"] - last_t >= min_gap:
            picked.append(r); last_t = r["time"]
    if not picked:
        return None
    if len(picked) > max_frames:
        idxs = np.linspace(0, len(picked) - 1, max_frames).astype(int)
        picked = [picked[i] for i in idxs]

    class _D:  # 标注用哑检测器(颜色/标签)
        color = detector_cls.color; label = detector_cls.label
    cap = cv2.VideoCapture(str(video))
    thumbs = []
    for r in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        ann = annotate(frame, True, r["bbox"], r["score"],
                       f"score={r['score']:.2f}", r["time"], _D, None)
        thumbs.append(cv2.resize(ann, (240, int(240*frame.shape[0]/frame.shape[1]))))
    cap.release()
    if not thumbs:
        return None
    th_ = max(t.shape[0] for t in thumbs)
    cols = min(8, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    header = 72
    sheet = np.full((header + rows * th_, cols * 240, 3), 255, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r_, c_ = divmod(i, cols)
        sheet[header + r_*th_: header + r_*th_ + t.shape[0], c_*240:(c_+1)*240] = t
    cv2.putText(sheet, f"{detector_cls.label} EVIDENCE  {len(picked)} frames  {video.name[:52]}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, detector_cls.color, 2, cv2.LINE_AA)
    cv2.putText(sheet, "please mark FP frames for iteration (blue circle)",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    path = out_dir / f"{stem}_evidence.jpg"
    cv2.imwrite(str(path), sheet)
    return path


def finalize(video, detector_cls, out_dir):
    video = Path(video); out_dir = Path(out_dir)
    stem = video.stem.encode("ascii", "ignore").decode() or "video"
    det_color = detector_cls.color
    recs = [json.loads(l) for l in (out_dir / f"{stem}_records.jsonl").open(encoding="utf-8")]
    seen, uniq = set(), []
    for r in sorted(recs, key=lambda r: r["frame"]):
        if r["frame"] not in seen:
            seen.add(r["frame"]); uniq.append(r)
    recs = uniq
    segs = sorted(out_dir.glob(f"{stem}_seg*.mp4"))
    processed = out_dir / f"{stem}_processed.mp4"
    if segs:
        lst = out_dir / f"{stem}_list.txt"
        lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(processed)], capture_output=True, check=True)
        for s in segs: s.unlink()
        lst.unlink()
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()

    # 间隙桥接: 相邻两段异常若间隔 ≤ bridge_gap_s 且 bbox IoU≥0.4, 中间帧补记为异常
    bridge = float(getattr(detector_cls, "bridge_gap_s", 0) or 0)
    if bridge > 0:
        def _iou(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            x1, y1 = max(ax, bx), max(ay, by)
            x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
            inter = max(0, x2-x1) * max(0, y2-y1)
            return inter / float(aw*ah + bw*bh - inter + 1e-9)
        runs = []
        for i, r in enumerate(recs):
            if r["abnormal"]:
                if runs and i == runs[-1][1] + 1: runs[-1][1] = i
                else: runs.append([i, i])
        for a, b in zip(runs, runs[1:]):
            r_end, r_start = recs[a[1]], recs[b[0]]
            if (r_start["time"] - r_end["time"] <= bridge
                    and r_end["bbox"] and r_start["bbox"]
                    and _iou(r_end["bbox"], r_start["bbox"]) >= 0.4):
                for j in range(a[1]+1, b[0]):
                    recs[j]["abnormal"] = True
                    recs[j]["bbox"] = r_end["bbox"]
                    recs[j]["score"] = min(r_end["score"], r_start["score"])
    # 区间
    intervals, s_, last = [], None, None
    for r in recs:
        if r["abnormal"]:
            if s_ is None: s_ = r["time"]
            last = r["time"]
        elif s_ is not None:
            intervals.append([s_, last + 1.0 / fps]); s_ = None
    if s_ is not None:
        intervals.append([s_, last + 1.0 / fps])
    ab = [r for r in recs if r["abnormal"]]
    best = max(ab, key=lambda r: r["score"], default=None)
    summary = {
        "input": str(video), "abnormal_type": detector_cls.abnormal_type,
        "box_color_bgr": list(det_color), "processed_video": str(processed),
        "frame_count": len(recs), "abnormal_frame_count": len(ab),
        "abnormal_intervals_sec": intervals,
        "confidence": best["score"] if best else 0.0,
        "best_bbox": best["bbox"] if best else None,
    }
    (out_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    ev = build_evidence_sheet(video, detector_cls, out_dir)
    if ev:
        summary["evidence_sheet"] = str(ev)
        (out_dir / f"{stem}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"abnormal_frames": len(ab), "intervals": [[round(a,2), round(b,2)] for a, b in intervals][:10],
                      "confidence": summary["confidence"]}, ensure_ascii=False))


def cli(detector_cls, description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10**9)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--roi", help="跳过标定, 直接用 x,y,w,h")
    a = ap.parse_args()
    out = a.out or (Path(a.video).parent / "processed")
    if a.finalize:
        finalize(a.video, detector_cls, out)
    else:
        roi = tuple(int(v) for v in a.roi.split(",")) if a.roi else None
        process_chunk(a.video, detector_cls, out, a.start, a.count, roi=roi)
