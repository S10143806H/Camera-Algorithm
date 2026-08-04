import csv
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(os.environ.get(
    "CAMERA_ALGO_DATA_ROOT", Path(__file__).resolve().parent / "data_source"))
OUT = ROOT / "detected_results"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ascii_work_path(path: Path, tmpdir: Path) -> Path:
    try:
        str(path).encode("ascii")
        return path
    except UnicodeEncodeError:
        copy = tmpdir / f"input_{abs(hash(path))}{path.suffix.lower()}"
        shutil.copy2(path, copy)
        return copy


def _reading_order(rois):
    """按阅读顺序（上行→下行，行内左→右）排序，使屏幕编号在多次标定间稳定。"""
    if not rois:
        return []
    band = max(1, int(np.median([r[3] for r in rois]) * 0.5))
    rows = {}
    for r in sorted(rois, key=lambda r: r[1] + r[3] / 2):
        cy = r[1] + r[3] / 2
        key = next((k for k in rows if abs(k - cy) <= band), cy)
        rows.setdefault(key, []).append(r)
    out = []
    for key in sorted(rows):
        out += sorted(rows[key], key=lambda r: r[0])
    return out


def rois_from_maxbright(max_bright, max_screens=3, min_area_frac=0.01):
    """由逐像素最大亮度图求屏幕 bbox 列表（支持画面内多块屏幕）。

    max_screens=1 时等价于旧的单屏行为（取最大亮块）。
    返回按阅读顺序排序的 [(x, y, w, h), ...]，标定失败返回 []。
    """
    h, w = max_bright.shape
    mask = (max_bright > 90).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands, fallback = [], []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        box = (int(x), int(y), int(bw), int(bh))
        if area >= 0.05 * w * h:
            fallback.append((box, area))
        if area < min_area_frac * w * h:
            continue
        # 排除细长的反光条/桌沿：屏幕长宽比通常在 0.4~5 之间
        ar = bw / max(1, bh)
        if not (0.4 <= ar <= 5.0):
            continue
        # 排除轮廓填充率过低的非矩形亮区
        if cv2.contourArea(contour) < 0.45 * area:
            continue
        cands.append((box, area))

    # 形状过滤把所有候选都滤掉时，退回旧的“最大亮块 ≥5% 画面”规则
    if not cands:
        cands = fallback
    if not cands:
        return []
    cands.sort(key=lambda c: c[1], reverse=True)
    # 只保留与最大屏幕面积相差不超过 15 倍的块，滤掉零星小亮点
    biggest = cands[0][1]
    kept = [roi for roi, area in cands[:max_screens] if area * 15 >= biggest]
    return _reading_order(kept)


def calibrate_screen_rois(video_path, samples=40, max_screens=3):
    """标定物理屏幕位置：均匀抽帧累计逐像素最大亮度，返回 ROI 列表。

    摄像头固定时屏幕位置整段视频不变，且屏幕总有点亮的时刻，
    最大亮度图上 >90 的连通区域即物理屏幕 bbox；画面内有多块屏幕时
    返回多个，按阅读顺序（上→下、左→右）编号。标定失败返回 []。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []
    max_bright = None
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * max(1, total - 1) / max(1, samples - 1)))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_bright = gray if max_bright is None else np.maximum(max_bright, gray)
    cap.release()
    if max_bright is None:
        return []
    return rois_from_maxbright(max_bright, max_screens=max_screens)


def calibrate_screen_roi(video_path, samples=40):
    """单屏标定（保留旧签名）：返回面积最大的屏幕 bbox，失败返回 None。"""
    rois = calibrate_screen_rois(video_path, samples=samples, max_screens=1)
    return rois[0] if rois else None


def find_bright_screen(gray):
    """定位当前帧中点亮的屏幕区域（最大亮块 bbox），找不到返回 None。"""
    h, w = gray.shape
    bright = (gray > 90).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh > best_area:
            best_area = bw * bh
            best = (x, y, bw, bh)
    if best is not None and best_area >= 0.03 * w * h:
        return best
    return None


def _edge_ratio(roi):
    edges = cv2.Canny(roi, 50, 150)
    return float((edges > 0).mean() * 100)


def _region_stats(gray, x, y, bw, bh):
    roi = gray[y : y + bh, x : x + bw]
    return {
        "bbox": [int(x), int(y), int(bw), int(bh)],
        "area": int(bw * bh),
        "mean": float(roi.mean()),
        "dark_pct": float((roi < 60).mean() * 100),
        "std": float(roi.std()),
        "edge_pct": _edge_ratio(roi),
    }


def detect_dark_region(frame, screen_roi=None):
    """v3：屏幕相对黑屏检测。

    有 screen_roi（视频标定的物理屏幕 bbox）时，只在屏内找黑屏 pane，
    车内暗背景直接排除：
        P1 深黑 pane：均值<50 + 暗像素≥85% + 无边缘纹理
        P2 泛光黑屏（反光/glare）：均值<95 + 暗像素≥50% + 宽度≥40%屏宽
    无 screen_roi 时退回 v2 逻辑（单帧亮屏定位 + 侧带/屏内规则 + 整屏黑屏兜底）。
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    dark_pct = float((gray < 40).mean() * 100)
    frame_area = w * h
    candidates = []

    if screen_roi is not None:
        sx, sy, sw, sh = screen_roi
        screen_area = sw * sh
        sub = gray[sy : sy + sh, sx : sx + sw]
        dark_sub = sub < 60

        # P1：列/行暗度剖面找深黑 pane（黑屏 pane 整列/整行几乎全暗，
        #      暗色 UI 内容所在列/行总包含亮像素）
        def _runs(profile, min_len):
            runs, start = [], None
            for i, v in enumerate(profile >= 0.90):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    if i - start >= min_len:
                        runs.append((start, i))
                    start = None
            if start is not None and len(profile) - start >= min_len:
                runs.append((start, len(profile)))
            return runs

        col_dark = dark_sub.mean(axis=0)
        for cs, ce in _runs(col_dark, int(0.15 * sw)):
            stats = _region_stats(gray, sx + cs, sy, ce - cs, sh)
            if stats["mean"] < 50 and stats["dark_pct"] >= 85 and stats["edge_pct"] < 2.5:
                candidates.append(stats)
        row_dark = dark_sub.mean(axis=1)
        for rs, re in _runs(row_dark, int(0.15 * sh)):
            stats = _region_stats(gray, sx, sy + rs, sw, re - rs)
            if stats["mean"] < 50 and stats["dark_pct"] >= 85 and stats["edge_pct"] < 2.5:
                candidates.append(stats)

        # P2：泛光黑屏（反光导致均值偏高），轮廓级候选
        mask = dark_sub.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw_, bh_ = cv2.boundingRect(contour)
            area = bw_ * bh_
            if not 0.15 * screen_area <= area <= 0.85 * screen_area:
                continue
            if bw_ < 0.40 * sw:
                continue
            stats = _region_stats(gray, sx + x, sy + y, bw_, bh_)
            if stats["mean"] < 95 and stats["dark_pct"] >= 50 and stats["edge_pct"] < 5.5:
                candidates.append(stats)
        candidates.sort(key=lambda item: (item["dark_pct"], item["area"]), reverse=True)
        best = candidates[0] if candidates else None
        abnormal = bool(best)
        return {
            "mean": mean,
            "dark_pct": dark_pct,
            "abnormal": abnormal,
            "region": best,
            "screen_roi": [sx, sy, sw, sh],
        }

    # ---------- 无标定 ROI：v2 单帧逻辑 ----------
    screen = find_bright_screen(gray)
    if screen is not None:
        sx, sy, sw, sh = screen
        bands = []
        if sx >= max(0.13 * w, 0.12 * sw):
            bands.append((0, sy, sx, sh))
        right_w = w - (sx + sw)
        if right_w >= max(0.13 * w, 0.12 * sw):
            bands.append((sx + sw, sy, right_w, sh))
        if sy >= max(0.13 * h, 0.12 * sh):
            bands.append((sx, 0, sw, sy))
        bottom_h = h - (sy + sh)
        if bottom_h >= max(0.13 * h, 0.12 * sh):
            bands.append((sx, sy + sh, sw, bottom_h))
        for bx, by, bw_, bh_ in bands:
            if bw_ * bh_ < 0.015 * frame_area:
                continue
            stats = _region_stats(gray, bx, by, bw_, bh_)
            if stats["mean"] < 50 and stats["dark_pct"] >= 85 and stats["edge_pct"] < 2.5:
                candidates.append(stats)
        mask = (gray < 60).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw_, bh_ = cv2.boundingRect(contour)
            if (x < sx - 0.5 * sw or y < sy - 0.5 * sh
                    or x + bw_ > sx + sw + 0.5 * sw or y + bh_ > sy + sh + 0.5 * sh):
                continue
            area = bw_ * bh_
            if area < 0.15 * sw * sh:
                continue
            if bw_ < 0.40 * sw:
                continue
            aspect = bw_ / max(bh_, 1)
            if not 0.25 <= aspect <= 3.0:
                continue
            stats = _region_stats(gray, x, y, bw_, bh_)
            if stats["mean"] < 95 and stats["dark_pct"] >= 50 and stats["edge_pct"] < 5.5:
                candidates.append(stats)
        candidates.sort(key=lambda item: (item["dark_pct"], item["area"]), reverse=True)
        best = candidates[0] if candidates else None
        abnormal = bool(best)
    else:
        mask = (gray < 60).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw_, bh_ = cv2.boundingRect(contour)
            area = bw_ * bh_
            if area < frame_area * 0.04:
                continue
            if bw_ < w * 0.12 or bh_ < h * 0.15:
                continue
            aspect = bw_ / max(bh_, 1)
            if not 0.25 <= aspect <= 3.2:
                continue
            stats = _region_stats(gray, x, y, bw_, bh_)
            if stats["mean"] < 75 and stats["dark_pct"] > 55:
                candidates.append(stats)
        candidates.sort(key=lambda item: (item["dark_pct"], item["area"]), reverse=True)
        best = candidates[0] if candidates else None
        abnormal = bool(best and (best["mean"] < 70 or best["dark_pct"] > 70))

    return {
        "mean": mean,
        "dark_pct": dark_pct,
        "abnormal": abnormal,
        "region": best,
    }


def draw_annotation(frame, result, timestamp):
    annotated = frame.copy()
    if result.get("screen_roi"):
        rx, ry, rw, rh = result["screen_roi"]
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 160, 0), 2)
    label = f"{timestamp:05.2f}s"
    if result["abnormal"] and result["region"]:
        x, y, w, h = result["region"]["bbox"]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
        annotated = cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 5)
        text = (
            f"BLACK SCREEN DETECTED  "
            f"region_dark={result['region']['dark_pct']:.1f}%"
        )
        cv2.putText(
            annotated,
            text,
            (max(12, x), max(34, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            annotated,
            "normal/dim screen",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        label,
        (12, annotated.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def confidence_and_reason(result):
    if not result["abnormal"] or not result["region"]:
        return 0.0, "未命中：未找到满足面积、位置、宽高比和暗像素占比的屏幕候选区域"

    region = result["region"]
    dark_score = min(1.0, region["dark_pct"] / 100.0)
    mean_score = max(0.0, min(1.0, (90.0 - region["mean"]) / 90.0))
    confidence = round(0.65 * dark_score + 0.35 * mean_score, 3)
    reason = (
        f"命中：候选屏幕区域暗像素占比 {region['dark_pct']:.1f}%，"
        f"平均亮度 {region['mean']:.1f}，bbox={region['bbox']}"
    )
    return confidence, reason


def intervals_from_flags(records, fps):
    intervals = []
    start = None
    last = None
    for record in records:
        if record["abnormal"]:
            if start is None:
                start = record["time"]
            last = record["time"]
        elif start is not None:
            intervals.append([start, last + 1.0 / fps])
            start = None
            last = None
    if start is not None:
        intervals.append([start, last + 1.0 / fps])
    return intervals


def make_contact_sheet(frames, path):
    if not frames:
        return
    thumbs = [cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA) for frame in frames]
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * 180, cols * 320, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * 180 : (r + 1) * 180, c * 320 : (c + 1) * 320] = thumb
    cv2.imwrite(str(path), sheet)


def process_video(path: Path, tmpdir: Path):
    source = ascii_work_path(path, tmpdir)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = path.stem.encode("ascii", "ignore").decode("ascii") or "video"
    parent = path.parent.name.encode("ascii", "ignore").decode("ascii")
    prefix = f"{parent}_{stem}" if parent and path.parent != ROOT else stem
    out_video = OUT / f"{prefix}_highlighted.mp4"
    out_csv = OUT / f"{prefix}_detections.csv"
    out_sheet = OUT / f"{prefix}_contact_sheet.jpg"

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    screen_roi = calibrate_screen_roi(str(source))
    records = []
    contact_frames = []
    sample_every = max(1, int(round(fps * 0.5)))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = idx / fps
        result = detect_dark_region(frame, screen_roi=screen_roi)
        annotated = draw_annotation(frame, result, timestamp)
        writer.write(annotated)

        region = result["region"] or {}
        bbox = region.get("bbox", ["", "", "", ""])
        records.append(
            {
                "frame": idx,
                "time": timestamp,
                "mean": result["mean"],
                "dark_pct": result["dark_pct"],
                "abnormal": result["abnormal"],
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
                "region_mean": region.get("mean", ""),
                "region_dark_pct": region.get("dark_pct", ""),
            }
        )
        if idx % sample_every == 0:
            contact_frames.append(annotated)
        idx += 1

    cap.release()
    writer.release()
    make_contact_sheet(contact_frames, out_sheet)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(records)

    intervals = intervals_from_flags(records, fps)
    abnormal_records = [record for record in records if record["abnormal"]]
    best_record = max(
        abnormal_records,
        key=lambda record: float(record["region_dark_pct"] or 0),
        default=None,
    )
    if best_record:
        confidence = round(
            min(1.0, float(best_record["region_dark_pct"]) / 100.0),
            3,
        )
        reason = (
            f"连续帧命中黑屏区域；最高暗像素占比 {float(best_record['region_dark_pct']):.1f}%，"
            f"bbox=[{best_record['bbox_x']},{best_record['bbox_y']},"
            f"{best_record['bbox_w']},{best_record['bbox_h']}]"
        )
    else:
        confidence = 0.0
        reason = "未命中：视频帧未出现满足规则的黑屏候选区域"
    return {
        "input": str(path),
        "issue_id": path.parent.name,
        "media_type": "video",
        "highlighted_video": str(out_video),
        "highlighted_file": str(out_video),
        "contact_sheet": str(out_sheet),
        "csv": str(out_csv),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps if fps else None,
        "abnormal_intervals_sec": intervals,
        "abnormal_frame_count": sum(1 for record in records if record["abnormal"]),
        "abnormal_type": "black_screen" if intervals else "",
        "start_time": intervals[0][0] if intervals else "",
        "end_time": intervals[-1][1] if intervals else "",
        "confidence": confidence,
        "reason": reason,
    }


def process_image(path: Path, tmpdir: Path):
    source = ascii_work_path(path, tmpdir)
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Could not open image: {path}")

    stem = path.stem.encode("ascii", "ignore").decode("ascii") or "image"
    parent = path.parent.name.encode("ascii", "ignore").decode("ascii")
    prefix = f"{parent}_{stem}" if parent and path.parent != ROOT else stem
    out_image = OUT / f"{prefix}_highlighted.png"
    result = detect_dark_region(frame)
    annotated = draw_annotation(frame, result, 0.0)
    cv2.imwrite(str(out_image), annotated)

    confidence, reason = confidence_and_reason(result)
    return {
        "input": str(path),
        "issue_id": path.parent.name,
        "media_type": "image",
        "highlighted_file": str(out_image),
        "highlighted_video": "",
        "contact_sheet": "",
        "csv": "",
        "fps": "",
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "frame_count": 1,
        "duration_sec": 0,
        "abnormal_intervals_sec": [[0, 0]] if result["abnormal"] else [],
        "abnormal_frame_count": 1 if result["abnormal"] else 0,
        "abnormal_type": "black_screen" if result["abnormal"] else "",
        "start_time": 0 if result["abnormal"] else "",
        "end_time": 0 if result["abnormal"] else "",
        "confidence": confidence,
        "reason": reason,
    }


def write_review_table(summary, path):
    fields = [
        "issue_id",
        "input",
        "media_type",
        "abnormal_type",
        "start_time",
        "end_time",
        "confidence",
        "reason",
        "highlighted_file",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            writer.writerow({field: item.get(field, "") for field in fields})


def main():
    global ROOT, OUT

    parser = argparse.ArgumentParser(description="Detect black screen regions in videos/images.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Input root directory.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Scan input root recursively.")
    args = parser.parse_args()

    ROOT = args.root
    OUT = args.out or ROOT / "detected_results"
    if not ROOT.is_dir():
        parser.error(
            f"输入目录不存在: {ROOT}\n"
            "用 --root 指定，或设置环境变量 CAMERA_ALGO_DATA_ROOT，"
            "或在仓库下创建 data_source/ 并放入待测视频。")
    OUT.mkdir(parents=True, exist_ok=True)
    paths = ROOT.rglob("*") if args.recursive else ROOT.iterdir()
    media = [
        path
        for path in paths
        if path.is_file()
        and OUT not in path.parents
        and path.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS
        and path.name.lower() != "black_screen_sample.mp4"
    ]
    summary = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for path in media:
            if path.suffix.lower() in VIDEO_EXTS:
                summary.append(process_video(path, tmpdir))
            else:
                summary.append(process_image(path, tmpdir))

    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_review_table(summary, OUT / "review_table.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
