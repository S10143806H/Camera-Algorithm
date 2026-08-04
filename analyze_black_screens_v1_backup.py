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


def detect_dark_region(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    dark_pct = float((gray < 40).mean() * 100)

    # Threshold dark reflective displays, then merge neighboring chunks into screen-sized regions.
    mask = (gray < 60).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    frame_area = w * h
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        center_x = x + bw / 2
        if area < frame_area * 0.04:
            continue
        if center_x < w * 0.32:
            continue
        if bw < w * 0.12 or bh < h * 0.15:
            continue
        aspect = bw / max(bh, 1)
        if not 0.25 <= aspect <= 3.2:
            continue
        roi = gray[y : y + bh, x : x + bw]
        roi_mean = float(roi.mean())
        roi_dark = float((roi < 60).mean() * 100)
        if roi_mean < 75 and roi_dark > 55:
            candidates.append(
                {
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "area": int(area),
                    "mean": roi_mean,
                    "dark_pct": roi_dark,
                }
            )

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

    records = []
    contact_frames = []
    sample_every = max(1, int(round(fps * 0.5)))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = idx / fps
        result = detect_dark_region(frame)
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
