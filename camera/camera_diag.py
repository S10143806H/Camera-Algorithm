"""
USB2.0 摄像头诊断工具 —— 修复黑屏问题
核心修复：强制 MJPG 四字节码 + 遍历 backend/fourcc/分辨率，专门用来解决“摄像头打开了却是黑屏”的问题
跨平台：Windows(MSMF/DSHOW) / Linux(V4L2) / macOS(AVFoundation)，无需 pywin32。
依赖：pip install -r ../requirements.txt
运行：python3 camera_diag.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from platform_compat import (
    capture_backends,
    list_cameras,
    load_cjk_font,
    privacy_hint,
)

CHINESE_FONT = load_cjk_font()


def put_chinese_text(frame, text, position=(10, 5)):
    """在 OpenCV 图像上绘制中文文字。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.text(
        position,
        text,
        font=CHINESE_FONT,
        fill=(0, 255, 0),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(image)


def get_camera_symlinks():
    """列出系统摄像头设备（Windows: PowerShell / Linux: /dev/video* / macOS: 探测）。"""
    return list_cameras()


def try_open_camera(idx, backend, fourcc=None, width=None, height=None):
    """尝试用指定参数打开摄像头，返回 (cap, info_dict) 或 (None, info)"""
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        cap.release()
        return None, {"idx": idx, "backend": backend, "error": "无法打开"}

    # 设置 FOURCC
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 读取实际参数
    info = {
        "idx": idx,
        "backend": cap.getBackendName(),
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    fourcc_int = info["fourcc"]
    info["fourcc_str"] = "".join(chr((fourcc_int >> 8*i) & 0xFF) for i in range(4))

    # 预热读帧
    for _ in range(30):
        cap.read()

    # 采集10帧取平均
    means = []
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            means.append(frame.mean())

    info["mean"] = sum(means) / len(means) if means else -1
    info["ok"] = info["mean"] > 5.0

    if not info["ok"]:
        cap.release()
        return None, info

    return cap, info


def diagnose_all():
    """遍历所有 index × backend × fourcc 组合，找到可用画面"""
    print("=" * 60)
    print("  🔍 摄像头诊断 — 遍历所有组合")
    print("=" * 60)

    # 先列出系统设备
    devices = get_camera_symlinks()
    print(f"\n📋 系统摄像头设备 ({len(devices)} 个):")
    for d in devices:
        print(f"  📷 {d['name']}")
        print(f"     ID:  {d['instance_id']}")
        print(f"     Sym: {d['symlink'][:80]}")

    # 测试矩阵
    backends = capture_backends()
    fourccs = [
        ("MJPG", "MJPG"),
        ("YUY2", "YUY2"),
        (None,   "默认"),
    ]
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
    ]

    best = None  # (cap, info)

    for idx in range(4):
        for be_code, be_name in backends:
            for fc_str, fc_label in fourccs:
                info_try = {"idx": idx, "backend": be_name}
                cap = cv2.VideoCapture(idx, be_code)
                if not cap.isOpened():
                    cap.release()
                    continue

                if fc_str:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fc_str))

                # 获取实际参数
                actual_fc = int(cap.get(cv2.CAP_PROP_FOURCC))
                actual_fc_str = "".join(chr((actual_fc >> 8*i) & 0xFF) for i in range(4))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 预热
                for _ in range(30):
                    cap.read()

                # 采样
                means = []
                for _ in range(15):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        means.append(frame.mean())

                avg = sum(means) / len(means) if means else -1
                status = "🟢" if avg > 5 else "⚫"

                print(f"  {status} idx={idx} {be_name:4s} fourcc={fc_label:4s}→{actual_fc_str:4s} "
                      f"{w}x{h}@{fps:.0f}fps mean={avg:.1f}")

                if avg > 5 and best is None:
                    best = (cap, {
                        "idx": idx, "backend": be_name, "fourcc": actual_fc_str,
                        "w": w, "h": h, "fps": fps, "mean": avg,
                    })
                else:
                    cap.release()

                if best:
                    break
            if best:
                break
        if best:
            break

    # 也尝试不同分辨率
    if best is None:
        print("\n  ⚠️  默认格式全黑，尝试不同分辨率...")
        for idx in range(4):
            for be_code, be_name in backends:
                for w, h in resolutions:
                    for fc_str, fc_label in fourccs:
                        cap = cv2.VideoCapture(idx, be_code)
                        if not cap.isOpened():
                            cap.release()
                            continue
                        if fc_str:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fc_str))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

                        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        actual_fc_str = "".join(chr((actual_fc >> 8*i) & 0xFF) for i in range(4))

                        for _ in range(30):
                            cap.read()
                        means = []
                        for _ in range(10):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                means.append(frame.mean())
                        avg = sum(means) / len(means) if means else -1
                        status = "🟢" if avg > 5 else "⚫"
                        print(f"  {status} idx={idx} {be_name} {fc_label}→{actual_fc_str} "
                              f"{aw}x{ah} mean={avg:.1f}")

                        if avg > 5 and best is None:
                            best = (cap, {"idx": idx, "backend": be_name, "mean": avg,
                                          "fourcc": actual_fc_str, "w": aw, "h": ah})
                        else:
                            cap.release()
                        if best:
                            break
                    if best:
                        break
                if best:
                    break
            if best:
                break

    return best


def show_preview(cap, info):
    """显示预览画面"""
    print(f"\n  ✅ 找到可用画面!")
    print(f"     idx={info['idx']} backend={info['backend']} "
          f"fourcc={info.get('fourcc','?')} {info.get('w','?')}x{info.get('h','?')}")
    print(f"     mean={info['mean']:.1f}")
    print(f"  📺 预览窗口已打开，按 Q 关闭\n")

    win = f"USB2.0 Camera [{info['backend']}] - Press Q"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        overlay = (
            f"时间 {timestamp} | 平均亮度 {frame.mean():.0f} | "
            f"后端 {info['backend']}"
        )
        frame = put_chinese_text(frame, overlay)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    result = diagnose_all()
    if result:
        cap, info = result
        show_preview(cap, info)
    else:
        privacy_hint()


if __name__ == "__main__":
    main()
