"""
跨平台兼容层 —— 摄像头后端 / 设备枚举 / 自动曝光 / 中文字体
=========================================================
原实现仅支持 Windows（MSMF/DSHOW + PowerShell 枚举 + C:\\Windows\\Fonts）。
本模块按 sys.platform 分派，使同一套算法脚本可在 Linux(V4L2) / macOS(AVFoundation)
/ Windows 上直接运行，调用方无需再写平台分支。

对外接口:
  IS_WINDOWS / IS_LINUX / IS_MACOS   平台布尔量
  capture_backends()                 -> [(cv2.CAP_*, "名称"), ...] 按平台优先级排序
  list_cameras()                     -> [{"name","instance_id","symlink"}, ...]
  set_auto_exposure(cap, enable)     按平台写入正确的 AUTO_EXPOSURE 取值
  load_cjk_font(size)                -> PIL ImageFont，找不到时回落默认字体
  privacy_hint()                     -> 全黑画面时打印的平台化排查提示
"""

import glob
import os
import subprocess
import sys

import cv2

IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


# ---------------------------------------------------------------- 采集后端
def capture_backends():
    """返回当前平台可用的 VideoCapture 后端，按成功率优先级排序。"""
    if IS_WINDOWS:
        return [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW")]
    if IS_MACOS:
        return [(cv2.CAP_AVFOUNDATION, "AVFOUNDATION"), (cv2.CAP_ANY, "ANY")]
    # Linux: V4L2 为 UVC 相机主路径，GStreamer 作为兜底
    return [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "ANY")]


# ---------------------------------------------------------------- 设备枚举
def _list_cameras_windows():
    ps = r"""
$devs = Get-PnpDevice -Class Camera -Status OK
foreach ($d in $devs) {
    $name = $d.FriendlyName
    $iid  = $d.InstanceId
    $props = Get-PnpDeviceProperty -InstanceId $iid -KeyName 'DEVPKEY_Device_SymbolicLink' -ErrorAction SilentlyContinue
    $sym = if ($props) { $props.Data } else { "" }
    if (-not $sym) {
        $ifaces = Get-PnpDeviceInterface -InstanceId $iid -ErrorAction SilentlyContinue
        $sym = ($ifaces | Select-Object -First 1).SymbolicLink
    }
    "$name||$iid||$sym"
}
"""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps], text=True, timeout=20)
    except Exception as e:
        print(f"  ❌ PowerShell 查询失败: {e}")
        return []
    devices = []
    for line in out.strip().splitlines():
        if "||" in line:
            parts = line.strip().split("||")
            devices.append({"name": parts[0].strip(),
                            "instance_id": parts[1].strip() if len(parts) > 1 else "",
                            "symlink": parts[2].strip() if len(parts) > 2 else ""})
    return devices


def _v4l2_name(node):
    """读取 /sys 中的 v4l2 设备名，失败时回落节点名。"""
    base = os.path.basename(node)
    for path in (f"/sys/class/video4linux/{base}/name",):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        except OSError:
            pass
    return base


def _is_capture_node(node):
    """过滤 metadata 节点：UVC 相机常成对出现 videoN(采集)/videoN+1(metadata)，
    sysfs 的 index 属性为 0 的才是主采集节点；无该属性时不过滤。"""
    base = os.path.basename(node)
    try:
        with open(f"/sys/class/video4linux/{base}/index", "r") as f:
            return f.read().strip() == "0"
    except OSError:
        return True


def _list_cameras_linux():
    """枚举 /dev/video*，名称取自 sysfs，无需 root 或额外依赖。"""
    devices = []
    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
    for node in nodes:
        if not _is_capture_node(node):
            continue
        idx = "".join(c for c in os.path.basename(node) if c.isdigit())
        devices.append({"name": _v4l2_name(node),
                        "instance_id": f"index={idx}",
                        "symlink": node})
    return devices


def _list_cameras_macos():
    """macOS 无稳定的免依赖枚举方式，退化为按 index 探测。"""
    devices = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            devices.append({"name": f"AVFoundation camera {idx}",
                            "instance_id": f"index={idx}", "symlink": ""})
        cap.release()
    return devices


def list_cameras():
    """列出系统摄像头设备，字段与原 Windows 版保持一致。"""
    if IS_WINDOWS:
        return _list_cameras_windows()
    if IS_MACOS:
        return _list_cameras_macos()
    return _list_cameras_linux()


# ---------------------------------------------------------------- 自动曝光
def set_auto_exposure(cap, enable):
    """
    各后端的 AUTO_EXPOSURE 取值互不兼容，逐一写入直到生效：
      DSHOW  0.75=自动 0.25=手动
      MSMF   1=自动    0=手动
      V4L2   3=自动    1=手动（V4L2_EXPOSURE_AUTO / _MANUAL）
    """
    if IS_LINUX:
        values = (3, 1)
    elif IS_WINDOWS:
        values = ((0.75, 1) if enable else (0.25, 0))
        for v in values:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
        return
    else:
        values = (1, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, values[0] if enable else values[1])


# ---------------------------------------------------------------- 中文字体
_CJK_FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    # Linux (Noto CJK / 文泉驿 / 思源黑体)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def _fc_match_cjk():
    """用 fontconfig 兜底查询系统中文字体路径。"""
    try:
        out = subprocess.check_output(
            ["fc-match", "-f", "%{file}", ":lang=zh"], text=True, timeout=5).strip()
        return out or None
    except Exception:
        return None


def load_cjk_font(size=24):
    """加载中文字体（PIL ImageFont），全部失败时回落 PIL 默认位图字体。"""
    from PIL import ImageFont

    candidates = list(_CJK_FONT_CANDIDATES)
    fc = _fc_match_cjk()
    if fc:
        candidates.append(fc)
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------- 排查提示
def privacy_hint():
    """全部组合黑屏时的平台化排查建议。"""
    print("\n  ❌ 所有组合均黑屏！可能原因：")
    print("     1. 摄像头被其他程序占用")
    print("     2. 驱动/固件问题")
    if IS_WINDOWS:
        print("     3. 摄像头隐私开关未打开（物理开关或 Windows 隐私设置）")
        print("  💡 Windows 设置 → 隐私 → 相机 → 允许应用访问相机 = 开")
    elif IS_LINUX:
        print("     3. 当前用户无 /dev/video* 读写权限")
        print("  💡 检查: ls -l /dev/video*  ;  加入 video 组: sudo usermod -aG video $USER (需重新登录)")
        print("  💡 排查: v4l2-ctl --list-devices  ;  v4l2-ctl -d /dev/video0 --list-formats-ext")
    elif IS_MACOS:
        print("     3. 未授予终端/Python 相机权限")
        print("  💡 系统设置 → 隐私与安全性 → 相机 → 勾选运行本脚本的终端")
