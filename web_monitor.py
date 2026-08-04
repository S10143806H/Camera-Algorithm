"""
屏幕异常实时监控 —— Web 版（局域网内浏览器访问）
================================================
把采集 + 多屏黑屏检测跑成后台线程，通过 HTTP 对外提供：
  * MJPEG 实时画面（带 S1/S2/S3 标注与红框告警）
  * 网页内鼠标框选 ROI、一键自动标定
  * 网页内调相机参数（亮度/对比度/增益/曝光/对焦/自动曝光）
  * 事件列表与证据截图

相机同一时刻只能被一个进程占用：本服务运行期间不要再开 camera_diag.py 预览。

用法:
  python3 web_monitor.py                          # 0.0.0.0:8000, 自动挑相机
  python3 web_monitor.py --device 0 --screens 3
  python3 web_monitor.py --port 8080 --roi 20,60,300,200;340,40,320,220
  python3 web_monitor.py --notify                 # 事件推飞书(需 FEISHU_WEBHOOK)

访问: http://<本机IP>:8000/
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from camera_diag import (  # noqa: E402
    CAM_PARAMS,
    annotate_live_multi,
    diagnose_all,
    emit_merged_event,
    format_rois,
    open_device,
    order_rois,
    parse_rois,
    run_detect_multi,
)
from platform_compat import set_auto_exposure  # noqa: E402

try:
    from analyze_black_screens import calibrate_screen_rois, rois_from_maxbright
    HAVE_DETECTOR = True
except ImportError:
    HAVE_DETECTOR = False

ROI_STORE = ROOT / "screen_rois.json"
EVIDENCE_N = 10
EVIDENCE_MIN = 3
EVIDENCE_GAP = 0.4
COOLDOWN_S = 10.0
MAX_EVENTS_KEPT = 200


# ---------------------------------------------------------------- ROI 持久化
def load_rois():
    """读取上次保存的 ROI，使网页里标定过一次后重启仍生效。"""
    try:
        data = json.loads(ROI_STORE.read_text(encoding="utf-8"))
        return [tuple(int(v) for v in r) for r in data.get("rois", [])]
    except Exception:
        return []


def save_rois(rois):
    try:
        ROI_STORE.write_text(
            json.dumps({"rois": [list(r) for r in rois],
                        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds")},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError as e:
        print(f"⚠️ ROI 保存失败: {e}")


# ---------------------------------------------------------------- 监控服务
class MonitorService:
    """唯一的相机持有者：采集 → 逐屏检测 → 标注 → 编码 JPEG 供 HTTP 取用。"""

    def __init__(self, cap, info, max_screens=3, rois=None, detect=True,
                 notifier=None, jpeg_quality=75):
        self.cap = cap
        self.info = info
        self.max_screens = max(1, max_screens)
        self.detect_on = detect and HAVE_DETECTOR
        self.notifier = notifier
        self.jpeg_quality = jpeg_quality

        self.lock = threading.Lock()          # 保护 _jpeg / _raw / results / rois
        self.frame_event = threading.Event()  # 有新帧时唤醒所有 MJPEG 订阅者
        self._jpeg = None
        self._raw = None
        self.rois = list(rois or [])
        self.results = []
        self.fps = 0.0
        self.running = True

        self.events = []                      # 最近事件（内存），完整证据在磁盘
        self.event_seq = 0
        self._screen_state = {}
        self._calib_buf = []                  # 滚动标定缓存（每 0.5s 一帧，最多 40）
        self._last_calib_sample = 0.0
        self.event_root = (ROOT / "diag_captures" /
                           f"web_{datetime.now().strftime('%Y%m%d')}" / "black_screen")
        self.source_name = f"camera_{info.get('idx', '?')}"
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ---------------------------------------------------------- 生命周期
    def start(self):
        self._thread.start()

    def stop(self):
        self.running = False
        self._thread.join(timeout=3)
        self.cap.release()

    # ---------------------------------------------------------- 主循环
    def _state_of(self, no):
        return self._screen_state.setdefault(
            no, {"streak": 0, "cooldown_until": 0.0, "evidence": [], "last_evi_t": -1e9})

    def _loop(self):
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        t_last, n = time.time(), 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - self._last_calib_sample > 0.5:
                self._calib_buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                del self._calib_buf[:-40]
                self._last_calib_sample = now

            with self.lock:
                rois = list(self.rois)
                detect_on = self.detect_on

            results = run_detect_multi(frame, rois) if detect_on else []
            if detect_on:
                self._gate_events(frame, results, rois)

            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            display = (annotate_live_multi(frame, results, ts) if results
                       else self._plain(frame, rois, ts))

            ok, buf = cv2.imencode(".jpg", display, enc)
            if not ok:
                continue

            n += 1
            if now - t_last >= 1.0:
                self.fps, n, t_last = n / (now - t_last), 0, now

            with self.lock:
                self._jpeg = buf.tobytes()
                self._raw = frame
                self.results = results
            self.frame_event.set()
            self.frame_event.clear()

    def _plain(self, frame, rois, ts):
        from camera_diag import _put_text_clamped, draw_screen_boxes
        out = frame.copy()
        if rois:
            draw_screen_boxes(out, list(enumerate(rois, 1)))
        _put_text_clamped(out, ts, 12, out.shape[0] - 40, 0.7, (0, 215, 255))
        return out

    def _gate_events(self, frame, results, rois):
        """逐屏事件门控：连续命中攒够证据帧 → 落盘一次合并事件 → 进入冷却。"""
        wc = datetime.now()
        ts = wc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        cur = time.time()
        for no, roi, r in results:
            st = self._state_of(no)
            if r["abnormal"]:
                st["streak"] += 1
                if cur >= st["cooldown_until"] and cur - st["last_evi_t"] >= EVIDENCE_GAP:
                    st["evidence"].append((frame.copy(), r, ts, None, None, wc))
                    st["last_evi_t"] = cur
                    if len(st["evidence"]) >= EVIDENCE_N:
                        self._emit(st, no, roi, len(results))
            else:
                if len(st["evidence"]) >= EVIDENCE_MIN and cur >= st["cooldown_until"]:
                    self._emit(st, no, roi, len(results))
                st["evidence"] = []
                st["streak"] = 0

    def _emit(self, st, no, roi, total):
        self.event_seq += 1
        evidence, seq = st["evidence"], self.event_seq
        st["evidence"] = []
        st["cooldown_until"] = time.time() + COOLDOWN_S

        def work():
            try:
                ev = emit_merged_event(evidence, self.source_name, self.event_root, seq,
                                       notifier=self.notifier, screen_no=no,
                                       screen_roi=roi, screen_total=total)
                with self.lock:
                    self.events.append(ev)
                    del self.events[:-MAX_EVENTS_KEPT]
            except Exception as e:
                print(f"⚠️ 事件生成失败: {e}")

        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------- 对外操作
    def snapshot_jpeg(self):
        with self.lock:
            return self._jpeg

    def calibrate(self):
        """用滚动缓存的最大亮度图自动标定多块屏幕。"""
        if not HAVE_DETECTOR:
            raise RuntimeError("未找到 analyze_black_screens.py，无法自动标定")
        if not self._calib_buf:
            raise RuntimeError("标定缓存为空，等几秒再试")
        mb = self._calib_buf[0].copy()
        for g in self._calib_buf:
            mb = np.maximum(mb, g)
        rois = rois_from_maxbright(mb, max_screens=self.max_screens)
        self.set_rois(rois)
        return rois

    def set_rois(self, rois):
        rois = order_rois([tuple(int(v) for v in r) for r in rois])[:self.max_screens]
        with self.lock:
            self.rois = rois
            self.results = []
            self._screen_state.clear()
        save_rois(rois)
        return rois

    def set_detect(self, on):
        with self.lock:
            self.detect_on = bool(on) and HAVE_DETECTOR
            if not self.detect_on:
                self.results = []
        return self.detect_on

    def camera_params(self):
        out = []
        for name, prop, vmax, conv in CAM_PARAMS:
            cur = self.cap.get(prop)
            val = int(min(max(cur if conv(1) >= 0 else -cur, 0), vmax))
            out.append({"name": name, "value": val, "max": vmax})
        ae = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        out.append({"name": "AutoExp 0/1", "value": 0 if ae in (0, 1, 0.25) else 1, "max": 1})
        return out

    def set_camera_param(self, name, value):
        if name == "AutoExp 0/1":
            set_auto_exposure(self.cap, bool(int(value)))
            return True
        for pname, prop, vmax, conv in CAM_PARAMS:
            if pname == name:
                v = int(min(max(int(value), 0), vmax))
                self.cap.set(prop, conv(v))
                return True
        return False

    def status(self):
        with self.lock:
            results, rois = list(self.results), list(self.rois)
            n_events = len(self.events)
        screens = [{"no": no,
                    "roi": list(roi) if roi else None,
                    "abnormal": bool(r["abnormal"]),
                    "dark_pct": round(float(r["region"]["dark_pct"]), 1) if r.get("region") else 0.0}
                   for no, roi, r in results]
        return {
            "device": self.info.get("idx"),
            "backend": self.info.get("backend"),
            "width": int(self.info.get("w") or 0),
            "height": int(self.info.get("h") or 0),
            "fps": round(self.fps, 1),
            "detect_on": self.detect_on,
            "max_screens": self.max_screens,
            "rois": [list(r) for r in rois],
            "roi_arg": format_rois(rois),
            "screens": screens,
            "event_count": n_events,
            "have_detector": HAVE_DETECTOR,
        }


# ---------------------------------------------------------------- HTTP
service: MonitorService = None
server = None                     # uvicorn.Server，供 /stream 感知退出信号
app = FastAPI(title="Screen Anomaly Monitor")


def _should_stop():
    """服务要退出了吗。uvicorn 优雅关闭会先等所有连接结束，而 /stream 是
    无限生成器；不看 server.should_exit 的话，浏览器挂着流时 SIGTERM 会永久
    挂住，只能 kill -9。"""
    return (not service) or (not service.running) or (server is not None and server.should_exit)


class RoisIn(BaseModel):
    rois: list


class ParamIn(BaseModel):
    name: str
    value: int


class DetectIn(BaseModel):
    on: bool


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")


@app.get("/stream")
async def stream():
    """MJPEG 多部件流，浏览器 <img src="/stream"> 直接显示。"""
    boundary = "frame"

    async def gen():
        last = None
        try:
            while not _should_stop():
                jpg = service.snapshot_jpeg()
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield (b"--" + boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                           + jpg + b"\r\n")
                await asyncio.sleep(0.03)
        except asyncio.CancelledError:      # 客户端断开：正常收尾，不刷栈
            pass

    return StreamingResponse(
        gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@app.get("/api/status")
def api_status():
    return service.status()


@app.get("/api/rois")
def api_get_rois():
    st = service.status()
    return {"rois": st["rois"], "roi_arg": st["roi_arg"], "max_screens": st["max_screens"]}


@app.post("/api/rois")
def api_set_rois(body: RoisIn):
    bad = [r for r in body.rois if len(r) != 4]
    if bad:
        raise HTTPException(400, "每块 ROI 需为 [x, y, w, h]")
    rois = service.set_rois(body.rois)
    return {"rois": [list(r) for r in rois], "roi_arg": format_rois(rois)}


@app.delete("/api/rois")
def api_clear_rois():
    service.set_rois([])
    return {"rois": [], "roi_arg": ""}


@app.post("/api/rois/calibrate")
def api_calibrate():
    try:
        rois = service.calibrate()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    if not rois:
        raise HTTPException(422, "自动标定失败：屏幕未点亮或亮区过小，请手动框选")
    return {"rois": [list(r) for r in rois], "roi_arg": format_rois(rois)}


@app.get("/api/camera")
def api_get_camera():
    return {"params": service.camera_params()}


@app.post("/api/camera")
def api_set_camera(body: ParamIn):
    if not service.set_camera_param(body.name, body.value):
        raise HTTPException(404, f"未知参数: {body.name}")
    return {"ok": True, "params": service.camera_params()}


@app.post("/api/detect")
def api_detect(body: DetectIn):
    return {"detect_on": service.set_detect(body.on)}


@app.get("/api/events")
def api_events(limit: int = 50):
    with service.lock:
        evs = list(service.events)[-max(1, min(limit, MAX_EVENTS_KEPT)):]
    return {"events": list(reversed(evs))}


@app.get("/api/events/{event_id}/screenshot")
def api_event_shot(event_id: str):
    with service.lock:
        match = next((e for e in service.events if e["event_id"] == event_id), None)
    if not match:
        raise HTTPException(404, "事件不存在")
    p = Path(match["screenshot"])
    if not p.exists():
        raise HTTPException(404, "截图文件已被清理")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/snapshot")
def api_snapshot():
    """当前帧单张 JPEG，便于外部脚本抓图或做健康检查。"""
    jpg = service.snapshot_jpeg()
    if jpg is None:
        raise HTTPException(503, "尚无画面")
    return StreamingResponse(iter([jpg]), media_type="image/jpeg")


# ---------------------------------------------------------------- main
def main():
    global service
    ap = argparse.ArgumentParser(description="屏幕异常实时监控 Web 服务")
    ap.add_argument("--device", type=int, default=None, help="相机 index，缺省则自动诊断")
    ap.add_argument("--screens", type=int, default=3, help="最多识别几块屏幕（默认3）")
    ap.add_argument("--roi", help="启动时固定 ROI: x,y,w,h 多块用分号分隔")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址（默认全网卡）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-detect", action="store_true", help="只看画面，不做检测")
    ap.add_argument("--notify", action="store_true", help="事件推送飞书（需 FEISHU_WEBHOOK）")
    ap.add_argument("--quality", type=int, default=75, help="MJPEG 质量 1-100（默认75）")
    args = ap.parse_args()

    opened = open_device(args.device) if args.device is not None else diagnose_all()
    if not opened:
        print("❌ 打不开相机，Web 服务未启动")
        return 1
    cap, info = opened
    print(f"✅ 相机就绪: idx={info.get('idx')} {info.get('backend')} "
          f"{info.get('w')}x{info.get('h')}")

    rois = parse_rois(args.roi) if args.roi else load_rois()
    if rois:
        print(f"🖥️ 载入 ROI ({len(rois)} 块): {format_rois(rois)}")

    notifier = None
    if args.notify:
        try:
            from notify.feishu_notifier import send_event
            notifier = send_event
            print("📨 飞书告警: 开")
        except Exception as e:
            print(f"⚠️ 飞书告警不可用: {e}")

    service = MonitorService(cap, info, max_screens=args.screens, rois=rois,
                             detect=not args.no_detect, notifier=notifier,
                             jpeg_quality=args.quality)
    service.start()

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); lan = s.getsockname()[0]; s.close()
    except OSError:
        lan = "127.0.0.1"
    print(f"\n🌐 打开浏览器访问:  http://{lan}:{args.port}/   (本机 http://127.0.0.1:{args.port}/)")
    print("   相机被本服务独占，运行期间不要再开 camera_diag.py 预览\n")

    import uvicorn
    global server
    # 自建 Server 以便 /stream 读到 should_exit；再加优雅关闭超时兜底
    server = uvicorn.Server(uvicorn.Config(
        app, host=args.host, port=args.port, log_level="warning",
        timeout_graceful_shutdown=5))
    try:
        server.run()
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
