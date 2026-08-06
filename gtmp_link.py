"""
GTMP 任务上下文关联
===================
黑屏事件发生时，把当时 GTMP 上正在跑的任务信息一并记进证据，
用于回答"这次黑屏是哪个任务、哪个版本、跑到第几步时发生的"。

只做只读查询，不创建/修改/删除任何 GTMP 数据。

环境变量:
  GTMP_HOST   默认 http://gtmp-api.gua.com:9000
  GTMP_TOKEN  访问令牌，未设置时本模块自动禁用
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request

DEFAULT_HOST = "http://gtmp-api.gua.com:9000"
POLL_S = 15.0
TIMEOUT_S = 12.0


def _get(path, params=None):
    host = os.environ.get("GTMP_HOST", DEFAULT_HOST).rstrip("/")
    token = os.environ.get("GTMP_TOKEN", "")
    url = host + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"X-Token": token})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))


class GtmpLink:
    """后台轮询关联的 GTMP 任务，事件发生时提供快照。"""

    # GTMP task.status 取值含义（用于把数字翻成人话）
    STATUS = {0: "待执行", 1: "执行中", 2: "已完成", 3: "已取消", 4: "失败"}

    def __init__(self, task_id=None, bench_id=None):
        self.task_id = task_id
        self.bench_id = bench_id
        self.lock = threading.Lock()
        self.info = {"enabled": bool(os.environ.get("GTMP_TOKEN")),
                     "status": "未启动"}
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if not os.environ.get("GTMP_TOKEN"):
            with self.lock:
                self.info = {"enabled": False, "status": "GTMP_TOKEN 未设置，已跳过"}
            return
        self._thread.start()

    def stop(self):
        self.running = False

    def snapshot(self):
        with self.lock:
            return dict(self.info)

    # ------------------------------------------------------------ 内部
    def _loop(self):
        while self.running:
            try:
                self._refresh()
            except Exception as e:
                with self.lock:
                    self.info = {"enabled": True, "status": f"查询失败: {e}"}
            time.sleep(POLL_S)

    def _refresh(self):
        task = None
        if self.task_id:
            d = _get(f"/v1/crud/task/{self.task_id}")
            task = (d or {}).get("data")
        elif self.bench_id:
            # 找该台架上最近一个执行中的任务
            d = _get("/v1/crud/task", {"perPage": 20, "orderBy": "id", "orderDir": "desc",
                                       "status|multi-select": "0,1"})
            items = ((d or {}).get("data") or {}).get("items") or []
            for it in items:
                if self.bench_id in (it.get("benchIds") or []):
                    task = it
                    break
        if not task:
            with self.lock:
                self.info = {"enabled": True, "status": "未找到匹配的运行中任务",
                             "task_id": self.task_id, "bench_id": self.bench_id}
            return

        with self.lock:
            self.info = {
                "enabled": True,
                "status": "已关联",
                "task_id": task.get("id"),
                "task_name": task.get("name"),
                "task_status": self.STATUS.get(task.get("status"), task.get("status")),
                "progress": task.get("progress"),
                "version": task.get("version") or task.get("otaVersion"),
                "platform": task.get("platform"),
                "bench_ids": task.get("benchIds"),
                "device_ids": task.get("deviceIds"),
                "creator": task.get("userName") or task.get("creator"),
                "started_at": task.get("executeAt") or task.get("created_at"),
                "updated_at": task.get("updated_at"),
                "sampled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
