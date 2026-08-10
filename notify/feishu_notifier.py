"""
飞书 Webhook 告警模块（display_monitor/notify 第一版）
=====================================================
Webhook 从环境变量读取，绝不硬编码：

  # Linux / macOS (bash|zsh) —— 可直接 source set_feishu_env.sh
  export FEISHU_WEBHOOK="https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
  export FEISHU_WEBHOOK_SECRET="签名密钥"    # 机器人开了"签名校验"才需要
  # 卡片内嵌截图需要企业自建应用凭证(权限: im:resource 上传图片):
  export FEISHU_APP_ID="cli_xxx"
  export FEISHU_APP_SECRET="xxx"

  # 另一条路: 不用群机器人 webhook, 直接用应用往指定群发消息
  # (需权限 im:message:send_as_bot, 且把该应用拉进这个群)
  export FEISHU_CHAT_ID="oc_xxxx"

  # Windows PowerShell
  $env:FEISHU_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"
  $env:FEISHU_WEBHOOK_SECRET = "签名密钥"
  $env:FEISHU_APP_ID = "cli_xxx"
  $env:FEISHU_APP_SECRET = "xxx"

用法:
  from notify.feishu_notifier import send_text, send_event
  send_text("测试消息")
  send_event(event_dict)          # event.json 的内容, 发红色告警卡片

命令行自测:
  python3 notify/feishu_notifier.py --test                 # 发一条测试文本
  python3 notify/feishu_notifier.py --event <event.json>   # 发送事件卡片
  python3 notify/feishu_notifier.py --event <event.json> --dry-run  # 只打印payload不发送

说明:
- 仅用标准库(urllib), 无第三方依赖。
- 卡片消息(msg_type=interactive)默认; --text 降级为纯文本。
- 截图直传飞书需要应用凭证(image_key), 属第二步; 当前证据路径为本地文件。
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime

ENV_WEBHOOK = "FEISHU_WEBHOOK"
ENV_SECRET = "FEISHU_WEBHOOK_SECRET"
ENV_CHAT_ID = "FEISHU_CHAT_ID"
TIMEOUT_S = 10
RETRIES = 2


class FeishuError(RuntimeError):
    pass


def _webhook():
    url = os.environ.get(ENV_WEBHOOK, "").strip()
    if not url:
        raise FeishuError(
            f"环境变量 {ENV_WEBHOOK} 未设置。\n"
            f"  bash/zsh:   export {ENV_WEBHOOK}=\"<webhook url>\"  (或 source set_feishu_env.sh)\n"
            f"  PowerShell: $env:{ENV_WEBHOOK}=\"<webhook url>\"")
    return url


def _sign_fields():
    """机器人开启签名校验时, 附加 timestamp + sign 字段。"""
    secret = os.environ.get(ENV_SECRET, "").strip()
    if not secret:
        return {}
    ts = str(int(time.time()))
    string_to_sign = f"{ts}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), b"", hashlib.sha256).digest()
    return {"timestamp": ts, "sign": base64.b64encode(digest).decode("utf-8")}


def config_status():
    """当前可用的发送方式。返回 {mode, target, ready, problem}。

    两条路二选一：
      webhook —— 群里加「自定义机器人」，只需一个 URL，最省事
      app     —— 企业自建应用往指定群（chat_id）发，需要 app_id/app_secret，
                 且应用要被拉进该群、有 im:message:send_as_bot 权限
    只有 chat_id 而没有应用凭证是发不出去的：chat_id 只是收件地址，不是凭证。
    """
    hook = os.environ.get(ENV_WEBHOOK, "").strip()
    if hook:
        tail = hook.rsplit("/", 1)[-1]
        return {"mode": "webhook", "ready": True,
                "target": f"…{tail[-6:]}" if len(tail) > 6 else tail, "problem": None}

    chat = os.environ.get(ENV_CHAT_ID, "").strip()
    app_id = os.environ.get(ENV_APP_ID, "").strip()
    app_secret = os.environ.get(ENV_APP_SECRET, "").strip()
    if chat and app_id and app_secret:
        return {"mode": "app", "ready": True, "target": chat, "problem": None}

    if chat and not (app_id and app_secret):
        return {"mode": "app", "ready": False, "target": chat,
                "problem": f"已配置群 {chat}，但缺 {ENV_APP_ID}/{ENV_APP_SECRET}；"
                           f"chat_id 只是收件地址，还需要应用凭证才能发送"}
    return {"mode": None, "ready": False, "target": None,
            "problem": f"未配置 {ENV_WEBHOOK}（群自定义机器人 URL），"
                       f"也未配置 {ENV_APP_ID}/{ENV_APP_SECRET}+{ENV_CHAT_ID}"}


def _post_via_app(payload, dry_run=False):
    """用应用凭证往 chat_id 发。与 webhook 的报文结构不同：
    content 必须是 JSON 字符串，且 receive_id 走 query 参数指定类型。"""
    chat = os.environ.get(ENV_CHAT_ID, "").strip()
    msg_type = payload.get("msg_type", "text")
    content = payload.get("card") if msg_type == "interactive" else payload.get("content")
    body_obj = {"receive_id": chat, "msg_type": msg_type,
                "content": json.dumps(content, ensure_ascii=False)}
    if dry_run:
        print("[dry-run] im/v1/messages payload:")
        print(json.dumps(body_obj, ensure_ascii=False, indent=2))
        return {"code": 0, "dry_run": True}

    token = _tenant_token()
    if token is None:
        raise FeishuError(f"未配置 {ENV_APP_ID}/{ENV_APP_SECRET}，无法用 chat_id 发送")
    last_err = None
    for attempt in range(1 + RETRIES):
        try:
            req = urllib.request.Request(
                "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
                data=json.dumps(body_obj, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8",
                         "Authorization": f"Bearer {token}"})
            data = _read_json(req, TIMEOUT_S)
            if data.get("code") == 0:
                return data
            raise FeishuError(f"飞书返回错误: {data}")
        except (urllib.error.URLError, TimeoutError, FeishuError) as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise FeishuError(f"发送失败(已重试{RETRIES}次): {last_err}")


def _post(payload, dry_run=False):
    if not os.environ.get(ENV_WEBHOOK, "").strip() \
            and os.environ.get(ENV_CHAT_ID, "").strip():
        return _post_via_app(payload, dry_run=dry_run)

    payload = {**_sign_fields(), **payload}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if dry_run:
        print("[dry-run] payload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return {"code": 0, "dry_run": True}
    url = _webhook()
    last_err = None
    for attempt in range(1 + RETRIES):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            # 自定义机器人成功: {"code":0,...} 或旧版 {"StatusCode":0}
            if data.get("code", data.get("StatusCode", -1)) == 0:
                return data
            raise FeishuError(f"飞书返回错误: {data}")
        except (urllib.error.URLError, TimeoutError, FeishuError) as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise FeishuError(f"发送失败(已重试{RETRIES}次): {last_err}")


ENV_APP_ID = "FEISHU_APP_ID"
ENV_APP_SECRET = "FEISHU_APP_SECRET"


def _read_json(req, timeout):
    """urlopen 并解析 JSON；HTTP 4xx/5xx 时读取响应体给出飞书具体错误。"""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = ""
        raise FeishuError(f"HTTP {e.code}: {detail[:300]}") from None


def _tenant_token():
    """用企业自建应用凭证换 tenant_access_token；未配置返回 None。"""
    app_id = os.environ.get(ENV_APP_ID, "").strip()
    app_secret = os.environ.get(ENV_APP_SECRET, "").strip()
    if not app_id or not app_secret:
        return None
    body = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode("utf-8")
    req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        data=body, headers={"Content-Type": "application/json"})
    data = _read_json(req, TIMEOUT_S)
    if data.get("code") != 0:
        raise FeishuError(f"获取 tenant_access_token 失败: {data}")
    return data["tenant_access_token"]


def upload_image(image_path):
    """上传图片到飞书, 返回 image_key。需要 FEISHU_APP_ID/FEISHU_APP_SECRET。"""
    token = _tenant_token()
    if token is None:
        raise FeishuError(f"未配置 {ENV_APP_ID}/{ENV_APP_SECRET}, 无法上传图片")
    boundary = "----feishu_notifier_boundary"
    with open(image_path, "rb") as f:
        img = f.read()
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"image_type\"\r\n\r\nmessage\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"image\"; filename=\"evidence.jpg\"\r\n"
                 f"Content-Type: image/jpeg\r\n\r\n".encode())
    parts.append(img)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        "https://open.feishu.cn/open-apis/im/v1/images", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {token}"})
    data = _read_json(req, TIMEOUT_S * 2)
    if data.get("code") != 0:
        raise FeishuError(f"上传图片失败: {data}")
    return data["data"]["image_key"]


def _fmt_ts(ms):
    if ms is None:
        return "-"
    ms = int(ms)
    return f"{ms//3600000:02d}:{ms%3600000//60000:02d}:{ms%60000//1000:02d}.{ms%1000:03d}"


def send_text(text, dry_run=False):
    """发送纯文本消息。"""
    return _post({"msg_type": "text", "content": {"text": text}}, dry_run=dry_run)


# 每种异常一套标题与卡片配色：群里一眼分得清是黑屏还是花屏，
# 全都发"Black Screen Detected"的话，多类型同时开就没法从通知里区分
EVENT_STYLE = {
    "black_screen":     ("🚨 黑屏 · Black Screen", "red"),
    "white_screen":     ("⚪ 白屏 · White Screen", "orange"),
    "screen_flicking":  ("⚡ 闪屏 · Screen Flicking", "yellow"),
    "screen_distorted": ("🌈 花屏 · Screen Distorted", "purple"),
    "screen_freeze":    ("🧊 卡顿 · Screen Freeze", "blue"),
}


def _style(event):
    typ = event.get("event_type") or "black_screen"
    title, tmpl = EVENT_STYLE.get(typ, (f"🚨 {typ.upper()}", "red"))
    no = event.get("screen_no")
    if no:
        total = event.get("screen_total")
        title += f" · S{no}" + (f"/{total}" if total else "")
    return title, tmpl


def event_to_text(event):
    """事件 -> 文本消息（卡片被禁用时的降级格式）。"""
    return (
        f"{_style(event)[0]}\n"
        f"事件：{event.get('event_type', 'black_screen').upper()} ({event.get('event_id', '-')})\n"
        f"来源：{event.get('source', '-')}\n"
        f"视频位置：{_fmt_ts(event.get('source_timestamp_ms'))}\n"
        f"检测时间：{event.get('capture_time', '-')}\n"
        f"帧序号：{event.get('frame_index', '-')}\n"
        f"置信分数：{event.get('score', '-')}\n"
        f"证据路径：{event.get('screenshot', '-')}"
    )


def event_to_card(event, image_key=None):
    """事件 -> 飞书消息卡片(红色告警头), image_key 非空时内嵌截图。"""
    bbox = event.get("bbox")
    fields = [
        ("事件", f"{event.get('event_type', 'black_screen').upper()}  `{event.get('event_id', '-')}`"),
        ("屏幕", (f"S{event['screen_no']}" + (f" / 共 {event['screen_total']} 块"
                                              if event.get("screen_total") else ""))
                 if event.get("screen_no") else "整幅画面"),
        ("来源", str(event.get("source", "-"))),
        ("视频位置", _fmt_ts(event.get("source_timestamp_ms"))),
        ("检测时间", str(event.get("capture_time", "-"))),
        ("帧序号", str(event.get("frame_index", "-"))),
        ("置信分数", str(event.get("score", "-"))),
        ("检出区域", str(bbox) if bbox else "-"),
        ("证据路径", str(event.get("screenshot", "-"))),
    ]
    md = "\n".join(f"**{k}：**{v}" for k, v in fields)
    elements = [{"tag": "markdown", "content": md}]
    if image_key:
        elements.append({"tag": "img", "img_key": image_key,
                         "alt": {"tag": "plain_text", "content": "异常证据拼图"}})
    title, template = _style(event)
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": template,
                "title": {"tag": "plain_text", "content": title},
            },
            "elements": elements + [
                {"tag": "note", "elements": [{
                    "tag": "plain_text",
                    "content": f"display_monitor · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}]},
            ],
        },
    }


def send_event(event, card=True, dry_run=False, image_path=None):
    """发送黑屏事件告警。event 为 event.json 的 dict。
    image_path 非空且配置了应用凭证时, 截图内嵌进卡片; 否则降级为纯文字卡片。"""
    image_key = None
    if card and image_path and not dry_run:
        try:
            image_key = upload_image(image_path)
        except Exception as e:
            print(f"  ⚠️ 截图上传失败(卡片将不带图): {e}")
    if card:
        return _post(event_to_card(event, image_key=image_key), dry_run=dry_run)
    return send_text(event_to_text(event), dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="飞书 Webhook 告警自测")
    parser.add_argument("--test", action="store_true", help="发送一条测试文本")
    parser.add_argument("--event", help="发送指定 event.json")
    parser.add_argument("--text", action="store_true", help="用纯文本代替卡片")
    parser.add_argument("--dry-run", action="store_true", help="只打印 payload 不发送")
    parser.add_argument("--upload-test", help="单独测试图片上传, 参数为图片路径")
    args = parser.parse_args()

    if args.upload_test:
        print("image_key:", upload_image(args.upload_test))
        return

    if args.test:
        r = send_text(f"✅ feishu_notifier 测试消息 {datetime.now():%Y-%m-%d %H:%M:%S}",
                      dry_run=args.dry_run)
        print("发送结果:", r)
    elif args.event:
        with open(args.event, encoding="utf-8") as f:
            event = json.load(f)
        r = send_event(event, card=not args.text, dry_run=args.dry_run,
                       image_path=event.get("screenshot"))
        print("发送结果:", r)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
