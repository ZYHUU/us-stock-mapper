from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from websockets.asyncio.client import connect

from src.database import (
    DEFAULT_DB_PATH,
    PROJECT_ROOT,
    initialize,
    save_message,
    save_shadow_prediction,
    source_key,
)
from src.mapper import default_mapper
from src.message_parser import identify_event
from src.semantic_matcher import SemanticMatcher, default_lightgbm_matcher, default_lr_matcher


# 从项目根目录的 .env 加载环境变量（.env 已在 .gitignore 中，不会被提交）。
load_dotenv(PROJECT_ROOT / ".env")

# 具体的 WS 端点地址不写死在源码里（属于上游数据源信息，不对外公开），
# 改为从环境变量读取；未设置时用明显的占位符，直接连接会失败并提示配置。
DEFAULT_WS_URL = os.environ.get("WS_URL", "wss://YOUR-WS-HOST/YOUR-PATH")
# 旧端点：仅推送已命中股票名/代码正则的消息。仍可能需要，用 --url 指定后继续采集。
LEGACY_WS_URL = os.environ.get("WS_LEGACY_URL", "wss://YOUR-WS-HOST/YOUR-LEGACY-PATH")
# 未经股票名正则过滤的完整新闻网站源。token 是敏感信息，不写死在源码里，
# 运行前设置环境变量 WS_NEWS_TOKEN，连接时会自动替换 URL 里的 {token} 占位符。
NEWS_WS_URL = os.environ.get(
    "WS_NEWS_URL", "wss://YOUR-NEWS-HOST/websocket?token={token}&topics=news"
)
PREDICTION_VERSION = "rules-v2"


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        name, separator, header_value = value.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"请求头格式应为 NAME=VALUE，收到：{value}")
        headers[name.strip()] = header_value.strip()

    authorization = os.environ.get("WS_AUTHORIZATION")
    cookie = os.environ.get("WS_COOKIE")
    if authorization:
        headers["Authorization"] = authorization
    if cookie:
        headers["Cookie"] = cookie
    return headers


def _resolve_url(url: str) -> str:
    """把 URL 里的 {token} 占位符换成环境变量 WS_NEWS_TOKEN，避免把 token 写进源码或命令行历史。"""
    if "{token}" not in url:
        return url
    token = os.environ.get("WS_NEWS_TOKEN")
    if not token:
        raise ValueError(
            "URL 中包含 {token} 占位符，但环境变量 WS_NEWS_TOKEN 未设置"
        )
    return url.replace("{token}", token)


def _events_from_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(payload, dict):
        return

    # 心跳/协议控制消息（例如 {"type": "pong"}），不是真实事件。
    if (
        payload.get("type") in {"ping", "pong"}
        and "data" not in payload
        and "result" not in payload
    ):
        return

    # 有些服务会把真正的事件包在 data 字段中，新闻端点则包在 result 字段中。
    wrapped = payload.get("data")
    if wrapped is None:
        wrapped = payload.get("result")

    if isinstance(wrapped, dict):
        event = dict(wrapped)
        for field in ("stoks", "stocks"):
            if field not in event and field in payload:
                event[field] = payload[field]
        yield event
    elif isinstance(wrapped, list):
        for item in wrapped:
            if isinstance(item, dict):
                event = dict(item)
                for field in ("stoks", "stocks"):
                    if field not in event and field in payload:
                        event[field] = payload[field]
                yield event
    else:
        yield payload


def _event_key(event: dict[str, Any]) -> str:
    return source_key(
        {
            "source_id": event.get("id_str") or event.get("id") or event.get("new_id"),
            "raw_event": event,
        }
    )


def _save_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def consume(
    url: str,
    database: Path,
    jsonl_backup: Path | None,
    headers: dict[str, str],
    origin: str | None,
    subscribe_message: str | None,
    shadow_matchers: list[SemanticMatcher],
) -> None:
    mapper = default_mapper()
    code_to_company = {c.canonical_code: c for c in mapper.companies}
    ticker_to_code = {c.ticker: c.canonical_code for c in mapper.companies}
    seen: set[str] = set()
    initialize(database)

    async for websocket in connect(
        url,
        additional_headers=headers or None,
        origin=origin,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=15,
    ):
        print(f"已连接：{url}")
        try:
            if subscribe_message:
                await websocket.send(subscribe_message)
                print("已发送订阅消息")

            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8")

                try:
                    payload = json.loads(raw_message)
                except (json.JSONDecodeError, TypeError):
                    print("跳过非 JSON 消息")
                    continue

                for event in _events_from_payload(payload):
                    key = _event_key(event)
                    if key in seen:
                        continue
                    seen.add(key)

                    result = identify_event(event, mapper)
                    record = {
                        "received_at": datetime.now(timezone.utc).isoformat(),
                        "source_id": result["source_id"],
                        "source_type": result["source_type"],
                        "source_url": result["source_url"],
                        "published_at": result["published_at"],
                        "text": result["text"],
                        "predicted_codes": [
                            company["canonical_code"]
                            for company in result["companies"]
                        ],
                        "correct_codes": None,
                        "status": result["status"],
                        "companies": result["companies"],
                        "upstream_candidates": result["upstream_candidates"],
                        "raw_event": event,
                    }
                    message_id, inserted = save_message(
                        record,
                        model_version=PREDICTION_VERSION,
                        path=database,
                    )
                    if not inserted:
                        continue
                    if jsonl_backup is not None:
                        _save_record(jsonl_backup, record)

                    upstream_codes = {
                        ticker_to_code[str(candidate.get("stocker_code") or "")]
                        for candidate in result["upstream_candidates"]
                        if str(candidate.get("stocker_code") or "") in ticker_to_code
                    }
                    for shadow_matcher in shadow_matchers:
                        # 影子运行：只记录模型的判断，不影响上面已经写好的正式输出。
                        try:
                            scored = shadow_matcher.score(
                                result["text"],
                                result["source_type"],
                                result["matches"],
                                code_to_company,
                                upstream_codes=upstream_codes,
                            )
                            model_codes = [
                                item.canonical_code for item in scored if item.model_predicted
                            ]
                            save_shadow_prediction(
                                message_id=message_id,
                                model_version=shadow_matcher.model_version,
                                rule_codes=record["predicted_codes"],
                                model_codes=model_codes,
                                candidates=[item.to_dict() for item in scored],
                                path=database,
                            )
                        except Exception as error:
                            print(
                                f"影子运行打分失败（{shadow_matcher.model_version}，不影响正式输出）：{error}"
                            )

                    codes = record["predicted_codes"]
                    display = ", ".join(codes) if codes else "no_match"
                    print(f'{record["source_id"] or "无ID"}: {display}')
        except Exception as error:
            print(f"连接中断，准备自动重连：{error}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="监听 WS 消息并识别美股公司")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="WebSocket 地址")
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite 数据库文件",
    )
    parser.add_argument(
        "--jsonl-backup",
        type=Path,
        help="可选的 JSONL 追加备份文件",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="非敏感握手请求头，可重复传入",
    )
    parser.add_argument("--origin", help="服务端要求的 Origin 请求头")
    parser.add_argument(
        "--subscribe",
        help="连接成功后发送的订阅消息，通常是 JSON 字符串",
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help=(
            "启用影子运行：额外用 LR 基线模型给候选打分并记录到 shadow_predictions 表，"
            "不影响正式输出的 predicted_codes"
        ),
    )
    parser.add_argument(
        "--shadow-lightgbm",
        action="store_true",
        help=(
            "额外启用 LightGBM 影子模型（测试集准确率更高，但多候选消息的 p99 延迟"
            "略超100ms，所以暂不作为线上基线，只在影子运行里观察）"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        headers = _parse_headers(args.header)
        url = _resolve_url(args.url)
    except ValueError as error:
        parser.error(str(error))

    shadow_matchers: list[SemanticMatcher] = []
    if args.shadow:
        matcher = default_lr_matcher()
        shadow_matchers.append(matcher)
        print(f"影子运行已启用（LR 基线），模型版本：{matcher.model_version}")
    if args.shadow_lightgbm:
        matcher = default_lightgbm_matcher()
        shadow_matchers.append(matcher)
        print(f"影子运行已启用（LightGBM），模型版本：{matcher.model_version}")

    try:
        asyncio.run(
            consume(
                url=url,
                database=args.database,
                jsonl_backup=args.jsonl_backup,
                headers=headers,
                origin=args.origin,
                subscribe_message=args.subscribe,
                shadow_matchers=shadow_matchers,
            )
        )
    except KeyboardInterrupt:
        print("已停止监听")


if __name__ == "__main__":
    main()
