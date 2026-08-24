from __future__ import annotations

import argparse
import csv
import json
import ssl
import urllib.request
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import DEFAULT_DB_PATH, connect, initialize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "binance_tradfi_instruments.csv"
BINANCE_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
SEC_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
PLATFORM = "binance_futures"

EXCHANGE_NAMES = {"Nasdaq": "NASDAQ", "NYSE": "NYSE", "CBOE": "CBOE"}
SEC_TICKER_OVERRIDES = {"BRKB": "BRK-B"}

ETF_NAMES = {
    "BITO": "ProShares Bitcoin Strategy ETF",
    "DRAM": "Roundhill Memory ETF",
    "EWJ": "iShares MSCI Japan ETF",
    "EWT": "iShares MSCI Taiwan ETF",
    "EWY": "iShares MSCI South Korea ETF",
    "EWZ": "iShares MSCI Brazil ETF",
    "GDX": "VanEck Gold Miners ETF",
    "IWM": "iShares Russell 2000 ETF",
    "INTW": "GraniteShares 2x Long INTC Daily ETF",
    "KORU": "Direxion Daily South Korea Bull 3X Shares",
    "KSTR": "KraneShares SSE STAR Market 50 Index ETF",
    "LYTE": "Roundhill Photonics & Optics ETF",
    "MUU": "Direxion Daily MU Bull 2X Shares",
    "MVLL": "GraniteShares 2x Long MRVL Daily ETF",
    "QQQ": "Invesco QQQ Trust",
    "SMH": "VanEck Semiconductor ETF",
    "SNXX": "Xtrackers US National Critical Technologies ETF",
    "SOXL": "Direxion Daily Semiconductor Bull 3X Shares",
    "SOXS": "Direxion Daily Semiconductor Bear 3X Shares",
    "SQQQ": "ProShares UltraPro Short QQQ",
    "SPY": "SPDR S&P 500 ETF Trust",
    "STXX": "Tradr 2X Long STX Daily ETF",
    "TBT": "ProShares UltraShort 20+ Year Treasury",
    "TMF": "Direxion Daily 20+ Year Treasury Bull 3X Shares",
    "TQQQ": "ProShares UltraPro QQQ",
    "TZA": "Direxion Daily Small Cap Bear 3X Shares",
    "URNM": "Sprott Uranium Miners ETF",
    "UVXY": "ProShares Ultra VIX Short-Term Futures ETF",
    "XBI": "SPDR S&P Biotech ETF",
    "XLE": "Energy Select Sector SPDR Fund",
}

# asset_type, market, exchange, ticker, English name, Chinese name
SPECIAL_SECURITIES = {
    "OPENAI": ("private_company", "PRIVATE", "PRIVATE", "OPENAI", "OpenAI", "OpenAI"),
    "ANTHROPIC": ("private_company", "PRIVATE", "PRIVATE", "ANTHROPIC", "Anthropic", "Anthropic"),
    "CXMT": ("stock", "CN", "SSE", "688825", "CXMT Corporation", "长鑫科技"),
    "UNITREE": ("stock", "CN", "SSE", "688836", "Unitree Robotics", "宇树科技"),
    "QNTX": ("stock", "US", "NASDAQ", "QNT", "Quantinuum Inc.", "Quantinuum"),
    "BBX": ("stock", "US", "NYSE", "BB", "BlackBerry Limited", "黑莓"),
    "SPCX": ("stock", "US", "NASDAQ", "SPCX", "SpaceX", "太空探索技术公司"),
    "MINIMAX": ("stock", "HK", "HKEX", "0100", "MiniMax Group Inc.", "MiniMax"),
    "ZHIPU": ("stock", "HK", "HKEX", "2513", "Zhipu AI", "智谱"),
    "HK0700": ("stock", "HK", "HKEX", "0700", "Tencent Holdings", "腾讯控股"),
    "TENCENT": ("stock", "HK", "HKEX", "0700", "Tencent Holdings", "腾讯控股"),
    "HK1810": ("stock", "HK", "HKEX", "1810", "Xiaomi Corporation", "小米集团"),
    "POPMART": ("stock", "HK", "HKEX", "9992", "Pop Mart International", "泡泡玛特"),
    "GIGADEV": ("stock", "HK", "HKEX", "3986", "GigaDevice Semiconductor", "兆易创新"),
    "KUAISHOU": ("stock", "HK", "HKEX", "1024", "Kuaishou Technology", "快手"),
    "MEITUAN": ("stock", "HK", "HKEX", "3690", "Meituan", "美团"),
    "ZHONGJI": ("stock", "HK", "HKEX", "3308", "Zhongji Innolight", "中际旭创"),
    "CSOPSKHYNIX2L": ("etf", "HK", "HKEX", "CSOPSKHYNIX2L", "CSOP SK Hynix 2x Leveraged Product", "南方东英SK海力士两倍杠杆产品"),
    "CSOPSAMSUNG2L": ("etf", "HK", "HKEX", "CSOPSAMSUNG2L", "CSOP Samsung 2x Leveraged Product", "南方东英三星两倍杠杆产品"),
    "SKHYNIX": ("stock", "KR", "KRX", "000660", "SK hynix Inc.", "SK海力士"),
    "SAMSUNG": ("stock", "KR", "KRX", "005930", "Samsung Electronics", "三星电子"),
    "HYUNDAI": ("stock", "KR", "KRX", "005380", "Hyundai Motor", "现代汽车"),
    "SAMSUNGEM": ("stock", "KR", "KRX", "009150", "Samsung Electro-Mechanics", "三星电机"),
    "HANMI": ("stock", "KR", "KRX", "042700", "Hanmi Semiconductor", "韩美半导体"),
    "LGELECTRONICS": ("stock", "KR", "KRX", "066570", "LG Electronics", "LG电子"),
    "NAVER": ("stock", "KR", "KRX", "035420", "NAVER Corporation", "NAVER"),
    "KODEX200": ("etf", "KR", "KRX", "069500", "KODEX 200 ETF", "KODEX 200 ETF"),
}

COMMODITY_NAMES = {
    "XAU": ("Gold", "黄金"),
    "XAG": ("Silver", "白银"),
    "XPT": ("Platinum", "铂金"),
    "XPD": ("Palladium", "钯金"),
    "COPPER": ("Copper", "铜"),
    "CL": ("WTI Crude Oil", "WTI原油"),
    "BZ": ("Brent Crude Oil", "布伦特原油"),
    "NATGAS": ("Natural Gas", "天然气"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_json(url: str, insecure: bool = False, user_agent: str = "") -> Any:
    headers = {"Accept": "application/json"}
    if user_agent:
        headers["User-Agent"] = user_agent
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, context=context, timeout=30) as response:
        return json.load(response)


def load_sec_tickers(insecure: bool) -> dict[str, dict[str, str]]:
    payload = fetch_json(
        SEC_URL,
        insecure=insecure,
        user_agent="us-stock-mapper/1.0 local-use",
    )
    fields = payload["fields"]
    return {
        row[2]: dict(zip(fields, row, strict=True))
        for row in payload["data"]
    }


def canonical_code(exchange: str, ticker: str) -> str:
    if exchange in {"PRIVATE", "COMMODITY"}:
        return f"{exchange}:{ticker}"
    return f"{exchange}:{ticker}" if exchange and ticker else ""


def security_details(
    instrument: dict[str, Any],
    sec_tickers: dict[str, dict[str, str]],
) -> dict[str, Any]:
    base_asset = instrument["baseAsset"]
    underlying_type = instrument.get("underlyingType", "")

    if underlying_type == "COMMODITY":
        name_en, name_cn = COMMODITY_NAMES.get(base_asset, (base_asset, ""))
        return security_row(
            "commodity", "GLOBAL", "COMMODITY", base_asset, name_en, name_cn
        )

    if base_asset in SPECIAL_SECURITIES:
        return security_row(*SPECIAL_SECURITIES[base_asset])

    if base_asset in ETF_NAMES:
        return security_row("etf", "US", "US", base_asset, ETF_NAMES[base_asset], "")

    sec_ticker = SEC_TICKER_OVERRIDES.get(base_asset, base_asset)
    sec_record = sec_tickers.get(sec_ticker)
    if sec_record:
        exchange = EXCHANGE_NAMES.get(
            sec_record["exchange"], sec_record["exchange"].upper()
        )
        ticker = sec_record["ticker"].replace("-", ".")
        return security_row(
            "stock", "US", exchange, ticker, sec_record["name"], ""
        )

    return security_row(
        "unknown_equity", "", "BINANCE", base_asset, base_asset, ""
    )


def security_row(
    asset_type: str,
    market: str,
    exchange: str,
    ticker: str,
    name_en: str,
    name_cn: str,
) -> dict[str, Any]:
    return {
        "asset_type": asset_type,
        "market": market,
        "exchange": exchange,
        "ticker": ticker,
        "canonical_code": canonical_code(exchange, ticker),
        "name_en": name_en,
        "name_cn": name_cn,
        "mapper_candidate": int(
            asset_type == "stock" and market in {"US", "HK", "CN"}
        ),
    }


def format_onboard_date(value: Any) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(int(value) / 1000, timezone.utc).isoformat()


def build_rows(insecure: bool) -> list[dict[str, Any]]:
    exchange_info = fetch_json(BINANCE_URL, insecure=insecure)
    sec_tickers = load_sec_tickers(insecure)
    rows: list[dict[str, Any]] = []
    for instrument in exchange_info["symbols"]:
        if "TradFi" not in instrument.get("underlyingSubType", []):
            continue
        details = security_details(instrument, sec_tickers)
        rows.append(
            {
                **details,
                "platform": PLATFORM,
                "contract_symbol": instrument["symbol"],
                "base_asset": instrument.get("baseAsset", ""),
                "quote_asset": instrument.get("quoteAsset", ""),
                "underlying_type": instrument.get("underlyingType", ""),
                "underlying_subtypes": instrument.get("underlyingSubType", []),
                "status": instrument.get("status", ""),
                "onboard_date": format_onboard_date(instrument.get("onboardDate")),
                "active": int(instrument.get("status") == "TRADING"),
                "source_url": BINANCE_URL,
                "raw": instrument,
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["market"], row["asset_type"], row["contract_symbol"]),
    )


def save_to_database(rows: list[dict[str, Any]], db_path: Path) -> None:
    initialize(db_path)
    synced_at = utc_now()
    with closing(connect(db_path)) as connection, connection:
        connection.execute(
            "UPDATE platform_instruments SET active = 0 WHERE platform = ?",
            (PLATFORM,),
        )
        for row in rows:
            connection.execute(
                """
                INSERT INTO securities (
                    canonical_code, asset_type, market, exchange, ticker,
                    name_en, name_cn, mapper_candidate, source_url, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_code) DO UPDATE SET
                    asset_type = excluded.asset_type,
                    market = excluded.market,
                    exchange = excluded.exchange,
                    ticker = excluded.ticker,
                    name_en = excluded.name_en,
                    name_cn = CASE
                        WHEN excluded.name_cn <> '' THEN excluded.name_cn
                        ELSE securities.name_cn
                    END,
                    mapper_candidate = excluded.mapper_candidate,
                    source_url = excluded.source_url,
                    updated_at = excluded.updated_at
                """,
                (
                    row["canonical_code"], row["asset_type"], row["market"],
                    row["exchange"], row["ticker"], row["name_en"],
                    row["name_cn"], row["mapper_candidate"], row["source_url"],
                    synced_at,
                ),
            )
            security_id = connection.execute(
                "SELECT id FROM securities WHERE canonical_code = ?",
                (row["canonical_code"],),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO platform_instruments (
                    platform, contract_symbol, security_id, base_asset,
                    quote_asset, underlying_type, underlying_subtypes_json,
                    status, onboard_date, active, source_url, raw_json, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(platform, contract_symbol) DO UPDATE SET
                    security_id = excluded.security_id,
                    base_asset = excluded.base_asset,
                    quote_asset = excluded.quote_asset,
                    underlying_type = excluded.underlying_type,
                    underlying_subtypes_json = excluded.underlying_subtypes_json,
                    status = excluded.status,
                    onboard_date = excluded.onboard_date,
                    active = excluded.active,
                    source_url = excluded.source_url,
                    raw_json = excluded.raw_json,
                    synced_at = excluded.synced_at
                """,
                (
                    PLATFORM, row["contract_symbol"], security_id,
                    row["base_asset"], row["quote_asset"], row["underlying_type"],
                    json.dumps(row["underlying_subtypes"], ensure_ascii=False),
                    row["status"], row["onboard_date"], row["active"],
                    row["source_url"], json.dumps(row["raw"], ensure_ascii=False),
                    synced_at,
                ),
            )


def save_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "platform", "contract_symbol", "base_asset", "quote_asset",
        "asset_type", "market", "exchange", "ticker", "canonical_code",
        "name_en", "name_cn", "mapper_candidate", "underlying_type",
        "status", "onboard_date", "active", "source_url",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="同步币安 Futures TradFi 主数据")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="仅在本机证书链异常时跳过 TLS 校验",
    )
    args = parser.parse_args()

    rows = build_rows(insecure=args.insecure)
    save_to_database(rows, args.db)
    save_csv(rows, args.csv)
    active_rows = [row for row in rows if row["active"]]
    counts: dict[str, int] = {}
    for row in active_rows:
        counts[row["asset_type"]] = counts.get(row["asset_type"], 0) + 1
    print(
        json.dumps(
            {
                "total": len(rows),
                "active": len(active_rows),
                "by_asset_type": counts,
                "database": str(args.db),
                "csv": str(args.csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
