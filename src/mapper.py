from __future__ import annotations

import csv
import re
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .database import DEFAULT_DB_PATH, connect, initialize


@dataclass(frozen=True)
class Company:
    company_id: str
    company_name: str
    exchange: str
    ticker: str
    aliases: tuple[str, ...]
    brands: tuple[str, ...]
    negative_contexts: tuple[str, ...]

    @property
    def canonical_code(self) -> str:
        return f"{self.exchange}:{self.ticker}"


@dataclass(frozen=True)
class Match:
    company_id: str
    company_name: str
    canonical_code: str
    mention: str
    match_type: str
    confidence: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "canonical_code": self.canonical_code,
            "mention": self.mention,
            "match_type": self.match_type,
            "confidence": self.confidence,
        }


def _split_keywords(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split("|") if item.strip())


def load_companies(path: str | Path) -> list[Company]:
    companies: list[Company] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            companies.append(
                Company(
                    company_id=row["company_id"],
                    company_name=row["company_name"],
                    exchange=row["exchange"],
                    ticker=row["ticker"],
                    aliases=_split_keywords(row["aliases"]),
                    brands=_split_keywords(row["brands"]),
                    negative_contexts=_split_keywords(row["negative_contexts"]),
                )
            )
    return companies


def _automatic_aliases(name_en: str, name_cn: str) -> tuple[str, ...]:
    aliases = [value.strip() for value in (name_en, name_cn) if value.strip()]
    if name_en:
        simplified = re.sub(r"[,\.]", "", name_en).strip()
        simplified = re.sub(
            r"\s+(?:Inc|Corp|Corporation|Co|Ltd|PLC|N\s*V|S\s*p\s*A)$",
            "",
            simplified,
            flags=re.IGNORECASE,
        ).strip()
        if len(simplified) >= 4 and simplified.casefold() != name_en.casefold():
            aliases.append(simplified)
    return tuple(dict.fromkeys(aliases))


def load_enabled_companies(
    csv_path: str | Path,
    database_path: str | Path = DEFAULT_DB_PATH,
) -> list[Company]:
    """合并 SQLite 主数据和 CSV 人工规则，以 SQLite 的标准代码为准。"""
    manual_companies = load_companies(csv_path)
    manual_by_ticker = {company.ticker: company for company in manual_companies}
    merged_by_ticker: dict[str, Company] = {}

    database_path = Path(database_path)
    initialize(database_path)
    with closing(connect(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT canonical_code, exchange, ticker, name_en, name_cn
            FROM securities
            WHERE mapper_candidate = 1
              AND asset_type = 'stock'
            ORDER BY market, exchange, ticker
            """
        ).fetchall()

    for row in rows:
        ticker = str(row["ticker"])
        manual = manual_by_ticker.get(ticker)
        automatic_aliases = _automatic_aliases(
            str(row["name_en"]),
            str(row["name_cn"]),
        )
        if manual:
            aliases = tuple(dict.fromkeys((*manual.aliases, *automatic_aliases)))
            merged_by_ticker[ticker] = Company(
                company_id=manual.company_id,
                company_name=manual.company_name,
                exchange=str(row["exchange"]),
                ticker=ticker,
                aliases=aliases,
                brands=manual.brands,
                negative_contexts=manual.negative_contexts,
            )
            continue

        canonical_code = str(row["canonical_code"])
        merged_by_ticker[ticker] = Company(
            company_id=re.sub(r"[^a-z0-9]+", "_", canonical_code.casefold()).strip("_"),
            company_name=str(row["name_cn"] or row["name_en"] or ticker),
            exchange=str(row["exchange"]),
            ticker=ticker,
            aliases=automatic_aliases,
            brands=(),
            negative_contexts=(),
        )

    for manual in manual_companies:
        merged_by_ticker.setdefault(manual.ticker, manual)
    return list(merged_by_ticker.values())


def _contains(text: str, keyword: str) -> bool:
    """中文使用子串匹配，英文和股票代码要求完整单词匹配。"""
    if re.search(r"[一-鿿]", keyword):
        return keyword.casefold() in text.casefold()

    pattern = rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _contains_ticker(text: str, ticker: str) -> bool:
    """股票代码要求大写，避免把命令行的 -v 等普通小写词识别为代码。"""
    prefix = r"\$" if len(ticker) == 1 else r"\$?"
    pattern = rf"(?<![A-Za-z0-9]){prefix}{re.escape(ticker)}(?![A-Za-z0-9])"
    return re.search(pattern, text) is not None


_CJK_PATTERN = re.compile(r"[一-鿿]")


@dataclass(frozen=True)
class _CompiledTerm:
    """预编译的关键词：中文只存casefold后的字符串用于子串匹配，
    英文/代码预先编译好正则，避免 identify() 每次调用都重新构造和编译。"""

    keyword: str
    match_type: str
    confidence: float
    casefolded: str | None
    pattern: re.Pattern[str] | None

    def matches(self, text: str, text_casefold: str) -> bool:
        if self.casefolded is not None:
            return self.casefolded in text_casefold
        return self.pattern.search(text) is not None


def _compile_term(keyword: str, match_type: str, confidence: float) -> _CompiledTerm:
    if _CJK_PATTERN.search(keyword):
        return _CompiledTerm(keyword, match_type, confidence, keyword.casefold(), None)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])", flags=re.IGNORECASE
    )
    return _CompiledTerm(keyword, match_type, confidence, None, pattern)


def _compile_ticker_pattern(ticker: str) -> re.Pattern[str]:
    prefix = r"\$" if len(ticker) == 1 else r"\$?"
    return re.compile(rf"(?<![A-Za-z0-9]){prefix}{re.escape(ticker)}(?![A-Za-z0-9])")


@dataclass(frozen=True)
class _CompiledCompany:
    company: Company
    terms: tuple[_CompiledTerm, ...]
    negative_terms: tuple[_CompiledTerm, ...]
    ticker_pattern: re.Pattern[str]


def _compile_company(company: Company) -> _CompiledCompany:
    terms = tuple(
        [_compile_term(keyword, "alias", 0.98) for keyword in company.aliases]
        + [_compile_term(keyword, "brand", 0.90) for keyword in company.brands]
    )
    negative_terms = tuple(
        _compile_term(keyword, "", 0.0) for keyword in company.negative_contexts
    )
    return _CompiledCompany(
        company=company,
        terms=terms,
        negative_terms=negative_terms,
        ticker_pattern=_compile_ticker_pattern(company.ticker),
    )


class USStockMapper:
    def __init__(self, companies: list[Company]):
        self.companies = companies
        self._compiled_companies = [_compile_company(c) for c in companies]

    @classmethod
    def from_csv(cls, path: str | Path) -> "USStockMapper":
        return cls(load_companies(path))

    def identify(self, message: str) -> list[Match]:
        message = message.strip()
        if not message:
            return []

        message_for_matching = re.sub(r"https?://\S+", " ", message)
        # 消息的大小写归一化只算一次，供本条消息下所有公司的中文关键词复用，
        # 避免像优化前那样对同一条消息反复 casefold() 几百次。
        message_casefold = message_for_matching.casefold()

        matches: list[Match] = []
        for compiled in self._compiled_companies:
            company = compiled.company
            if any(
                term.matches(message_for_matching, message_casefold)
                for term in compiled.negative_terms
            ):
                continue

            found = [
                (term.keyword, term.match_type, term.confidence)
                for term in compiled.terms
                if term.matches(message_for_matching, message_casefold)
            ]
            if compiled.ticker_pattern.search(message_for_matching):
                found.append((company.ticker, "ticker", 0.99))
            if not found:
                continue

            mention, match_type, confidence = max(
                found,
                key=lambda item: (item[2], len(item[0])),
            )
            matches.append(
                Match(
                    company_id=company.company_id,
                    company_name=company.company_name,
                    canonical_code=company.canonical_code,
                    mention=mention,
                    match_type=match_type,
                    confidence=confidence,
                )
            )

        return sorted(matches, key=lambda item: item.confidence, reverse=True)


def default_mapper() -> USStockMapper:
    project_root = Path(__file__).resolve().parents[1]
    return USStockMapper(
        load_enabled_companies(
            project_root / "data" / "companies.csv",
            DEFAULT_DB_PATH,
        )
    )
