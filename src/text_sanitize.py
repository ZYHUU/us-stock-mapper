from __future__ import annotations

import re
from typing import Iterable

from src.mapper import Company

# 训练和线上推理必须用同一份脱敏逻辑（同一个函数），否则会重演之前踩过的
# candidates_in_message train/serve 偏差坑——这次脱敏在
# build_training_dataset.py（训练特征落盘时）和 semantic_matcher.py
# （SemanticMatcher.score() 里，线上打分前）分别调用，但都是这一个函数。

_URL_PATTERN = re.compile(r"https?://\S+")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_HANDLE_PATTERN = re.compile(r"@(\w{2,})")
# 6位及以上连续数字：区块号/追踪ID/雪花ID等常见长度，但要放过已注册的纯数字
# 股票代码（比如A股688825、韩交所000660）——这些不是可识别个人信息，是模型
# 需要的真实信号。前面加 (?<!\$) 避免误伤紧跟在 $ 后面的金额数字。
_LONG_DIGIT_PATTERN = re.compile(r"(?<!\$)\b\d{6,}\b")


def build_handle_to_company_name(companies: Iterable[Company]) -> dict[str, str]:
    """把"看起来像官方账号"的 @handle 精确匹配（不是子串）到公司名，这样脱敏时
    能把 @Tesla 这种真官方/品牌账号替换成公司名保留语义信号，同时把
    @Tesla_Teslaway 这种粉丝账号（handle 整体跟公司别名不完全相等）当成普通
    handle 脱敏掉——这正好也修正了标注时反复遇到的"handle 子串误判"问题
    （见项目记忆 project_annotation_false_positive_patterns.md），而不是加重它。"""
    mapping: dict[str, str] = {}
    for company in companies:
        names = {company.ticker, company.company_name}
        names.update(company.aliases)
        names.update(company.brands)
        for name in names:
            key = re.sub(r"[^a-z0-9]", "", name.lower())
            if len(key) >= 2:
                mapping.setdefault(key, company.company_name)
    return mapping


def collect_numeric_tickers(companies: Iterable[Company]) -> set[str]:
    return {company.ticker for company in companies if company.ticker.isdigit()}


def sanitize_message_text(
    text: str,
    handle_to_company_name: dict[str, str] | None = None,
    numeric_tickers: set[str] | None = None,
) -> str:
    """把喂给 TF-IDF 向量化器和结构化特征提取的文本里，可能识别到具体个人/账号
    的片段（URL、邮箱、@handle、长数字ID）替换成占位符，只保留公司名/别名/
    股票代码这类分类任务真正需要的信号。不修改 mapper.identify() 用的原始文本
    ——候选召回（规则层）需要看到完整原文，脱敏只发生在文本进入分类器特征
    提取的这一步。"""
    handle_to_company_name = handle_to_company_name or {}
    numeric_tickers = numeric_tickers or set()

    text = _URL_PATTERN.sub(" [URL] ", text)
    text = _EMAIL_PATTERN.sub(" [EMAIL] ", text)

    def _replace_handle(match: re.Match[str]) -> str:
        key = match.group(1).lower()
        return handle_to_company_name.get(key, "[HANDLE]")

    text = _HANDLE_PATTERN.sub(_replace_handle, text)

    def _replace_digits(match: re.Match[str]) -> str:
        run = match.group(0)
        return run if run in numeric_tickers else "[ID]"

    text = _LONG_DIGIT_PATTERN.sub(_replace_digits, text)
    return text
