from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from src.build_training_dataset import build_candidate_profile
from src.database import PROJECT_ROOT
from src.mapper import Company, Match
from src.train_classic_model import PairExample, extract_structured_features

MODEL_ROOT = PROJECT_ROOT / "models" / "company_classifier"

# 注：v1 版本的阈值是在"候选数量"结构化特征存在训练/线上不一致bug时定的
# （训练时数了填充的负样本，线上却数真实候选数），已在这一轮修复并重新训练+
# 重新做过一次冻结测试集确认。下面是修复后的阈值。

# LR：当前线上基线 + 回退方案。所有场景的端到端延迟都稳定在100ms预算内
# （见 reports/pipeline_latency.md）。修复bug后测试集真实Precision比之前误报的数字更低，
# 离98%目标还有距离。
LR_MODEL_PATH = MODEL_ROOT / "tfidf_structured__balanced" / "model.joblib"
LR_THRESHOLD = 0.6555152247903872
LR_MODEL_VERSION = "classic-lr-v3"

# LightGBM：影子模型。修复bug后测试集 F1 依然比 LR 更好（0.959 vs 0.924），
# 但多候选消息的 p99 延迟接近/可能超出100ms预算，
# 所以先只在影子运行里观察，不接管正式输出。
LIGHTGBM_MODEL_PATH = MODEL_ROOT / "lgbm_svd_structured__balanced" / "model.joblib"
LIGHTGBM_THRESHOLD = 0.5605722797419384
LIGHTGBM_MODEL_VERSION = "lightgbm-shadow-v2"

# 向后兼容：旧代码里 import 的默认值，指向当前基线（LR）。
DEFAULT_MODEL_PATH = LR_MODEL_PATH
DEFAULT_THRESHOLD = LR_THRESHOLD
MODEL_VERSION = LR_MODEL_VERSION


@dataclass(frozen=True)
class ScoredMatch:
    canonical_code: str
    mention: str
    match_type: str
    rule_confidence: float
    model_score: float
    model_predicted: bool

    def to_dict(self) -> dict[str, str | float | bool]:
        return {
            "canonical_code": self.canonical_code,
            "mention": self.mention,
            "match_type": self.match_type,
            "rule_confidence": self.rule_confidence,
            "model_score": self.model_score,
            "model_predicted": self.model_predicted,
        }


class SemanticMatcher:
    """线上模型推理封装：对 mapper.identify() 已经召回的候选打分，
    不重新做候选召回，也不查询 SQLite（canonical_code 已经是标准代码）。
    同时支持两种模型 bundle：普通 TF-IDF(+结构化特征) 的 LR/LogisticRegression，
    以及 TF-IDF 先降维(SVD)再加结构化特征的 LightGBM。"""

    def __init__(
        self,
        model_path: Path = LR_MODEL_PATH,
        threshold: float = LR_THRESHOLD,
        model_version: str = LR_MODEL_VERSION,
    ):
        bundle = joblib.load(model_path)
        self.word_vectorizer = bundle["word_vectorizer"]
        self.char_vectorizer = bundle["char_vectorizer"]
        self.classifier = bundle["classifier"]
        self.svd = bundle.get("svd")
        self.use_structured = bool(bundle.get("use_structured", False)) or self.svd is not None
        self.threshold = threshold
        self.model_version = model_version

    def score(
        self,
        text: str,
        source_type: str,
        matches: list[Match],
        code_to_company: dict[str, Company],
        upstream_codes: set[str] | None = None,
    ) -> list[ScoredMatch]:
        if not matches:
            return []

        # 候选数量特征必须和训练时的算法一致：规则+上游候选的真实并集大小，
        # 不能只数本地规则命中数（否则会复现我们踩过的"候选数量特征"训练/线上不一致的坑）。
        real_candidate_count = len(
            {match.canonical_code for match in matches} | (upstream_codes or set())
        )

        pair_texts: list[str] = []
        structured_rows: list[list[float]] = []
        for match in matches:
            company = code_to_company[match.canonical_code]
            profile = build_candidate_profile(company)
            pair_texts.append(f"{text}\n[CANDIDATE]\n{profile}")
            if self.use_structured:
                example = PairExample(
                    message_id=0,
                    source_type=source_type,
                    message_text=text,
                    candidate_code=match.canonical_code,
                    candidate_profile=profile,
                    label=0,
                    negative_type="",
                    split="",
                )
                structured_rows.append(
                    extract_structured_features(example, company, real_candidate_count)
                )

        word_matrix = self.word_vectorizer.transform(pair_texts)
        char_matrix = self.char_vectorizer.transform(pair_texts)

        if self.svd is not None:
            tfidf = hstack([word_matrix, char_matrix]).tocsr()
            reduced = self.svd.transform(tfidf)
            structured = np.asarray(structured_rows, dtype=np.float64)
            X = np.hstack([reduced, structured])
        else:
            parts = [word_matrix, char_matrix]
            if self.use_structured:
                parts.append(csr_matrix(np.asarray(structured_rows, dtype=np.float64)))
            X = hstack(parts).tocsr()

        scores = self.classifier.predict_proba(X)[:, 1]
        return [
            ScoredMatch(
                canonical_code=match.canonical_code,
                mention=match.mention,
                match_type=match.match_type,
                rule_confidence=match.confidence,
                model_score=float(score),
                model_predicted=bool(score >= self.threshold),
            )
            for match, score in zip(matches, scores)
        ]


def default_lr_matcher() -> SemanticMatcher:
    return SemanticMatcher(LR_MODEL_PATH, LR_THRESHOLD, LR_MODEL_VERSION)


def default_lightgbm_matcher() -> SemanticMatcher:
    return SemanticMatcher(LIGHTGBM_MODEL_PATH, LIGHTGBM_THRESHOLD, LIGHTGBM_MODEL_VERSION)
