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

# 2026-08-24：用 training-v4 快照（标注积压清空后，正样本消息从约1484条涨到
# 2162条）重新训练。冻结测试集沿用 training-v3 切分出的那754条消息id
# （见 data/frozen_test_message_ids.json），新旧模型在同一份测试集上比较，
# 且这次切分逻辑修了一个真实存在的近重复内容跨集合泄漏问题（同一条转发/
# 复制粘贴文案换个消息id出现在训练集又出现在测试集，见
# src/build_training_dataset.py 的 find_duplicate_clusters）——先发现泄漏、
# 修好去重逻辑、重新切分训练，之前一版带泄漏的指标已经作废。

# classic-lr-v4 训出来后做公开发布前的隐私检查，又发现了第二个问题：
# word_vectorizer 用 sklearn 默认 token_pattern，不切分中文，一整句没有空格的
# 中文会被当成一个"词"，词表里混进了接近完整句子的原文片段（第三方抓取内容，
# 公开发布模型文件会把这些片段带出去）。修了 WORD_TOKEN_PATTERN
# （train_classic_model.py，只匹配 ASCII 单词或单个中文字符，中文泛化信号
# 交给不受此影响的 char n-gram 承担）后重新训练成 classic-lr-v5，测试集指标
# 与 v4 基本持平甚至略好（P 0.938、R 0.968、F1 0.953），说明那些整句 token
# 本来就是过拟合的噪声特征，不是有用信号。v4 从未发布过，不需要单独记录。

# LR：当前线上基线 + 回退方案。v5 相对 v3（上一个正式发布过/线上跑过的版本）
# 是干净的提升（测试集 Precision 0.868->0.938，Recall 0.982->0.968，
# F1 0.921->0.953），但 Precision 仍未达到 98% 的长期目标，v0.1.0 按用户
# 2026-08-24 的决定不以 98% 为硬门槛发布。已过 serving smoke test + 延迟测试
# + 回滚验证，见 MODEL_CHANGELOG.md。旧版 v3 模型文件归档在
# models/company_classifier_v_prev_archive/，需要回滚可以直接复制回来。
LR_MODEL_PATH = MODEL_ROOT / "tfidf_structured__balanced" / "model.joblib"
LR_THRESHOLD = 0.6804017265908725
LR_MODEL_VERSION = "classic-lr-v5"

# LightGBM：影子模型。同一批数据训出的 lightgbm-shadow-v3 在冻结测试集上是权衡
# 而非提升（Precision 0.938->0.906 下降，Recall 0.975->1.000，F1 0.956->0.951
# 基本打平，且新旧两版 Precision 都没达到98%目标），按 2026-08-24 的决定
# 不转正，继续用 v2 跑影子观察；v3 的模型文件单独存在
# models/company_classifier/lgbm_svd_structured__balanced_v3_shadow_experimental/
# 里，不接管这里的路径。
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
