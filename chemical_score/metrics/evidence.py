"""Corpus-dependent evidence-support metrics."""

from __future__ import annotations

import math

from chemical_score.context import ReactionContext
from chemical_score.evidence import EvidenceComparison, EvidenceIndex
from chemical_score.metrics.base import MetricSpec, clamp_score
from chemical_score.models import MetricOutcome


def _comparison(context: ReactionContext) -> EvidenceComparison | None:
    index = context.get_resource("evidence_index")
    if not isinstance(index, EvidenceIndex) or len(index) == 0:
        return None
    return context.memoize(
        f"evidence_comparison:{id(index)}", lambda: index.compare(context)
    )


def _unavailable() -> MetricOutcome:
    return MetricOutcome.not_applicable(
        "未配置非空历史反应证据库；该指标不参与上级加权"
    )


class NearestReactionSimilarity:
    spec = MetricSpec(
        "nearest_reaction_similarity",
        "最近反应相似度",
        "以 RDKit 反应差分指纹计算查询反应与最近历史反应的 Tanimoto 相似度。",
        ("evidence_support", "similarity"),
        1.2,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        comparison = _comparison(context)
        if comparison is None:
            return _unavailable()
        return MetricOutcome(
            score=100.0 * comparison.top_similarity,
            raw_value=comparison.top_similarity,
            unit="tanimoto",
            evidence={
                "index_size": comparison.index_size,
                "neighbors": [item.to_dict() for item in comparison.neighbors],
                "fingerprint": "RDKit difference fingerprint",
            },
            warnings=["相似度衡量历史反应空间邻近性，不等同于可行性或成功概率"],
        )


class LocalPrecedentDensity:
    spec = MetricSpec(
        "local_precedent_density",
        "局部先例密度",
        "统计相似度阈值以上的历史反应数量，并按最多 5 个先例归一化。",
        ("evidence_support", "similarity"),
        0.8,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        comparison = _comparison(context)
        if comparison is None:
            return _unavailable()
        target = max(1, min(5, comparison.index_size))
        score = 100.0 * min(1.0, comparison.neighbor_count / target)
        return MetricOutcome(
            score=score,
            raw_value=comparison.neighbor_count,
            unit="precedents",
            evidence={
                "index_size": comparison.index_size,
                "similarity_threshold": comparison.similarity_threshold,
                "normalization_target": target,
            },
        )


class ExactReactionPrecedent:
    spec = MetricSpec(
        "exact_reaction_precedent",
        "精确反应先例",
        "统计忽略组分顺序和原子映射编号后完全相同的历史反应。",
        ("evidence_support", "precedent"),
        0.8,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        comparison = _comparison(context)
        if comparison is None:
            return _unavailable()
        return MetricOutcome(
            score=100.0 * (1.0 - math.exp(-comparison.exact_count)),
            raw_value=comparison.exact_count,
            unit="exact_precedents",
            evidence={"index_size": comparison.index_size},
        )


class MappedTransformationPrecedent:
    spec = MetricSpec(
        "mapped_transformation_precedent",
        "映射键变换先例",
        "比较原子映射反应中的成键、断键和键级变化；签名与映射编号无关。",
        ("evidence_support", "precedent"),
        1.2,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        comparison = _comparison(context)
        if comparison is None:
            return _unavailable()
        if comparison.transformation_signature is None:
            return MetricOutcome.not_applicable(
                "查询反应缺少可用原子映射，无法可靠比较键变换先例",
                evidence={"index_size": comparison.index_size},
            )
        count = comparison.transformation_count or 0
        return MetricOutcome(
            score=clamp_score(100.0 * (1.0 - math.exp(-count / 3.0))),
            raw_value=count,
            unit="transformation_precedents",
            evidence={
                "transformation_signature": comparison.transformation_signature,
                "index_size": comparison.index_size,
            },
        )


class HistoricalOutcomeSupport:
    spec = MetricSpec(
        "historical_outcome_support",
        "历史结果支持",
        "对相似度阈值以上先例的实验收率或成功标签做相似度加权。",
        ("evidence_support", "outcomes"),
        1.0,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        comparison = _comparison(context)
        if comparison is None:
            return _unavailable()
        if comparison.outcome_score is None:
            return MetricOutcome.not_applicable(
                "相似历史反应没有可用的收率或成功标签",
                evidence={
                    "neighbor_count": comparison.neighbor_count,
                    "similarity_threshold": comparison.similarity_threshold,
                },
            )
        warnings = ["成功布尔标签按 100/0 换算；该分数未经数据集外概率校准"]
        if comparison.outcome_basis.get("yield", 0) and comparison.outcome_basis.get(
            "success", 0
        ):
            warnings.append("本次聚合混合了收率和成功布尔标签，建议参考库统一标签口径")
        return MetricOutcome(
            score=comparison.outcome_score,
            raw_value=comparison.outcome_score,
            unit="percent_support",
            evidence={
                "outcome_count": comparison.outcome_count,
                "outcome_basis": comparison.outcome_basis,
                "similarity_threshold": comparison.similarity_threshold,
            },
            warnings=warnings,
        )


DEFAULT_EVIDENCE_METRICS = (
    NearestReactionSimilarity(),
    LocalPrecedentDensity(),
    ExactReactionPrecedent(),
    MappedTransformationPrecedent(),
    HistoricalOutcomeSupport(),
)
