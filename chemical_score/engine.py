"""Hierarchical reaction scoring engine."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

from chemical_score.context import ReactionContext, ReactionInputError
from chemical_score.evidence import EvidenceIndex
from chemical_score.metrics import MetricRegistry, build_default_registry
from chemical_score.models import (
    EvaluationResult,
    MetricOutcome,
    MetricStatus,
    ScoreNode,
)


@dataclass(frozen=True, slots=True)
class NodeConfig:
    id: str
    name: str
    weight: float
    description: str


DIMENSIONS = (
    NodeConfig("feasibility", "可行性", 0.48, "结构守恒、转化合理性与选择性"),
    NodeConfig(
        "evidence_support",
        "证据支持度",
        0.20,
        "历史反应相似性、转化先例和实验结果支持",
    ),
    NodeConfig("safety", "安全性", 0.16, "结构风险与反应性状态筛查"),
    NodeConfig("economy", "经济性", 0.16, "物料效率与合成策略成本"),
)

GROUPS = {
    "feasibility": (
        NodeConfig("conservation", "组成守恒", 0.30, "产物元素是否有明确来源"),
        NodeConfig("consistency", "反应一致性", 0.20, "排除恒等、伪转化和异常碎片"),
        NodeConfig("structure", "结构与转化", 0.30, "骨架、官能团及前体支持"),
        NodeConfig("selectivity", "选择性与变化幅度", 0.20, "环、手性和描述符变化"),
    ),
    "safety": (
        NodeConfig("structural_hazards", "结构风险", 0.60, "高能或敏感结构警报"),
        NodeConfig("reactive_state", "反应性状态", 0.40, "自由基和形式电荷状态"),
    ),
    "evidence_support": (
        NodeConfig("similarity", "反应空间相似性", 0.50, "近邻相似度与局部先例密度"),
        NodeConfig("precedent", "反应先例", 0.30, "精确反应与映射键变换先例"),
        NodeConfig("outcomes", "历史结果", 0.20, "相似先例的收率或成功标签支持"),
    ),
    "economy": (
        NodeConfig("material_efficiency", "物料效率", 0.60, "原子与碳保留效率"),
        NodeConfig("synthesis_strategy", "合成策略", 0.40, "前体复杂度和保护基负担"),
    ),
}


class ReactionEvaluator:
    """Evaluate one reaction and return a fully decomposable score tree."""

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        *,
        evidence_index: EvidenceIndex | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.evidence_index = evidence_index

    def evaluate(
        self,
        reactants_smiles: str,
        product_smiles: str,
        agents_smiles: str | None = None,
    ) -> dict[str, object]:
        return self.evaluate_result(
            reactants_smiles, product_smiles, agents_smiles
        ).to_dict()

    def evaluate_result(
        self,
        reactants_smiles: str,
        product_smiles: str,
        agents_smiles: str | None = None,
    ) -> EvaluationResult:
        started = perf_counter()
        try:
            context = ReactionContext(
                reactants_smiles=reactants_smiles,
                product_smiles=product_smiles,
                agents_smiles=agents_smiles,
                resources={"evidence_index": self.evidence_index},
            )
        except ReactionInputError as exc:
            return EvaluationResult(
                status="invalid_input",
                score=None,
                score_tree=None,
                reaction={
                    "reactants_smiles": reactants_smiles,
                    "agents_smiles": agents_smiles,
                    "product_smiles": product_smiles,
                },
                coverage=0.0,
                errors=[str(exc)],
                duration_ms=(perf_counter() - started) * 1000.0,
            )

        leaves: dict[tuple[str, str], list[ScoreNode]] = {}
        metric_errors: list[str] = []
        for metric in self.registry.metrics:
            metric_started = perf_counter()
            try:
                outcome = metric.evaluate(context)
                if outcome.score is not None:
                    outcome.score = max(0.0, min(100.0, float(outcome.score)))
            except Exception as exc:  # noqa: BLE001 - extension boundary isolation
                outcome = MetricOutcome(
                    score=None,
                    status=MetricStatus.ERROR,
                    warnings=[f"{type(exc).__name__}: {exc}"],
                )
                metric_errors.append(metric.spec.id)
            leaf = ScoreNode(
                id=metric.spec.id,
                name=metric.spec.name,
                node_type="metric",
                score=outcome.score,
                weight=metric.spec.weight,
                description=metric.spec.description,
                status=outcome.status,
                raw_value=outcome.raw_value,
                unit=outcome.unit,
                evidence=outcome.evidence,
                warnings=outcome.warnings,
                duration_ms=(perf_counter() - metric_started) * 1000.0,
            )
            leaves.setdefault(metric.spec.path, []).append(leaf)

        dimension_nodes: list[ScoreNode] = []
        for dimension in DIMENSIONS:
            group_nodes: list[ScoreNode] = []
            for group in GROUPS[dimension.id]:
                group_nodes.append(
                    self._aggregate(
                        group.id,
                        group.name,
                        "group",
                        group.weight,
                        group.description,
                        leaves.get((dimension.id, group.id), []),
                    )
                )
            dimension_nodes.append(
                self._aggregate(
                    dimension.id,
                    dimension.name,
                    "dimension",
                    dimension.weight,
                    dimension.description,
                    group_nodes,
                )
            )

        root = self._aggregate(
            "overall",
            "综合评分",
            "total",
            1.0,
            "所有可用维度的加权总分",
            dimension_nodes,
        )
        flags, cap = self._critical_flags(root)
        if root.score is not None and cap is not None and root.score > cap:
            unconstrained = root.score
            root.score = cap
            root.evidence.update(
                {
                    "unconstrained_score": round(unconstrained, 4),
                    "critical_cap": cap,
                }
            )

        total_metrics = len(self.registry.metrics)
        coverage = (
            (total_metrics - len(metric_errors)) / total_metrics
            if total_metrics
            else 0.0
        )
        reaction = context.reaction_dict()
        warnings = list(reaction.pop("warnings", []))
        if metric_errors:
            warnings.append(
                "部分指标执行失败，已从上级加权中排除: " + ", ".join(metric_errors)
            )
        warnings.append(
            "本结果是可解释的规则启发式评分，不是实验成功率、产率或安全结论。"
        )
        return EvaluationResult(
            status="success" if not metric_errors else "partial_success",
            score=root.score,
            score_tree=root,
            reaction=reaction,
            coverage=coverage,
            flags=flags,
            warnings=warnings,
            duration_ms=(perf_counter() - started) * 1000.0,
        )

    def evaluate_many(
        self,
        reactions: Iterable[tuple[str, str, str | None]],
        *,
        concurrency: int = 1,
    ) -> list[dict[str, object]]:
        items = list(reactions)
        if concurrency <= 1 or len(items) <= 1:
            return [self.evaluate(*item) for item in items]
        workers = max(1, min(int(concurrency), 32, len(items)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda item: self.evaluate(*item), items))

    def describe(self) -> dict[str, object]:
        dimensions: list[dict[str, object]] = []
        metrics_by_path: dict[tuple[str, str], list[dict[str, object]]] = {}
        for metric in self.registry.describe():
            path = tuple(metric["path"])
            metrics_by_path.setdefault(path, []).append(metric)
        for dimension in DIMENSIONS:
            groups: list[dict[str, object]] = []
            for group in GROUPS[dimension.id]:
                groups.append(
                    {
                        "id": group.id,
                        "name": group.name,
                        "weight": group.weight,
                        "description": group.description,
                        "metrics": metrics_by_path.get((dimension.id, group.id), []),
                    }
                )
            dimensions.append(
                {
                    "id": dimension.id,
                    "name": dimension.name,
                    "weight": dimension.weight,
                    "description": dimension.description,
                    "groups": groups,
                }
            )
        return {"id": "overall", "name": "综合评分", "dimensions": dimensions}

    def evidence_status(self) -> dict[str, object]:
        if self.evidence_index is None:
            return {
                "configured": False,
                "record_count": 0,
                "message": "未配置历史反应证据库",
            }
        return self.evidence_index.status()

    @staticmethod
    def _aggregate(
        id_: str,
        name: str,
        node_type: str,
        weight: float,
        description: str,
        children: list[ScoreNode],
    ) -> ScoreNode:
        available = [
            child
            for child in children
            if child.score is not None and child.status == MetricStatus.EVALUATED
        ]
        total_weight = sum(child.weight for child in available)
        if not available or total_weight <= 0:
            status = (
                MetricStatus.ERROR
                if any(child.status == MetricStatus.ERROR for child in children)
                else MetricStatus.NOT_APPLICABLE
            )
            return ScoreNode(
                id=id_,
                name=name,
                node_type=node_type,
                score=None,
                weight=weight,
                description=description,
                status=status,
                children=children,
            )
        score = 0.0
        for child in available:
            child.effective_weight = child.weight / total_weight
            child.contribution = child.score * child.effective_weight
            score += child.contribution
        return ScoreNode(
            id=id_,
            name=name,
            node_type=node_type,
            score=score,
            weight=weight,
            description=description,
            children=children,
        )

    @staticmethod
    def _critical_flags(
        root: ScoreNode,
    ) -> tuple[list[dict[str, object]], float | None]:
        leaves: dict[str, ScoreNode] = {}

        def visit(node: ScoreNode) -> None:
            if node.node_type == "metric":
                leaves[node.id] = node
            for child in node.children:
                visit(child)

        visit(root)
        policies = (
            (
                "identity_check",
                50.0,
                20.0,
                "identity_reaction",
                "产物与反应物完全相同，总分被限制。",
            ),
            (
                "core_element_conservation",
                40.0,
                35.0,
                "missing_core_elements",
                "产物缺少可追溯的 C/N/O 来源，总分被限制。",
            ),
            (
                "key_element_conservation",
                40.0,
                35.0,
                "missing_key_elements",
                "产物缺少可追溯的关键杂元素来源，总分被限制。",
            ),
        )
        flags: list[dict[str, object]] = []
        caps: list[float] = []
        for metric_id, threshold, cap, code, message in policies:
            node = leaves.get(metric_id)
            if node and node.score is not None and node.score < threshold:
                flags.append(
                    {
                        "code": code,
                        "severity": "critical",
                        "metric_id": metric_id,
                        "message": message,
                        "score_cap": cap,
                    }
                )
                caps.append(cap)
        return flags, min(caps) if caps else None
