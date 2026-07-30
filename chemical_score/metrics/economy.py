"""Material-efficiency and synthesis-strategy metrics."""

from __future__ import annotations

import os
import sys

from rdkit.Chem import Descriptors, RDConfig

from chemical_score.context import ReactionContext
from chemical_score.metrics.base import MetricSpec, clamp_score
from chemical_score.metrics.patterns import PROTECTING_GROUPS
from chemical_score.models import MetricOutcome


def _load_sa_scorer():
    contrib_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if contrib_path not in sys.path:
        sys.path.append(contrib_path)
    try:
        import sascorer  # type: ignore

        return sascorer
    except ImportError:
        return None


SA_SCORER = _load_sa_scorer()


class AtomEconomyEstimate:
    spec = MetricSpec(
        "atom_economy_estimate",
        "原子经济性估计",
        "以主产物分子量/反应物总分子量估算；未计量化学计量数和未记录物料。",
        ("economy", "material_efficiency"),
        1.2,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        reactant_mass = sum(Descriptors.MolWt(mol) for mol in context.reactant_mols)
        product_mass = Descriptors.MolWt(context.product_mol)
        if reactant_mass <= 0:
            return MetricOutcome.not_applicable("反应物分子量无法计算")
        ratio = product_mass / reactant_mass
        warnings = ["该值是由 SMILES 推导的估计，不包含当量、溶剂、收率和后处理物料"]
        if ratio > 1.0:
            warnings.append("估计值超过 100%，说明反应记录可能缺少参与成键的物料")
        return MetricOutcome(
            score=clamp_score(ratio * 100.0),
            raw_value=ratio,
            unit="fraction",
            evidence={
                "reactant_total_molecular_weight": round(reactant_mass, 4),
                "main_product_molecular_weight": round(product_mass, 4),
            },
            warnings=warnings,
        )


class CarbonEfficiency:
    spec = MetricSpec(
        "carbon_efficiency",
        "碳保留率",
        "计算主产物碳原子数与全部反应物碳原子数之比。",
        ("economy", "material_efficiency"),
        1.0,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        reactant_carbon = context.reactant_atom_counts["C"]
        product_carbon = context.product_atom_counts["C"]
        if reactant_carbon == 0:
            return MetricOutcome.not_applicable("反应物不含碳，无法计算碳保留率")
        ratio = product_carbon / reactant_carbon
        warnings: list[str] = []
        if ratio > 1.0:
            warnings.append("碳保留率超过 100%，反应记录可能缺少含碳反应物")
        return MetricOutcome(
            score=clamp_score(ratio * 100.0),
            raw_value=ratio,
            unit="fraction",
            evidence={
                "reactant_carbon_atoms": reactant_carbon,
                "product_carbon_atoms": product_carbon,
            },
            warnings=warnings,
        )


class SyntheticAccessibilityChange:
    spec = MetricSpec(
        "synthetic_accessibility_change",
        "合成可及性变化",
        "比较最复杂前体与主产物的 RDKit SA Score；逆合成前体更简单时得分更高。",
        ("economy", "synthesis_strategy"),
        1.0,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        if SA_SCORER is None:
            return MetricOutcome.not_applicable("当前 RDKit 安装不包含 SA_Score")
        product_sa = float(SA_SCORER.calculateScore(context.product_mol))
        reactant_scores = [
            float(SA_SCORER.calculateScore(molecule))
            for molecule in context.reactant_mols
        ]
        hardest_precursor = max(reactant_scores)
        improvement = product_sa - hardest_precursor
        return MetricOutcome(
            score=clamp_score(50.0 + improvement * 12.0),
            raw_value=improvement,
            unit="sa_score_improvement",
            evidence={
                "product_sa_score": round(product_sa, 4),
                "reactant_sa_scores": [round(value, 4) for value in reactant_scores],
                "hardest_precursor_sa_score": round(hardest_precursor, 4),
            },
        )


class ProtectingGroupBurden:
    spec = MetricSpec(
        "protecting_group_burden",
        "保护基负担",
        "识别常见保护基的引入或脱除；按步骤经济性计为成本而非奖励。",
        ("economy", "synthesis_strategy"),
        0.7,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        reactant_counts: dict[str, int] = {}
        product_counts: dict[str, int] = {}
        for name, pattern in PROTECTING_GROUPS.items():
            reactant_counts[name] = sum(
                len(mol.GetSubstructMatches(pattern, uniquify=True))
                for mol in context.reactant_mols
            )
            product_counts[name] = len(
                context.product_mol.GetSubstructMatches(pattern, uniquify=True)
            )
        introduced = {
            name: product_counts[name] - reactant_counts[name]
            for name in PROTECTING_GROUPS
            if product_counts[name] > reactant_counts[name]
        }
        removed = {
            name: reactant_counts[name] - product_counts[name]
            for name in PROTECTING_GROUPS
            if reactant_counts[name] > product_counts[name]
        }
        if introduced:
            score = 45.0
        elif removed:
            score = 65.0
        else:
            score = 100.0
        return MetricOutcome(
            score=score,
            raw_value=sum(introduced.values()) - sum(removed.values()),
            unit="net_protecting_groups_introduced",
            evidence={"introduced": introduced, "removed": removed},
        )


DEFAULT_ECONOMY_METRICS = (
    AtomEconomyEstimate(),
    CarbonEfficiency(),
    SyntheticAccessibilityChange(),
    ProtectingGroupBurden(),
)
