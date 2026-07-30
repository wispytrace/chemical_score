"""Structural and reactive-state safety indicators."""

from __future__ import annotations

from rdkit.Chem import Descriptors

from chemical_score.context import ReactionContext
from chemical_score.metrics.base import MetricSpec, clamp_score
from chemical_score.metrics.patterns import STRUCTURAL_ALERTS
from chemical_score.models import MetricOutcome


class StructuralAlerts:
    spec = MetricSpec(
        "structural_alerts",
        "结构风险警报",
        "匹配过氧键、叠氮、高能杂原子链等启发式结构警报。",
        ("safety", "structural_hazards"),
        1.5,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        matches: list[dict[str, object]] = []
        total_penalty = 0.0
        components = [
            *(
                ("reactant", index, molecule)
                for index, molecule in enumerate(context.reactant_mols)
            ),
            *(
                ("agent", index, molecule)
                for index, molecule in enumerate(context.agent_mols)
            ),
        ]
        for role, molecule_index, molecule in components:
            for alert in STRUCTURAL_ALERTS:
                count = len(molecule.GetSubstructMatches(alert.pattern, uniquify=True))
                if count:
                    penalty = alert.penalty * min(count, 2)
                    total_penalty += penalty
                    matches.append(
                        {
                            "id": alert.id,
                            "name": alert.name,
                            "component_role": role,
                            "component_index": molecule_index,
                            "matches": count,
                            "penalty": penalty,
                        }
                    )
        return MetricOutcome(
            score=clamp_score(100.0 - total_penalty),
            raw_value=len(matches),
            unit="alert_types",
            evidence={"alerts": matches},
            warnings=(
                ["结构警报只适合筛查，不能替代 SDS、实验规模和反应条件风险评估"]
                if matches
                else []
            ),
        )


class RadicalState:
    spec = MetricSpec(
        "radical_state",
        "自由基状态",
        "检查反应物侧是否出现无法由产物状态解释的额外自由基电子。",
        ("safety", "reactive_state"),
        1.2,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        reactant_radicals = sum(
            Descriptors.NumRadicalElectrons(molecule)
            for molecule in context.reactant_mols
        )
        product_radicals = Descriptors.NumRadicalElectrons(context.product_mol)
        excess = max(0, reactant_radicals - product_radicals)
        return MetricOutcome(
            score=clamp_score(100.0 - 40.0 * excess),
            raw_value=excess,
            unit="excess_radical_electrons",
            evidence={
                "reactant_radical_electrons": reactant_radicals,
                "product_radical_electrons": product_radicals,
            },
        )


class ChargeBalance:
    spec = MetricSpec(
        "charge_balance",
        "形式电荷平衡",
        "比较反应物与主产物总形式电荷；对可能缺少反离子的记录仅温和惩罚。",
        ("safety", "reactive_state"),
        0.8,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        reactant_charge = sum(
            atom.GetFormalCharge()
            for molecule in context.reactant_mols
            for atom in molecule.GetAtoms()
        )
        product_charge = sum(
            atom.GetFormalCharge() for atom in context.product_mol.GetAtoms()
        )
        difference = abs(reactant_charge - product_charge)
        score = 100.0 if difference == 0 else 80.0 if difference == 1 else 55.0
        return MetricOutcome(
            score=score,
            raw_value=difference,
            unit="formal_charge_delta",
            evidence={
                "reactant_formal_charge": reactant_charge,
                "product_formal_charge": product_charge,
            },
            warnings=(["电荷差也可能来自反离子或副产物未记录"] if difference else []),
        )


DEFAULT_SAFETY_METRICS = (StructuralAlerts(), RadicalState(), ChargeBalance())
