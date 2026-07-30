"""Rule-based feasibility metrics.

These are transparent heuristics, not calibrated probabilities of experimental
success. Every metric emits evidence so its score can be audited.
"""

from __future__ import annotations

from collections import Counter
from typing import ClassVar

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from chemical_score.context import ReactionContext
from chemical_score.metrics.base import MetricSpec, clamp_score
from chemical_score.metrics.patterns import FUNCTIONAL_GROUPS, LEAVING_GROUPS
from chemical_score.models import MetricOutcome


def _count_matches(molecule: Chem.Mol, pattern: Chem.Mol) -> int:
    return len(molecule.GetSubstructMatches(pattern, uniquify=True))


def _reactant_match_count(context: ReactionContext, key: str) -> int:
    pattern = FUNCTIONAL_GROUPS[key]
    return sum(_count_matches(molecule, pattern) for molecule in context.reactant_mols)


class IdentityCheck:
    spec = MetricSpec(
        "identity_check",
        "非恒等反应",
        "检查产物是否被原样作为反应物返回。",
        ("feasibility", "consistency"),
        2.0,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        matches = [
            smiles
            for smiles in context.canonical_reactants
            if smiles == context.canonical_main_product
        ]
        return MetricOutcome(
            score=0.0 if matches else 100.0,
            raw_value=len(matches),
            unit="matching_components",
            evidence={"identical_reactants": matches},
            warnings=["产物与至少一个反应物完全相同"] if matches else [],
        )


class MeaningfulChange:
    spec = MetricSpec(
        "meaningful_change",
        "实质转化程度",
        "识别高相似主前体加无意义微小碎片的伪反应。",
        ("feasibility", "consistency"),
        1.1,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        similarity = context.main_similarity
        other_heavy_atoms = sum(
            molecule.GetNumHeavyAtoms()
            for molecule in context.reactant_mols
            if molecule is not context.main_reactant
        )
        if similarity > 0.98 and other_heavy_atoms <= 2:
            score = 20.0
        elif similarity > 0.95 and other_heavy_atoms <= 2:
            score = 45.0
        elif similarity > 0.90 and other_heavy_atoms == 0:
            score = 60.0
        else:
            score = 100.0
        return MetricOutcome(
            score=score,
            raw_value=similarity,
            unit="tanimoto",
            evidence={
                "main_reactant_similarity": round(similarity, 4),
                "other_reactant_heavy_atoms": other_heavy_atoms,
            },
        )


class FragmentationAndSize:
    spec = MetricSpec(
        "fragmentation_and_size",
        "组分与尺寸合理性",
        "惩罚过多反应物组分和极端的总重原子膨胀。",
        ("feasibility", "consistency"),
        0.8,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        reactant_heavy = sum(mol.GetNumHeavyAtoms() for mol in context.reactant_mols)
        product_heavy = context.product_mol.GetNumHeavyAtoms()
        ratio = reactant_heavy / max(product_heavy, 1)
        penalty = max(0, len(context.reactant_mols) - 4) * 15.0
        if ratio > 3.0:
            penalty += min(50.0, (ratio - 3.0) * 20.0)
        return MetricOutcome(
            score=clamp_score(100.0 - penalty),
            raw_value=ratio,
            unit="reactant_to_product_heavy_atom_ratio",
            evidence={
                "reactant_components": len(context.reactant_mols),
                "reactant_heavy_atoms": reactant_heavy,
                "product_heavy_atoms": product_heavy,
            },
        )


class CoreElementConservation:
    spec = MetricSpec(
        "core_element_conservation",
        "核心元素来源",
        "检查产物中的 C/N/O 是否能由反应物提供；不因缺失副产物而惩罚多余元素。",
        ("feasibility", "conservation"),
        2.0,
    )

    _penalty: ClassVar[dict[str, float]] = {"C": 35.0, "N": 20.0, "O": 16.0}

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        missing = {
            element: max(
                0,
                context.product_atom_counts[element]
                - context.reactant_atom_counts[element],
            )
            for element in self._penalty
        }
        penalty = sum(
            self._penalty[element] * count for element, count in missing.items()
        )
        missing = {element: count for element, count in missing.items() if count}
        return MetricOutcome(
            score=clamp_score(100.0 - penalty),
            raw_value=sum(missing.values()),
            unit="missing_atoms",
            evidence={"missing_product_atoms": missing},
            warnings=["产物存在无法由反应物提供的核心元素"] if missing else [],
        )


class KeyElementConservation:
    spec = MetricSpec(
        "key_element_conservation",
        "关键杂元素来源",
        "独立检查卤素、P/S/B/Si 等关键元素，避免与 C/N/O 重复计分。",
        ("feasibility", "conservation"),
        1.4,
    )

    _elements = ("F", "Cl", "Br", "I", "P", "S", "B", "Si")
    _permitted_extras: ClassVar[frozenset[str]] = frozenset({"Cl", "Br", "I", "S"})

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        missing: dict[str, int] = {}
        suspicious_extra: dict[str, int] = {}
        for element in self._elements:
            difference = (
                context.reactant_atom_counts[element]
                - context.product_atom_counts[element]
            )
            if difference < 0:
                missing[element] = abs(difference)
            elif difference > 0 and element not in self._permitted_extras:
                suspicious_extra[element] = difference
        penalty = 30.0 * sum(missing.values()) + 6.0 * sum(suspicious_extra.values())
        return MetricOutcome(
            score=clamp_score(100.0 - penalty),
            raw_value=sum(missing.values()),
            unit="missing_atoms",
            evidence={
                "missing_product_atoms": missing,
                "suspicious_extra_reactant_atoms": suspicious_extra,
            },
            warnings=["产物存在无法由反应物提供的关键杂元素"] if missing else [],
        )


class StructuralContinuity:
    spec = MetricSpec(
        "structural_continuity",
        "结构连续性",
        "用 Morgan 指纹衡量主反应物和产物的结构连续性。",
        ("feasibility", "structure"),
        1.4,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        similarity = context.main_similarity
        if similarity < 0.05:
            score = 10.0
        elif similarity < 0.15:
            score = 30.0 + (similarity - 0.05) * 200.0
        elif similarity <= 0.90:
            score = 50.0 + (similarity - 0.15) / 0.75 * 50.0
        else:
            score = 95.0
        return MetricOutcome(
            score=clamp_score(score),
            raw_value=similarity,
            unit="tanimoto",
            evidence={"main_reactant_similarity": round(similarity, 4)},
        )


class ScaffoldContinuity:
    spec = MetricSpec(
        "scaffold_continuity",
        "骨架连续性",
        "比较主反应物和产物的 Bemis–Murcko 骨架。",
        ("feasibility", "structure"),
        1.0,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        product = context.scaffold(context.product_mol)
        reactant = context.scaffold(context.main_reactant)
        if not product.GetNumHeavyAtoms() or not reactant.GetNumHeavyAtoms():
            return MetricOutcome.not_applicable("至少一侧没有可比较的环状骨架")
        product_smiles = Chem.MolToSmiles(product, canonical=True)
        reactant_smiles = Chem.MolToSmiles(reactant, canonical=True)
        if product_smiles == reactant_smiles:
            score, relation = 100.0, "identical"
        elif product.HasSubstructMatch(reactant) or reactant.HasSubstructMatch(product):
            score, relation = 85.0, "substructure"
        else:
            score, relation = 35.0, "unrelated"
        return MetricOutcome(
            score=score,
            raw_value=relation,
            evidence={
                "reactant_scaffold": reactant_smiles,
                "product_scaffold": product_smiles,
            },
        )


class FunctionalGroupPlausibility:
    spec = MetricSpec(
        "functional_group_plausibility",
        "官能团转化支持",
        "检查新增酯、酰胺、醚和芳基偶联结构是否存在常见前体支持。",
        ("feasibility", "structure"),
        1.2,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        checks: list[dict[str, object]] = []
        rules = (
            (
                "ester_formation",
                "ester",
                lambda: (
                    _reactant_match_count(context, "alcohol") > 0
                    and (
                        _reactant_match_count(context, "carboxylic_acid") > 0
                        or _reactant_match_count(context, "acyl_halide") > 0
                    )
                ),
            ),
            (
                "amide_formation",
                "amide",
                lambda: (
                    _reactant_match_count(context, "amine") > 0
                    and (
                        _reactant_match_count(context, "carboxylic_acid") > 0
                        or _reactant_match_count(context, "acyl_halide") > 0
                    )
                ),
            ),
            (
                "ether_formation",
                "ether",
                lambda: (
                    _reactant_match_count(context, "alcohol") > 0
                    and (
                        _reactant_match_count(context, "sulfonate") > 0
                        or _reactant_match_count(context, "aryl_halide") > 0
                    )
                ),
            ),
        )
        for rule_name, product_group, supported in rules:
            product_count = _count_matches(
                context.product_mol, FUNCTIONAL_GROUPS[product_group]
            )
            reactant_count = _reactant_match_count(context, product_group)
            if product_count > reactant_count:
                checks.append(
                    {
                        "rule": rule_name,
                        "supported": bool(supported()),
                        "new_groups": product_count - reactant_count,
                    }
                )
        if (
            rdMolDescriptors.CalcNumAromaticRings(context.product_mol) > 0
            and _reactant_match_count(context, "aryl_halide") > 0
            and _reactant_match_count(context, "boronic_acid_or_ester") > 0
        ):
            checks.append(
                {
                    "rule": "aryl_boron_coupling",
                    "supported": True,
                    "new_groups": None,
                }
            )
        if not checks:
            return MetricOutcome.not_applicable("未检测到当前规则库覆盖的新增官能团")
        score = sum(95.0 if item["supported"] else 35.0 for item in checks) / len(
            checks
        )
        return MetricOutcome(
            score=score,
            raw_value=sum(bool(item["supported"]) for item in checks) / len(checks),
            unit="supported_rule_fraction",
            evidence={"checks": checks},
        )


class LeavingGroupSupport:
    spec = MetricSpec(
        "leaving_group_support",
        "离去基支持",
        "在新增酯/酰胺/醚场景下检查常见活化或离去基，证据不足时不作强判定。",
        ("feasibility", "structure"),
        0.5,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        newly_formed = any(
            _count_matches(context.product_mol, FUNCTIONAL_GROUPS[key])
            > _reactant_match_count(context, key)
            for key in ("ester", "amide", "ether")
        )
        if not newly_formed:
            return MetricOutcome.not_applicable("未检测到适合本指标判断的新成键场景")
        found: Counter[str] = Counter()
        for name, pattern in LEAVING_GROUPS.items():
            found[name] = sum(
                _count_matches(molecule, pattern) for molecule in context.reactant_mols
            )
        found = Counter({name: count for name, count in found.items() if count})
        good = sum(
            found[name]
            for name in ("bromide", "iodide", "tosylate", "mesylate", "acyl_halide")
        )
        if good:
            score = 90.0
        elif found["chloride"]:
            score = 70.0
        elif found["fluoride"]:
            score = 45.0
        else:
            score = 55.0
        return MetricOutcome(
            score=score,
            raw_value=good,
            unit="supporting_groups",
            evidence={"matched_leaving_groups": dict(found)},
            warnings=["离去基启发式不能替代具体反应条件判断"],
        )


class RingTopologyChange:
    spec = MetricSpec(
        "ring_topology_change",
        "环拓扑变化",
        "对单步中极端芳香环数量变化给出温和惩罚，允许正常成环/开环。",
        ("feasibility", "selectivity"),
        0.6,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        product_rings = rdMolDescriptors.CalcNumAromaticRings(context.product_mol)
        reactant_rings = sum(
            rdMolDescriptors.CalcNumAromaticRings(mol) for mol in context.reactant_mols
        )
        difference = abs(product_rings - reactant_rings)
        score = 100.0 if difference <= 1 else 70.0 if difference == 2 else 40.0
        return MetricOutcome(
            score=score,
            raw_value=difference,
            unit="aromatic_ring_count_delta",
            evidence={
                "reactant_aromatic_rings": reactant_rings,
                "product_aromatic_rings": product_rings,
            },
        )


class StereochemistryChange:
    spec = MetricSpec(
        "stereochemistry_change",
        "立体化学变化",
        "比较手性中心数量；没有原子映射时只作低权重预警。",
        ("feasibility", "selectivity"),
        0.5,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        product_count = len(
            Chem.FindMolChiralCenters(context.product_mol, includeUnassigned=True)
        )
        reactant_count = sum(
            len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            for mol in context.reactant_mols
        )
        if product_count == 0 and reactant_count == 0:
            return MetricOutcome.not_applicable("反应两侧均未检测到手性中心")
        difference = abs(product_count - reactant_count)
        score = 100.0 if difference <= 1 else 75.0 if difference == 2 else 50.0
        return MetricOutcome(
            score=score,
            raw_value=difference,
            unit="chiral_center_count_delta",
            evidence={
                "reactant_chiral_centers": reactant_count,
                "product_chiral_centers": product_count,
            },
            warnings=["未使用原子映射，无法判断具体中心的保持、翻转或消旋"],
        )


class DescriptorChange:
    spec = MetricSpec(
        "descriptor_change",
        "整体描述符变化",
        "比较主反应物和产物的环、杂原子、柔性及氢键描述符。",
        ("feasibility", "selectivity"),
        0.7,
    )

    _penalties: ClassVar[dict[str, float]] = {
        "rings": 10.0,
        "aromatic_rings": 10.0,
        "heteroatoms": 5.0,
        "rotatable_bonds": 3.0,
        "hba": 4.0,
        "hbd": 4.0,
    }

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        product = context.descriptors(context.product_mol)
        reactant = context.descriptors(context.main_reactant)
        delta = {key: abs(product[key] - reactant[key]) for key in self._penalties}
        penalty = sum(delta[key] * weight for key, weight in self._penalties.items())
        return MetricOutcome(
            score=clamp_score(100.0 - penalty),
            raw_value=sum(delta.values()),
            unit="weighted_descriptor_delta",
            evidence={"absolute_deltas": delta},
        )


class ChemoselectivityRisk:
    spec = MetricSpec(
        "chemoselectivity_risk",
        "化学选择性风险",
        "识别未保护游离胺存在时却优先形成酯的特定竞争位点风险。",
        ("feasibility", "selectivity"),
        1.0,
    )

    def evaluate(self, context: ReactionContext) -> MetricOutcome:
        ester = FUNCTIONAL_GROUPS["ester"]
        amine = FUNCTIONAL_GROUPS["amine"]
        new_ester = _count_matches(context.product_mol, ester) > sum(
            _count_matches(molecule, ester) for molecule in context.reactant_mols
        )
        amine_retained = context.product_mol.HasSubstructMatch(amine) and any(
            molecule.HasSubstructMatch(amine) for molecule in context.reactant_mols
        )
        if not new_ester or not amine_retained:
            return MetricOutcome.not_applicable("未出现该启发式覆盖的胺/酯竞争场景")
        return MetricOutcome(
            score=25.0,
            raw_value="unprotected_amine_competes_with_esterification",
            evidence={"new_ester": True, "free_amine_retained": True},
            warnings=["可能需要保护胺或证明条件具有足够选择性"],
        )


DEFAULT_FEASIBILITY_METRICS = (
    IdentityCheck(),
    MeaningfulChange(),
    FragmentationAndSize(),
    CoreElementConservation(),
    KeyElementConservation(),
    StructuralContinuity(),
    ScaffoldContinuity(),
    FunctionalGroupPlausibility(),
    LeavingGroupSupport(),
    RingTopologyChange(),
    StereochemistryChange(),
    DescriptorChange(),
    ChemoselectivityRisk(),
)
