"""Stable Python API for reaction scoring."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from chemical_score.context import ReactionInputError
from chemical_score.engine import ReactionEvaluator
from chemical_score.evidence import EvidenceIndex

_DEFAULT_EVALUATOR: ReactionEvaluator | None = None


def get_default_evaluator() -> ReactionEvaluator:
    global _DEFAULT_EVALUATOR
    if _DEFAULT_EVALUATOR is None:
        evidence_path = os.getenv("CHEMICAL_SCORE_EVIDENCE_PATH")
        evidence_index = (
            EvidenceIndex.from_file(evidence_path, strict=False)
            if evidence_path
            else None
        )
        _DEFAULT_EVALUATOR = ReactionEvaluator(evidence_index=evidence_index)
    return _DEFAULT_EVALUATOR


def set_default_evaluator(evaluator: ReactionEvaluator | None) -> None:
    """Inject an application-wide evaluator; pass ``None`` to reset lazy setup."""

    global _DEFAULT_EVALUATOR
    _DEFAULT_EVALUATOR = evaluator


def split_reaction_smiles(reaction_smiles: str) -> tuple[str, str, str | None]:
    """Parse ``reactants>agents>products`` or ``reactants>>products``."""

    parts = reaction_smiles.split(">")
    if len(parts) != 3:
        raise ReactionInputError(
            "reaction_smiles must use 'reactants>agents>products' format"
        )
    reactants, agents, products = (part.strip() for part in parts)
    if not reactants or not products:
        raise ReactionInputError("reaction_smiles requires reactants and products")
    return reactants, products, agents or None


def evaluate_reaction(
    *,
    reaction_smiles: str | None = None,
    reactants_smiles: str | None = None,
    product_smiles: str | None = None,
    agents_smiles: str | None = None,
    evaluator: ReactionEvaluator | None = None,
) -> dict[str, object]:
    """Evaluate a reaction using either a reaction SMILES or separate fields."""

    if reaction_smiles:
        if reactants_smiles or product_smiles or agents_smiles:
            raise ReactionInputError(
                "reaction_smiles cannot be combined with separate SMILES fields"
            )
        reactants_smiles, product_smiles, agents_smiles = split_reaction_smiles(
            reaction_smiles
        )
    if not reactants_smiles or not product_smiles:
        raise ReactionInputError(
            "provide reaction_smiles or both reactants_smiles and product_smiles"
        )
    engine = evaluator or get_default_evaluator()
    return engine.evaluate(reactants_smiles, product_smiles, agents_smiles)


def evaluate_reactions(
    reactions: Iterable[dict[str, Any]],
    *,
    concurrency: int = 1,
    evaluator: ReactionEvaluator | None = None,
) -> list[dict[str, object]]:
    engine = evaluator or get_default_evaluator()
    parsed: list[tuple[str, str, str | None]] = []
    for reaction in reactions:
        if reaction.get("reaction_smiles"):
            if any(
                reaction.get(field)
                for field in ("reactants_smiles", "product_smiles", "agents_smiles")
            ):
                raise ReactionInputError(
                    "reaction_smiles cannot be combined with separate SMILES fields"
                )
            parsed.append(split_reaction_smiles(str(reaction["reaction_smiles"])))
        else:
            reactants = reaction.get("reactants_smiles")
            product = reaction.get("product_smiles")
            if not reactants or not product:
                raise ReactionInputError(
                    "each reaction requires reaction_smiles or separate reactants/product"
                )
            parsed.append((str(reactants), str(product), reaction.get("agents_smiles")))
    return engine.evaluate_many(parsed, concurrency=concurrency)


def list_metrics(evaluator: ReactionEvaluator | None = None) -> dict[str, object]:
    return (evaluator or get_default_evaluator()).describe()
