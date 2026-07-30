"""Hierarchical, explainable chemical reaction scoring."""

from chemical_score.api import (
    evaluate_reaction,
    evaluate_reactions,
    get_default_evaluator,
    list_metrics,
    set_default_evaluator,
    split_reaction_smiles,
)
from chemical_score.context import ReactionInputError
from chemical_score.engine import ReactionEvaluator
from chemical_score.evidence import EvidenceIndex, EvidenceRecord

__version__ = "0.3.0"

__all__ = [
    "EvidenceIndex",
    "EvidenceRecord",
    "ReactionEvaluator",
    "ReactionInputError",
    "evaluate_reaction",
    "evaluate_reactions",
    "get_default_evaluator",
    "list_metrics",
    "set_default_evaluator",
    "split_reaction_smiles",
]
