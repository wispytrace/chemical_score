"""Compatibility imports for the former combined scoring/LLM module."""

from chemical_score import ReactionEvaluator, evaluate_reaction, list_metrics
from chemical_score.metrics import build_default_registry
from chemical_score.reviewer import QwenReviewer

# Kept as a discoverable compatibility name. It is now a typed registry object,
# not a nested dictionary containing executable functions.
METRICS_REGISTRY = build_default_registry()

__all__ = [
    "METRICS_REGISTRY",
    "QwenReviewer",
    "ReactionEvaluator",
    "evaluate_reaction",
    "list_metrics",
]
