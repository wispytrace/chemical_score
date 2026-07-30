"""Compatibility exports for the deterministic reaction scoring engine."""

from chemical_score import ReactionEvaluator, evaluate_reaction, list_metrics
from chemical_score.metrics import build_default_registry

# Kept as a discoverable compatibility name. It is now a typed registry object,
# not a nested dictionary containing executable functions.
METRICS_REGISTRY = build_default_registry()

__all__ = [
    "METRICS_REGISTRY",
    "ReactionEvaluator",
    "evaluate_reaction",
    "list_metrics",
]
