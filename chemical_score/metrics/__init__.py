"""Default metric registry."""

from chemical_score.metrics.base import MetricRegistry
from chemical_score.metrics.economy import DEFAULT_ECONOMY_METRICS
from chemical_score.metrics.evidence import DEFAULT_EVIDENCE_METRICS
from chemical_score.metrics.feasibility import DEFAULT_FEASIBILITY_METRICS
from chemical_score.metrics.safety import DEFAULT_SAFETY_METRICS


def build_default_registry() -> MetricRegistry:
    return MetricRegistry(
        (
            *DEFAULT_FEASIBILITY_METRICS,
            *DEFAULT_EVIDENCE_METRICS,
            *DEFAULT_SAFETY_METRICS,
            *DEFAULT_ECONOMY_METRICS,
        )
    )


__all__ = ["MetricRegistry", "build_default_registry"]
