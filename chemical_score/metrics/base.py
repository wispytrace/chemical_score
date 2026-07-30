"""Metric interfaces and registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from chemical_score.context import ReactionContext
from chemical_score.models import MetricOutcome


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


@dataclass(frozen=True, slots=True)
class MetricSpec:
    id: str
    name: str
    description: str
    path: tuple[str, str]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.path) != 2:
            raise ValueError("metric path must contain dimension and group")
        if self.weight <= 0:
            raise ValueError("metric weight must be positive")


class Metric(Protocol):
    spec: MetricSpec

    def evaluate(self, context: ReactionContext) -> MetricOutcome: ...


class MetricRegistry:
    def __init__(self, metrics: Iterable[Metric] = ()) -> None:
        self._metrics: list[Metric] = []
        self._ids: set[str] = set()
        for metric in metrics:
            self.register(metric)

    def register(self, metric: Metric) -> None:
        if metric.spec.id in self._ids:
            raise ValueError(f"duplicate metric id: {metric.spec.id}")
        self._ids.add(metric.spec.id)
        self._metrics.append(metric)

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return tuple(self._metrics)

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "id": metric.spec.id,
                "name": metric.spec.name,
                "description": metric.spec.description,
                "path": list(metric.spec.path),
                "weight": metric.spec.weight,
            }
            for metric in self._metrics
        ]
