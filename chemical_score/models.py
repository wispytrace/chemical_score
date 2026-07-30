"""Serializable domain models used by the scoring engine.

The core package deliberately uses dataclasses instead of web-framework models so
that it can be embedded without FastAPI or Pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricStatus(str, Enum):
    EVALUATED = "evaluated"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


@dataclass(slots=True)
class MetricOutcome:
    """Unweighted output produced by one metric implementation."""

    score: float | None
    raw_value: float | int | str | None = None
    unit: str | None = None
    status: MetricStatus = MetricStatus.EVALUATED
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def not_applicable(
        cls, reason: str, *, evidence: dict[str, Any] | None = None
    ) -> MetricOutcome:
        return cls(
            score=None,
            status=MetricStatus.NOT_APPLICABLE,
            evidence=evidence or {},
            warnings=[reason],
        )


@dataclass(slots=True)
class ScoreNode:
    """One node in the total -> dimension -> group -> metric score tree."""

    id: str
    name: str
    node_type: str
    score: float | None
    weight: float
    effective_weight: float = 0.0
    contribution: float = 0.0
    description: str | None = None
    status: MetricStatus = MetricStatus.EVALUATED
    raw_value: float | int | str | None = None
    unit: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float | None = None
    children: list[ScoreNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "type": self.node_type,
            "score": _round(self.score),
            "weight": self.weight,
            "effective_weight": _round(self.effective_weight),
            "contribution": _round(self.contribution),
            "status": self.status.value,
        }
        if self.description:
            payload["description"] = self.description
        if self.raw_value is not None:
            payload["raw_value"] = _round(self.raw_value)
        if self.unit:
            payload["unit"] = self.unit
        if self.evidence:
            payload["evidence"] = self.evidence
        if self.warnings:
            payload["warnings"] = self.warnings
        if self.duration_ms is not None:
            payload["duration_ms"] = round(self.duration_ms, 3)
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload


@dataclass(slots=True)
class EvaluationResult:
    status: str
    score: float | None
    score_tree: ScoreNode | None
    reaction: dict[str, Any]
    coverage: float
    flags: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    schema_version: str = "1.0"
    engine_version: str = "0.3.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "status": self.status,
            "score": _round(self.score),
            "coverage": _round(self.coverage),
            "reaction": self.reaction,
            "flags": self.flags,
            "warnings": self.warnings,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 3),
            "score_tree": self.score_tree.to_dict() if self.score_tree else None,
        }


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value
