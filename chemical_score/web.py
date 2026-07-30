"""FastAPI transport for the scoring engine.

Run with: ``uvicorn chemical_score.web:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from chemical_score import __version__
from chemical_score.api import (
    evaluate_reaction,
    get_default_evaluator,
    list_metrics,
    split_reaction_smiles,
)
from chemical_score.context import ReactionInputError


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reaction_smiles: str | None = Field(
        default=None, examples=["CC(=O)O.CCO>>CCOC(C)=O"]
    )
    reactants_smiles: str | None = Field(default=None, examples=["CC(=O)O.CCO"])
    product_smiles: str | None = Field(default=None, examples=["CCOC(C)=O"])
    agents_smiles: str | None = None

    def score(self) -> dict[str, object]:
        return evaluate_reaction(
            reaction_smiles=self.reaction_smiles,
            reactants_smiles=self.reactants_smiles,
            product_smiles=self.product_smiles,
            agents_smiles=self.agents_smiles,
        )

    def as_tuple(self) -> tuple[str, str, str | None]:
        if self.reaction_smiles:
            if self.reactants_smiles or self.product_smiles or self.agents_smiles:
                raise ReactionInputError(
                    "reaction_smiles cannot be combined with separate SMILES fields"
                )
            return split_reaction_smiles(self.reaction_smiles)
        if not self.reactants_smiles or not self.product_smiles:
            raise ReactionInputError(
                "provide reaction_smiles or both reactants_smiles and product_smiles"
            )
        return self.reactants_smiles, self.product_smiles, self.agents_smiles


class BatchEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reactions: list[EvaluationRequest] = Field(min_length=1, max_length=100)
    concurrency: int = Field(default=1, ge=1, le=16)


app = FastAPI(
    title="Chemical Score API",
    version=__version__,
    description=("可解释的多维化学反应规则评分。分数不是实验成功率、产率或安全结论。"),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/v1/metrics")
def metrics() -> dict[str, object]:
    return list_metrics()


@app.get("/v1/evidence/status")
def evidence_status() -> dict[str, object]:
    return get_default_evaluator().evidence_status()


@app.post("/v1/evaluations")
def evaluate(request: EvaluationRequest) -> dict[str, object]:
    try:
        result = request.score()
    except ReactionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result["status"] == "invalid_input":
        raise HTTPException(status_code=422, detail=result["errors"])
    return result


@app.post("/v1/evaluations/batch")
def evaluate_batch(request: BatchEvaluationRequest) -> dict[str, Any]:
    results: list[dict[str, object] | None] = [None] * len(request.reactions)
    valid_indices: list[int] = []
    valid_reactions: list[tuple[str, str, str | None]] = []
    for index, reaction in enumerate(request.reactions):
        try:
            valid_reactions.append(reaction.as_tuple())
            valid_indices.append(index)
        except ReactionInputError as exc:
            results[index] = {
                "status": "invalid_input",
                "score": None,
                "errors": [str(exc)],
                "index": index,
            }
    evaluated = get_default_evaluator().evaluate_many(
        valid_reactions, concurrency=request.concurrency
    )
    for index, result in zip(valid_indices, evaluated, strict=True):
        results[index] = result
    return {"count": len(results), "results": results}
