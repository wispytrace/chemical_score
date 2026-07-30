"""Backward-compatible wrappers for the original public functions.

New integrations should import :func:`chemical_score.evaluate_reaction`, which
returns the complete hierarchical 0-100 score tree.
"""

from __future__ import annotations

import math

from chemical_score import evaluate_reaction


def evaluate_expert_reward(reactants_smi: str, product_smi: str) -> float:
    """Return the legacy ``[-1, 1]`` reward derived from the new total score."""

    result = evaluate_reaction(
        reactants_smiles=reactants_smi,
        product_smiles=product_smi,
    )
    score = result.get("score")
    if score is None:
        return -1.0
    return max(-1.0, min(1.0, (float(score) - 50.0) / 50.0))


def check_reaction_feasibility(
    pred_reactants_smi: str, target_product_smi: str, threshold: float = 50.0
) -> bool:
    """Legacy boolean gate backed by the hierarchical score and critical flags."""

    result = evaluate_reaction(
        reactants_smiles=pred_reactants_smi,
        product_smiles=target_product_smi,
    )
    return bool(
        result.get("status") in {"success", "partial_success"}
        and result.get("score") is not None
        and float(result["score"]) >= threshold
        and not any(
            flag.get("severity") == "critical" for flag in result.get("flags", [])
        )
    )


def smooth_rl_reward(
    raw_score: float,
    center: float = 0.45,
    pos_temp: float = 1.6,
    neg_temp: float = 1.1,
    clip_range: tuple[float, float] = (-6.0, 6.0),
) -> float:
    """Retained for callers that used the original standalone transform."""

    shifted = max(clip_range[0], min(clip_range[1], raw_score - center))
    temperature = pos_temp if shifted >= 0 else neg_temp
    return math.tanh(shifted / temperature)


if __name__ == "__main__":
    examples = (
        ("CC(=O)O.CCO", "CCOC(=O)C"),
        ("CCOC(=O)C", "CCOC(=O)C"),
        ("F", "c1ccccc1"),
    )
    for reactants, product in examples:
        print(reactants, ">>", product, evaluate_expert_reward(reactants, product))
