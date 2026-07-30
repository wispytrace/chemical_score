from reaction_ai_score import ReactionEvaluator
from scorer import check_reaction_feasibility, evaluate_expert_reward


def test_legacy_reward_range_and_class_import():
    reward = evaluate_expert_reward("CCO", "CC=O")
    assert -1 <= reward <= 1
    assert ReactionEvaluator().evaluate("CCO", "CC=O")["score"] is not None


def test_legacy_feasibility_rejects_identity():
    assert not check_reaction_feasibility("CCO", "CCO")
