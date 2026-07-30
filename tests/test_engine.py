from __future__ import annotations

from chemical_score import ReactionEvaluator, evaluate_reaction
from chemical_score.metrics.base import MetricRegistry, MetricSpec


def walk(node: dict):
    yield node
    for child in node.get("children", []):
        yield from walk(child)


def test_score_is_hierarchical_and_decomposable():
    result = evaluate_reaction(reaction_smiles="CC(=O)O.CCO>>CCOC(C)=O")

    assert result["status"] == "success"
    assert 0 <= result["score"] <= 100
    root = result["score_tree"]
    assert root["type"] == "total"
    assert {node["id"] for node in root["children"]} == {
        "feasibility",
        "evidence_support",
        "safety",
        "economy",
    }
    assert all(node["children"] for node in root["children"])
    leaf_ids = {node["id"] for node in walk(root) if node["type"] == "metric"}
    assert "functional_group_plausibility" in leaf_ids
    assert "atom_economy_estimate" in leaf_ids
    assert "structural_alerts" in leaf_ids
    evidence = next(
        node for node in root["children"] if node["id"] == "evidence_support"
    )
    assert evidence["status"] == "not_applicable"
    assert evidence["effective_weight"] == 0


def test_invalid_component_is_not_silently_dropped():
    result = evaluate_reaction(
        reactants_smiles="CCO.not-a-smiles",
        product_smiles="CCO",
    )

    assert result["status"] == "invalid_input"
    assert result["score"] is None
    assert "index 1" in result["errors"][0]


def test_identity_reaction_has_transparent_score_cap():
    result = evaluate_reaction(reactants_smiles="CCOC", product_smiles="CCOC")

    assert result["score"] <= 20
    assert any(flag["code"] == "identity_reaction" for flag in result["flags"])
    assert result["score_tree"]["evidence"]["critical_cap"] == 20


def test_identity_check_ignores_atom_map_annotations():
    result = evaluate_reaction(
        reactants_smiles="[CH3:1][CH2:2][OH:3]",
        product_smiles="[CH3:7][CH2:8][OH:9]",
    )

    assert any(flag["code"] == "identity_reaction" for flag in result["flags"])


def test_missing_product_elements_trigger_critical_flag():
    result = evaluate_reaction(reactants_smiles="O", product_smiles="c1ccccc1")

    assert result["score"] <= 35
    assert any(flag["code"] == "missing_core_elements" for flag in result["flags"])


def test_not_applicable_metric_is_not_treated_as_zero():
    result = evaluate_reaction(
        reactants_smiles="CC(=O)O.CCO", product_smiles="CCOC(C)=O"
    )
    leaves = {node["id"]: node for node in walk(result["score_tree"])}

    scaffold = leaves["scaffold_continuity"]
    assert scaffold["status"] == "not_applicable"
    assert scaffold["score"] is None
    assert scaffold["effective_weight"] == 0


class BrokenMetric:
    spec = MetricSpec(
        "broken",
        "broken",
        "test metric",
        ("feasibility", "consistency"),
    )

    def evaluate(self, context):
        raise RuntimeError("intentional")


def test_metric_failure_is_isolated_and_visible():
    evaluator = ReactionEvaluator(MetricRegistry([BrokenMetric()]))
    result = evaluator.evaluate("CCO", "CC=O")

    assert result["status"] == "partial_success"
    assert result["coverage"] == 0
    leaves = {node["id"]: node for node in walk(result["score_tree"])}
    assert leaves["broken"]["status"] == "error"
    assert "RuntimeError" in leaves["broken"]["warnings"][0]


def test_batch_preserves_order():
    evaluator = ReactionEvaluator()
    results = evaluator.evaluate_many(
        [("CCO", "CC=O", None), ("CCOC", "CCOC", None)], concurrency=2
    )

    assert len(results) == 2
    assert results[0]["reaction"]["main_product_smiles"] == "CC=O"
    assert any(flag["code"] == "identity_reaction" for flag in results[1]["flags"])


def test_agent_is_included_in_structural_safety_screening():
    result = evaluate_reaction(
        reactants_smiles="CCO",
        agents_smiles="COOC",
        product_smiles="CC=O",
    )
    leaves = {node["id"]: node for node in walk(result["score_tree"])}
    alerts = leaves["structural_alerts"]["evidence"]["alerts"]

    assert any(alert["component_role"] == "agent" for alert in alerts)
