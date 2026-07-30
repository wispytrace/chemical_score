from __future__ import annotations

import json

import pytest

from chemical_score import EvidenceIndex, EvidenceRecord, ReactionEvaluator
from chemical_score.context import ReactionContext
from chemical_score.evidence import transformation_signature


def walk(node: dict):
    yield node
    for child in node.get("children", []):
        yield from walk(child)


def test_exact_precedent_and_outcome_support_are_exposed():
    index = EvidenceIndex(
        [
            EvidenceRecord(
                identifier="ord-1",
                reaction_smiles="CC(=O)O.CCO>>CCOC(C)=O",
                yield_percent=82.0,
                success=True,
                source="unit-test",
            ),
            EvidenceRecord(
                identifier="oxidation-1",
                reaction_smiles="CCO>>CC=O",
                yield_percent=65.0,
            ),
        ]
    )
    evaluator = ReactionEvaluator(evidence_index=index)
    assert evaluator.evidence_status()["record_count"] == 2

    result = evaluator.evaluate("CC(=O)O.CCO", "CCOC(C)=O")
    nodes = {node["id"]: node for node in walk(result["score_tree"])}

    assert nodes["evidence_support"]["status"] == "evaluated"
    assert nodes["nearest_reaction_similarity"]["score"] == 100
    assert nodes["exact_reaction_precedent"]["raw_value"] == 1
    assert nodes["historical_outcome_support"]["score"] == pytest.approx(82.0)
    neighbor = nodes["nearest_reaction_similarity"]["evidence"]["neighbors"][0]
    assert neighbor["identifier"] == "ord-1"
    assert neighbor["source"] == "unit-test"


def test_unconfigured_evidence_does_not_change_existing_total():
    evaluator = ReactionEvaluator()
    result = evaluator.evaluate("CC(=O)O.CCO", "CCOC(C)=O")
    dimensions = {node["id"]: node for node in result["score_tree"]["children"]}

    assert dimensions["evidence_support"]["score"] is None
    assert dimensions["evidence_support"]["effective_weight"] == 0
    assert dimensions["feasibility"]["effective_weight"] == pytest.approx(0.6)
    assert dimensions["safety"]["effective_weight"] == pytest.approx(0.2)
    assert dimensions["economy"]["effective_weight"] == pytest.approx(0.2)


def test_mapped_bond_edit_signature_ignores_map_number_identity():
    reference = ReactionContext(
        "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]",
        "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]",
    )
    query = ReactionContext(
        "[CH3:11][CH2:12][C:13](=[O:14])[OH:15].[CH3:16][OH:17]",
        "[CH3:11][CH2:12][C:13](=[O:14])[O:17][CH3:16]",
    )

    reference_signature = transformation_signature(
        reference.reactant_mols, reference.product_mols
    )
    query_signature = transformation_signature(query.reactant_mols, query.product_mols)

    assert reference_signature == query_signature
    assert reference_signature == "broken:C-O:1|formed:C-O:1"

    index = EvidenceIndex(
        [
            {
                "reaction_smiles": (
                    "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>"
                    "[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]"
                )
            }
        ]
    )
    result = ReactionEvaluator(evidence_index=index).evaluate(
        query.input_reactants_smiles, query.input_product_smiles
    )
    nodes = {node["id"]: node for node in walk(result["score_tree"])}
    assert nodes["mapped_transformation_precedent"]["raw_value"] == 1


def test_jsonl_loader_can_skip_invalid_records(tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "valid",
                        "reaction_smiles": "CCO>>CC=O",
                        "success": True,
                    }
                ),
                json.dumps({"id": "invalid", "reaction_smiles": "bad"}),
            ]
        ),
        encoding="utf-8",
    )

    index = EvidenceIndex.from_file(path, strict=False)

    assert len(index) == 1
    assert index.status()["rejected_count"] == 1
