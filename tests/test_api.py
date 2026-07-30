from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_and_metric_catalog():
    assert client.get("/health").status_code == 200
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert len(response.json()["dimensions"]) == 4

    evidence = client.get("/v1/evidence/status")
    assert evidence.status_code == 200
    assert evidence.json()["configured"] is False


def test_single_evaluation_endpoint():
    response = client.post(
        "/v1/evaluations",
        json={"reaction_smiles": "CC(=O)O.CCO>>CCOC(C)=O"},
    )
    assert response.status_code == 200
    assert response.json()["score_tree"]["type"] == "total"


def test_invalid_smiles_returns_422():
    response = client.post(
        "/v1/evaluations",
        json={"reaction_smiles": "invalid>>CCO"},
    )
    assert response.status_code == 422


def test_http_schema_only_exposes_reaction_smiles():
    schema = client.get("/openapi.json").json()
    request_schema = schema["components"]["schemas"]["EvaluationRequest"]

    assert request_schema["required"] == ["reaction_smiles"]
    assert set(request_schema["properties"]) == {"reaction_smiles"}


def test_batch_keeps_item_level_errors():
    response = client.post(
        "/v1/evaluations/batch",
        json={
            "concurrency": 2,
            "reactions": [
                {"reaction_smiles": "CCO>>CC=O"},
                {"reaction_smiles": "bad-format"},
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][1]["status"] == "invalid_input"
