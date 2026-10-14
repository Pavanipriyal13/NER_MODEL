"""
tests/test_api.py
=================
API tests for the Medical NER FastAPI application.

Uses FastAPI's TestClient (backed by httpx) — no running server needed.

Run:
    python tests/test_api.py
    # or with pytest:
    python -m pytest tests/test_api.py -v

Tests cover:
    - GET /health
    - POST /ner  — DRUG, DISEASE, SYMPTOM, CONDITION, multiple entities,
                   combination drug, no entities, empty text, invalid request
    - POST /ner/batch — multiple texts, single text, empty list
    - Offset correctness
    - Only four labels returned
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app

# One client for all tests — model is loaded once inside ner_service
client = TestClient(app)

ALLOWED_LABELS = {"DRUG", "DISEASE", "SYMPTOM", "CONDITION"}


# =============================================================================
# HELPERS
# =============================================================================

def ner(text: str):
    """POST /ner and return the parsed response dict."""
    r = client.post("/ner", json={"text": text})
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    return r.json()


def labels_for(text: str):
    return {e["label"] for e in ner(text)["entities"]}


def entity_texts_for(text: str, label: str):
    return [e["text"] for e in ner(text)["entities"] if e["label"] == label]


# =============================================================================
# HEALTH CHECK
# =============================================================================

def test_health_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data


# =============================================================================
# POST /ner — basic entity extraction
# =============================================================================

def test_ner_drug():
    data = ner("What are the side effects of Amiodarone?")
    drugs = [e["text"] for e in data["entities"] if e["label"] == "DRUG"]
    assert "Amiodarone" in drugs, f"Expected Amiodarone as DRUG, got: {drugs}"


def test_ner_disease():
    data = ner("What is the treatment for hypertension?")
    diseases = [e["text"] for e in data["entities"] if e["label"] == "DISEASE"]
    assert any("hypertension" in d.lower() for d in diseases), (
        f"Expected hypertension as DISEASE, got: {diseases}"
    )


def test_ner_symptom():
    data = ner("The patient is experiencing nausea.")
    syms = [e["text"] for e in data["entities"] if e["label"] == "SYMPTOM"]
    assert "nausea" in syms, f"Expected nausea as SYMPTOM, got: {syms}"


def test_ner_condition():
    data = ner("Is this medication safe during pregnancy?")
    conds = [e["text"] for e in data["entities"] if e["label"] == "CONDITION"]
    assert "pregnancy" in conds, f"Expected pregnancy as CONDITION, got: {conds}"


def test_ner_multiple_entities():
    data = ner("Can Metformin be used for diabetes in renal impairment?")
    lab = {e["label"] for e in data["entities"]}
    assert "DRUG"      in lab, data
    assert "DISEASE"   in lab, data
    assert "CONDITION" in lab, data


def test_ner_combination_drug_two_parts():
    """Full two-drug combination must be ONE DRUG entity."""
    data = ner("Can Frusemide + Spironolactone be used for heart failure?")
    drugs = [e["text"] for e in data["entities"] if e["label"] == "DRUG"]
    assert "Frusemide + Spironolactone" in drugs, (
        f"Expected full combo drug, got: {drugs}"
    )
    assert len(drugs) == 1, f"Combo drug must not be split: {drugs}"


def test_ner_combination_drug_three_parts():
    data = ner(
        "Is Metformin Hydrochloride (SR) + Glimepiride + Voglibose appropriate "
        "for type 2 diabetes?"
    )
    drugs = [e["text"] for e in data["entities"] if e["label"] == "DRUG"]
    assert "Metformin Hydrochloride (SR) + Glimepiride + Voglibose" in drugs, (
        f"Expected three-part combo as one entity, got: {drugs}"
    )


def test_ner_combination_drug_with_parentheses():
    data = ner(
        "Should Metformin Hydrochloride (Sustained Release) + Voglibose "
        "be avoided in severe hepatic impairment?"
    )
    drugs = [e["text"] for e in data["entities"] if e["label"] == "DRUG"]
    assert "Metformin Hydrochloride (Sustained Release) + Voglibose" in drugs, (
        f"Got: {drugs}"
    )


def test_ner_no_entities():
    data = ner("What are the available treatment options?")
    assert data["entities"] == [], (
        f"Expected no entities for entity-free query, got: {data['entities']}"
    )


def test_ner_response_contains_text_field():
    text = "The patient has diabetes."
    data = ner(text)
    assert data["text"] == text


def test_ner_only_four_labels():
    """No built-in spaCy labels (PERSON, ORG, GPE…) should appear."""
    texts = [
        "Can Metformin be used for diabetes?",
        "Frusemide + Spironolactone for heart failure.",
        "The patient reports dizziness and nausea.",
    ]
    for text in texts:
        for e in ner(text)["entities"]:
            assert e["label"] in ALLOWED_LABELS, (
                f"Unexpected label '{e['label']}' in: {text}"
            )


def test_ner_offsets_correct():
    """text[start:end] must equal entity text for every prediction."""
    texts = [
        "Can Metformin be used for diabetes?",
        "Frusemide + Spironolactone for heart failure during pregnancy.",
        "The patient reports dizziness and nausea.",
    ]
    for text in texts:
        data = ner(text)
        for e in data["entities"]:
            actual = text[e["start"]:e["end"]]
            assert actual == e["text"], (
                f"Offset mismatch in '{text}': "
                f"stored='{e['text']}' slice='{actual}' [{e['start']}:{e['end']}]"
            )


# =============================================================================
# POST /ner — validation errors
# =============================================================================

def test_ner_empty_string_422():
    r = client.post("/ner", json={"text": ""})
    assert r.status_code == 422, f"Expected 422 for empty text, got {r.status_code}"


def test_ner_whitespace_only_422():
    r = client.post("/ner", json={"text": "   "})
    assert r.status_code == 422, f"Expected 422 for whitespace-only, got {r.status_code}"


def test_ner_missing_text_field_422():
    r = client.post("/ner", json={})
    assert r.status_code == 422, f"Expected 422 for missing field, got {r.status_code}"


def test_ner_wrong_type_422():
    r = client.post("/ner", json={"text": 12345})
    # pydantic coerces int to str in v1 — accept 200 or 422
    # The important thing is the server doesn't crash (no 500)
    assert r.status_code in (200, 422), (
        f"Expected 200 or 422, got {r.status_code}: {r.text}"
    )


# =============================================================================
# POST /ner/batch
# =============================================================================

def test_batch_multiple_texts():
    texts = [
        "The patient has headache.",
        "Can Metformin be used during pregnancy?",
        "Frusemide + Spironolactone for heart failure.",
    ]
    r = client.post("/ner/batch", json={"texts": texts})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == len(texts)
    # Each result has the correct text
    for i, result in enumerate(data["results"]):
        assert result["text"] == texts[i]


def test_batch_single_text():
    r = client.post("/ner/batch", json={"texts": ["The patient has diabetes."]})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 1


def test_batch_preserves_order():
    texts = [
        "Can Metformin be used for diabetes?",
        "What are the available treatment options?",
        "Is Telmisartan safe in renal impairment?",
    ]
    r = client.post("/ner/batch", json={"texts": texts})
    assert r.status_code == 200
    results = r.json()["results"]
    for i, res in enumerate(results):
        assert res["text"] == texts[i], (
            f"Order mismatch at index {i}: expected '{texts[i]}', got '{res['text']}'"
        )


def test_batch_offsets_correct():
    texts = [
        "Can Metformin be used for diabetes?",
        "Frusemide + Spironolactone for heart failure.",
    ]
    r = client.post("/ner/batch", json={"texts": texts})
    assert r.status_code == 200
    for result in r.json()["results"]:
        text = result["text"]
        for e in result["entities"]:
            actual = text[e["start"]:e["end"]]
            assert actual == e["text"], (
                f"Offset mismatch: stored='{e['text']}' slice='{actual}'"
            )


def test_batch_empty_list_422():
    r = client.post("/ner/batch", json={"texts": []})
    assert r.status_code == 422, f"Expected 422 for empty list, got {r.status_code}"


def test_batch_combination_drug():
    """Combo drug must come through correctly in batch mode."""
    r = client.post("/ner/batch", json={
        "texts": ["Is Ramipril + Hydrochlorothiazide indicated for hypertension?"]
    })
    assert r.status_code == 200
    entities = r.json()["results"][0]["entities"]
    drugs = [e["text"] for e in entities if e["label"] == "DRUG"]
    assert "Ramipril + Hydrochlorothiazide" in drugs, (
        f"Combo drug not preserved in batch: {drugs}"
    )


# =============================================================================
# SIMPLE TEST RUNNER (no pytest required)
# =============================================================================

if __name__ == "__main__":
    tests = [
        test_health_root,
        test_health_endpoint,
        test_ner_drug,
        test_ner_disease,
        test_ner_symptom,
        test_ner_condition,
        test_ner_multiple_entities,
        test_ner_combination_drug_two_parts,
        test_ner_combination_drug_three_parts,
        test_ner_combination_drug_with_parentheses,
        test_ner_no_entities,
        test_ner_response_contains_text_field,
        test_ner_only_four_labels,
        test_ner_offsets_correct,
        test_ner_empty_string_422,
        test_ner_whitespace_only_422,
        test_ner_missing_text_field_422,
        test_ner_wrong_type_422,
        test_batch_multiple_texts,
        test_batch_single_text,
        test_batch_preserves_order,
        test_batch_offsets_correct,
        test_batch_empty_list_422,
        test_batch_combination_drug,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
