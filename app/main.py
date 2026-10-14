"""
app/main.py
===========
FastAPI application for the custom Medical NER model.

Endpoints:
    GET  /          — health check
    GET  /health    — health check (alias)
    POST /ner       — extract entities from a single text
    POST /ner/batch — extract entities from multiple texts

Run:
    uvicorn app.main:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import (
    NERRequest,
    NERResponse,
    NERBatchRequest,
    NERBatchResponse,
    Entity,
)
from app.services import ner_service

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Medical NER API",
    description=(
        "Custom Named Entity Recognition for medical drug-search queries.\n\n"
        "Supported entity labels:\n"
        "- **DRUG** — medication / drug name\n"
        "- **DISEASE** — recognised disease or disorder\n"
        "- **SYMPTOM** — patient-reported symptom\n"
        "- **CONDITION** — physiological / functional health state\n\n"
        "Model: `ner_final` (spaCy `en_core_web_sm` + EntityRuler for combination drugs)"
    ),
    version="1.0.0",
)


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/", tags=["health"], summary="Root health check")
@app.get("/health", tags=["health"], summary="Health check")
def health():
    """Returns a simple status message confirming the API is running."""
    return {"status": "ok", "model": "ner_final"}


# =============================================================================
# POST /ner  — single text
# =============================================================================

@app.post(
    "/ner",
    response_model=NERResponse,
    tags=["NER"],
    summary="Extract entities from a single text",
    response_description="The input text and its extracted medical entities",
)
def extract_entities(request: NERRequest):
    """
    Run NER on a single medical query.

    **Request body:**
    ```json
    {"text": "Can Metformin be used during pregnancy?"}
    ```

    **Response:**
    ```json
    {
      "text": "Can Metformin be used during pregnancy?",
      "entities": [
        {"text": "Metformin", "label": "DRUG",      "start": 4,  "end": 13},
        {"text": "pregnancy", "label": "CONDITION", "start": 29, "end": 38}
      ]
    }
    ```

    Only DRUG / DISEASE / SYMPTOM / CONDITION entities are returned.
    Built-in spaCy labels (PERSON, ORG, GPE, …) are filtered out.
    """
    try:
        raw = ner_service.run(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NER inference error: {exc}")

    entities = [Entity(**e) for e in raw]
    return NERResponse(text=request.text, entities=entities)


# =============================================================================
# POST /ner/batch  — multiple texts
# =============================================================================

@app.post(
    "/ner/batch",
    response_model=NERBatchResponse,
    tags=["NER"],
    summary="Extract entities from multiple texts",
    response_description="NER results for every input text, in order",
)
def extract_entities_batch(request: NERBatchRequest):
    """
    Run NER on a list of medical queries in a single request.

    **Request body:**
    ```json
    {
      "texts": [
        "The patient has headache.",
        "Can Metformin be used during pregnancy?"
      ]
    }
    ```

    **Response:**
    ```json
    {
      "results": [
        {
          "text": "The patient has headache.",
          "entities": [{"text": "headache", "label": "SYMPTOM", "start": 16, "end": 24}]
        },
        {
          "text": "Can Metformin be used during pregnancy?",
          "entities": [
            {"text": "Metformin", "label": "DRUG",      "start": 4,  "end": 13},
            {"text": "pregnancy", "label": "CONDITION", "start": 29, "end": 38}
          ]
        }
      ]
    }
    ```
    """
    try:
        batch_raw = ner_service.run_batch(request.texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NER inference error: {exc}")

    results = [
        NERResponse(text=text, entities=[Entity(**e) for e in raw])
        for text, raw in zip(request.texts, batch_raw)
    ]
    return NERBatchResponse(results=results)
