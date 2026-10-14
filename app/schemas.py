"""
app/schemas.py
==============
Pydantic request and response schemas for the NER API.

All schemas validate their fields automatically; FastAPI surfaces
validation errors as 422 Unprocessable Entity responses.
"""

from typing import List
from pydantic import BaseModel, validator


# =============================================================================
# SHARED
# =============================================================================

class Entity(BaseModel):
    """A single extracted entity."""
    text:  str
    label: str   # one of DRUG / DISEASE / SYMPTOM / CONDITION
    start: int   # character start offset (inclusive)
    end:   int   # character end offset (exclusive)


# =============================================================================
# POST /ner  — single text
# =============================================================================

class NERRequest(BaseModel):
    """Request body for POST /ner."""
    text: str

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v

    class Config:
        schema_extra = {
            "example": {
                "text": "Can Metformin be used during pregnancy?"
            }
        }


class NERResponse(BaseModel):
    """Response body for POST /ner."""
    text:     str
    entities: List[Entity]

    class Config:
        schema_extra = {
            "example": {
                "text": "Can Metformin be used during pregnancy?",
                "entities": [
                    {"text": "Metformin", "label": "DRUG",      "start": 4,  "end": 13},
                    {"text": "pregnancy", "label": "CONDITION", "start": 29, "end": 38},
                ],
            }
        }


# =============================================================================
# POST /ner/batch  — multiple texts
# =============================================================================

class NERBatchRequest(BaseModel):
    """Request body for POST /ner/batch."""
    texts: List[str]

    @validator("texts")
    def texts_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("texts list must not be empty")
        return v

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "The patient has headache.",
                    "Can Metformin be used during pregnancy?",
                ]
            }
        }


class NERBatchResponse(BaseModel):
    """Response body for POST /ner/batch."""
    results: List[NERResponse]
