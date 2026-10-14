"""
app/services/ner_service.py
===========================
NER service layer.

Loads the final model ONCE at module import time.
All endpoint handlers call run() — no per-request model loading.

The service filters output to the four project labels only:
    DRUG / DISEASE / SYMPTOM / CONDITION
"""

import sys
from pathlib import Path
from typing import List, Dict

# Allow running the app from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ner.model import NERModel

# Load the model once when the module is first imported.
# FastAPI imports this module at startup, so the model is ready
# before any request arrives.
_model = NERModel()   # loads models/ner_final/


def run(text: str) -> List[Dict]:
    """
    Run NER on a single text string.

    Parameters
    ----------
    text : str — raw input query

    Returns
    -------
    List of entity dicts:
        {"text": str, "label": str, "start": int, "end": int}
    Only DRUG / DISEASE / SYMPTOM / CONDITION entities are returned.
    """
    return _model.predict(text)


def run_batch(texts: List[str]) -> List[List[Dict]]:
    """
    Run NER on a list of texts.

    Returns a list of entity lists — one per input text, in the same order.
    """
    return [_model.predict(t) for t in texts]
