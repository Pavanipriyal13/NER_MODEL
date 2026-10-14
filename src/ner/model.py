"""
src/ner/model.py
================
Reusable inference module for the custom medical NER model.

Usage:
    from src.ner.model import NERModel

    # Load once at application start-up
    ner = NERModel()

    # Use for every prediction
    entities = ner.predict("Can Metformin be used for diabetes?")
    # [{"text": "Metformin", "label": "DRUG", "start": 4, "end": 13},
    #  {"text": "diabetes", "label": "DISEASE", "start": 27, "end": 35}]

Alternatively use the module-level helpers:
    from src.ner.model import load_model, predict

    nlp = load_model()
    entities = predict(nlp, "Can Metformin be used for diabetes?")
"""

from pathlib import Path
from typing import List, Dict

import spacy

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default path — resolved relative to this file so it works from any cwd
_THIS_DIR  = Path(__file__).resolve().parent        # src/ner/
_PROJ_ROOT = _THIS_DIR.parent.parent                # NER_MODEL/
DEFAULT_MODEL_PATH = _PROJ_ROOT / "models" / "ner_final"

# The four project labels — the only ones we expose in results
SUPPORTED_LABELS = {"DRUG", "DISEASE", "SYMPTOM", "CONDITION"}


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

def load_model(model_path=None) -> spacy.language.Language:
    """
    Load and return the spaCy NER pipeline.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the spaCy model directory.
        Defaults to models/ner_final/ relative to the project root.

    Returns
    -------
    spacy.language.Language
        The loaded pipeline, ready for inference.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    nlp  = spacy.load(path)
    return nlp


def predict(nlp: spacy.language.Language, text: str) -> List[Dict]:
    """
    Run NER on a single text string.

    Parameters
    ----------
    nlp  : loaded spaCy pipeline (from load_model)
    text : plain-text query, e.g. "Can Metformin be used for diabetes?"

    Returns
    -------
    List of dicts, each with keys:
        text   – entity surface form
        label  – one of DRUG / DISEASE / SYMPTOM / CONDITION
        start  – character start offset (inclusive)
        end    – character end offset (exclusive)

    Only entities with one of the four supported labels are returned.
    spaCy's built-in en_core_web_sm labels (PERSON, ORG, GPE, etc.)
    are filtered out automatically.
    """
    if not text or not text.strip():
        return []
    doc = nlp(text)
    return [
        {
            "text":  ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end":   ent.end_char,
        }
        for ent in doc.ents
        if ent.label_ in SUPPORTED_LABELS
    ]


# =============================================================================
# CLASS INTERFACE
# =============================================================================

class NERModel:
    """
    Thin wrapper that loads the model once and exposes a predict() method.

    Example
    -------
        ner = NERModel()
        results = ner.predict("The patient has headache.")
        # [{"text": "headache", "label": "SYMPTOM", "start": 16, "end": 24}]
    """

    def __init__(self, model_path=None):
        """
        Load the model at construction time.
        Pass model_path to override the default models/ner_final/ location.
        """
        self._nlp = load_model(model_path)

    def predict(self, text: str) -> List[Dict]:
        """
        Run NER and return a list of entity dicts.

        Parameters
        ----------
        text : str — the input query

        Returns
        -------
        list of {"text", "label", "start", "end"} dicts
        """
        return predict(self._nlp, text)

    @property
    def labels(self):
        """Return the four supported NER labels."""
        return sorted(SUPPORTED_LABELS)
