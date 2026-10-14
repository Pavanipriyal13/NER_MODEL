# Custom Medical NER Model

A custom Named Entity Recognition (NER) model that extracts four medical entity types from natural-language queries.

---

## Supported Labels

| Label | Definition | Examples |
|---|---|---|
| `DRUG` | A medication or drug name | `Metformin`, `Frusemide + Spironolactone` |
| `DISEASE` | A recognised disease or disorder | `type 2 diabetes`, `migraine`, `dementia` |
| `SYMPTOM` | A patient-reported symptom | `headache`, `nausea`, `dyspnea` |
| `CONDITION` | A physiological/functional health state | `pregnancy`, `renal impairment`, `cognitive impairment` |

---

## Example

**Input:**
```
Can Metformin be used during pregnancy in a patient with type 2 diabetes and headache?
```

**Output:**
```
Metformin       → DRUG
pregnancy       → CONDITION
type 2 diabetes → DISEASE
headache        → SYMPTOM
```

---

## Quick Start

```bash
# Interactive prediction
python predict.py

# Single sentence
python predict.py --text "Can Frusemide + Spironolactone be used for heart failure?"

# Demo sentences
python predict.py --demo

# Start REST API
python -m uvicorn app.main:app --port 8000
# Then open http://127.0.0.1:8000/docs
```

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ner` | Extract entities from a single text |
| `POST` | `/ner/batch` | Extract entities from multiple texts |

```json
POST /ner
{"text": "Can Metformin be used during pregnancy?"}

→ {"text": "...", "entities": [
    {"text": "Metformin", "label": "DRUG",      "start": 4,  "end": 13},
    {"text": "pregnancy", "label": "CONDITION", "start": 29, "end": 38}
  ]}
```

---

## Python Usage

```python
from src.ner.model import NERModel

ner = NERModel()  # loads models/ner_final
entities = ner.predict("The patient has migraine and renal impairment.")
# [{"text": "migraine",         "label": "DISEASE",   "start": 16, "end": 24},
#  {"text": "renal impairment", "label": "CONDITION", "start": 29, "end": 45}]
```

---

## Model

- **Path:** `models/ner_final/`
- **Base:** `en_core_web_sm` (spaCy 3.5)
- **Components:** EntityRuler (combination drugs) + NER
- **Labels:** DRUG, DISEASE, SYMPTOM, CONDITION

---

## Training Data

**Source of truth:** `data/processed/preannotated_sentences.jsonl` — 2,357 annotated sentences

| Source | Sentences |
|---|---|
| Batch 1 (generated from `sanofi_drugs.csv`) | 2,000 |
| Batch 2 (combination-drug targeted) | 250 |
| Phase 7 manually authored corrections | 107 |
| **Total** | **2,357** |

Entity counts: DRUG=1,287 · DISEASE=1,155 · CONDITION=904 · SYMPTOM=890

---

## Evaluation

**Phase 7 realistic evaluation** (`data/processed/realistic_eval.jsonl` — 68 sentences, never trained on):

| Label | F1 |
|---|---|
| DRUG | 1.0000 |
| DISEASE | 0.9512 |
| SYMPTOM | 0.9048 |
| CONDITION | 1.0000 |
| **Micro F1** | **0.9540** |

Zero wrong-label errors · zero missed entities · zero false positives · 4 minor boundary errors

---

## Project Structure

```
NER_MODEL/
│
├── predict.py                    # CLI prediction script
├── README.md
│
├── app/                          # FastAPI REST API
│   ├── main.py
│   ├── schemas.py
│   └── services/ner_service.py
│
├── src/ner/model.py              # NERModel class — use this for inference
│
├── tests/
│   ├── test_ner_inference.py     # 24 inference tests
│   └── test_api.py               # 24 API tests
│
├── scripts/
│   ├── train_ner.py              # Train from scratch
│   ├── train_iteration.py        # Iterative targeted training
│   ├── evaluate_confusion.py     # Confusion matrix + per-label metrics
│   └── validate_annotations.py   # Validate annotation offsets
│
├── data/
│   ├── raw/sanofi_drugs.csv              # ~45 Sanofi drugs source
│   └── processed/
│       ├── preannotated_sentences.jsonl  # ★ Full annotated dataset (2,357)
│       ├── train.jsonl                   # Training split (1,602)
│       ├── dev.jsonl                     # Dev split (199)
│       ├── test.jsonl                    # Test split (199, held out)
│       ├── realistic_eval.jsonl          # ★ Evaluation benchmark (68, held out)
│       ├── dataset_report.json           # Dataset statistics
│       └── phase7_final_report.json      # Final project report
│
└── models/
    └── ner_final/                # ★ The model
```

---

## Tests

```bash
python tests/test_ner_inference.py   # 24/24
python tests/test_api.py             # 24/24
```

---

## Dependencies

```bash
pip install spacy==3.5.4 fastapi==0.111.1 "uvicorn[standard]==0.30.1" httpx==0.27.0
python -m spacy download en_core_web_sm
```

---

## Limitations

- Trained on ~45 Sanofi drugs — unseen drug names may be missed
- Combination-drug EntityRuler covers only the 15 patterns from `sanofi_drugs.csv`
- `migraine` as DISEASE vs SYMPTOM depends on sentence context — unusual phrasings may error
- 4 minor span-boundary errors remain on the 68-sentence evaluation set
- Not clinically validated — do not use for clinical decision-making
