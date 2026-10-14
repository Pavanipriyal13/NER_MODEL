"""
scripts/train_ner_fixed.py
==========================
Trains a new Medical NER model from en_core_web_sm.

This script replaces the original train_ner.py (baseline) and the iterative
train_iteration.py for the common case where you want to retrain from scratch
on the full consolidated training corpus.

PROCESS (easy to follow in order):
  1. Load training data   (preannotated_sentences.jsonl minus dev/test/eval)
  2. Load dev data        (dev.jsonl)
  3. Validate both sets   (offsets, labels, duplicates)
  4. Load en_core_web_sm
  5. Add custom labels
  6. Build spaCy Example objects
  7. Train for N_ITER epochs
     - Print epoch / loss / dev precision / dev recall / dev F1
     - Save the BEST dev-F1 checkpoint to memory
  8. Save the LAST epoch model  → models/ner_last
  9. Restore the BEST checkpoint → save to models/ner_best
 10. Report both saved locations

WHAT IS SAVED
-------------
  models/ner_last/  — model weights after the final training epoch.
  models/ner_best/  — model weights from the epoch with the highest dev F1.

  These are TRAINING checkpoints only.
  They are NOT automatically promoted to models/ner_final.

  To promote a model to production (models/ner_final), run:
      python scripts/compare_and_promote.py --candidate models/ner_best

IMPORTANT
---------
  models/ner_final is NEVER touched by this script.

Usage:
    python scripts/train_ner_fixed.py

Optional arguments (edit CONFIG below or pass via command line):
    --n_iter      25        number of training epochs
    --lr          0.001     learning rate
    --dropout     0.35      dropout during training
    --seed        42        random seed
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

# Allow running from the project root.
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from scripts.ner_eval import evaluate_model_on_jsonl, print_results, LABELS

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

BASE_MODEL  = "en_core_web_sm"
N_ITER      = 25
LEARN_RATE  = 0.001
DROPOUT     = 0.35
SEED        = 42

# Input data paths.
PREANNOTATED = PROJ_ROOT / "data" / "processed" / "preannotated_sentences.jsonl"
MANUAL_DATA  = PROJ_ROOT / "data" / "processed" / "manual_sentences_1000.jsonl"
DEV_FILE     = PROJ_ROOT / "data" / "processed" / "dev.jsonl"
TEST_FILE    = PROJ_ROOT / "data" / "processed" / "test.jsonl"
EVAL_FILE    = PROJ_ROOT / "data" / "processed" / "realistic_eval.jsonl"

# Output model paths.  NEVER points at ner_final.
MODEL_LAST = PROJ_ROOT / "models" / "ner_last"
MODEL_BEST = PROJ_ROOT / "models" / "ner_best"

# Combination-drug patterns for the EntityRuler.
# These are re-attached after training so the ruler is always present at
# inference time without affecting what the NER component learns.
COMBO_DRUG_NAMES = [
    "Frusemide + Spironolactone",
    "Ramipril + Amlodipine",
    "Ramipril + Hydrochlorothiazide",
    "Ramipril + Metoprolol Succinate Extended Release",
    "Telmisartan + Amlodipine",
    "Telmisartan + Amlodipine + Hydrochlorothiazide",
    "Glibenclamide + Metformin Hydrochloride (Sustained Release)",
    "Metformin Hydrochloride (Prolonged Release) + Glimepiride",
    "Metformin Hydrochloride (Prolonged Release) 1000 mg + Glimepiride",
    "Metformin Hydrochloride (SR) + Glimepiride + Voglibose",
    "Metformin Hydrochloride (Sustained Release) + Voglibose",
    "Glimepiride + Pioglitazone + Metformin Hydrochloride Extended Release",
    "Insulin Glargine + Lixisenatide",
    "Insulin Injection Soluble + Biphasic Isophane Insulin",
    "Beclomethasone Dipropionate + Phenylephrine Hydrochloride + Lignocaine Hydrochloride",
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl(path):
    """Load a JSONL file. Returns a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] JSON error on line {line_num} of {path}: {exc}")
    return records


def load_training_data(preannotated_path, dev_path, test_path, eval_path):
    """
    Load the training corpus from preannotated_sentences.jsonl plus
    manual_sentences_1000.jsonl, removing any sentences that appear
    in dev, test, or eval sets.

    Returns (train_records, dev_records).
    """
    # Build exclusion sets from the held-out files.
    excluded_texts = set()
    for p in [dev_path, test_path, eval_path]:
        if Path(p).exists():
            for rec in load_jsonl(p):
                excluded_texts.add(rec["text"])

    # Base corpus (preannotated) minus held-out sentences.
    all_pre = load_jsonl(preannotated_path)
    base_records = [r for r in all_pre if r["text"] not in excluded_texts]

    excluded_count = len(all_pre) - len(base_records)
    print(f"  preannotated_sentences.jsonl : {len(all_pre)} total")
    print(f"  Excluded (in dev/test/eval)  : {excluded_count}")
    print(f"  Base training records        : {len(base_records)}")

    # New manual sentences — also exclude any accidental overlaps.
    manual_records = []
    if MANUAL_DATA.exists():
        all_manual = load_jsonl(MANUAL_DATA)
        manual_records = [r for r in all_manual if r["text"] not in excluded_texts]
        manual_excluded = len(all_manual) - len(manual_records)
        print(f"  manual_sentences_1000.jsonl  : {len(all_manual)} total")
        if manual_excluded:
            print(f"  Excluded from manual         : {manual_excluded}")
        print(f"  Manual training records      : {len(manual_records)}")
    else:
        print(f"  manual_sentences_1000.jsonl  : NOT FOUND — skipping")

    train_records = base_records + manual_records
    print(f"  TOTAL training records       : {len(train_records)}")

    dev_records = load_jsonl(dev_path) if Path(dev_path).exists() else []
    print(f"  Dev records                  : {len(dev_records)}")

    return train_records, dev_records


# =============================================================================
# DATA VALIDATION (light, fast checks before training)
# =============================================================================

def validate_records(records, split_name):
    """
    Check for common annotation problems.
    Prints a summary. Returns the number of invalid records found.
    """
    invalid = 0
    for i, rec in enumerate(records):
        text = rec.get("text", "")
        for j, ent in enumerate(rec.get("entities", [])):
            s = ent.get("start", -1)
            e = ent.get("end",   -1)
            lbl = ent.get("label", "")
            stored_text = ent.get("text", "")

            # Offset bounds.
            if s < 0 or e > len(text) or e <= s:
                print(f"  [INVALID] {split_name} record {i}, entity {j}: "
                      f"bad offset [{s}:{e}] (text len={len(text)})")
                invalid += 1
                continue

            # Label check.
            if lbl not in LABELS:
                print(f"  [INVALID] {split_name} record {i}, entity {j}: "
                      f"unknown label '{lbl}'")
                invalid += 1

            # Text/span mismatch.
            if stored_text and text[s:e] != stored_text:
                print(f"  [WARN] {split_name} record {i}, entity {j}: "
                      f"stored text '{stored_text}' != actual '{text[s:e]}'")

    if invalid == 0:
        print(f"  {split_name}: validation passed ({len(records)} records, 0 errors)")
    else:
        print(f"  {split_name}: {invalid} invalid entities found")
    return invalid


# =============================================================================
# CONVERT RECORDS TO SPACY EXAMPLES
# =============================================================================

def make_examples(nlp, records):
    """
    Convert JSONL records into spaCy Example objects.

    Skips records where alignment fails and reports how many were skipped.
    Does NOT silently discard records without printing why.
    """
    examples = []
    skipped  = 0
    for i, rec in enumerate(records):
        text = rec["text"]
        ents = [
            (e["start"], e["end"], e["label"])
            for e in rec.get("entities", [])
            if e["label"] in LABELS
        ]
        try:
            doc     = nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": ents})
            examples.append(example)
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"  [SKIP] record {i}: alignment error → {exc}")
            elif skipped == 6:
                print(f"  [SKIP] (further skip messages suppressed...)")

    if skipped > 0:
        print(f"  Skipped {skipped} records due to alignment errors.")
    print(f"  Usable spaCy examples: {len(examples)}")
    return examples


# =============================================================================
# ATTACH ENTITY RULER (called AFTER training, before saving)
# =============================================================================

def attach_entity_ruler(nlp):
    """
    Add an EntityRuler for combination drug names.
    The ruler runs BEFORE the NER component and overwrites its output
    for any matched combination drug span.

    This is re-attached from scratch after training so it does not
    interfere with what the NER component learns during training.
    """
    # Remove any existing ruler to avoid duplicates.
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")

    ruler = nlp.add_pipe(
        "entity_ruler",
        before="ner",
        config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"},
    )
    patterns = [{"label": "DRUG", "pattern": name} for name in COMBO_DRUG_NAMES]
    ruler.add_patterns(patterns)
    print(f"  EntityRuler attached with {len(patterns)} combination-drug patterns.")
    return nlp


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(n_iter, learn_rate, dropout, seed):
    """
    Full training pipeline.

    Steps (in order, matching the docstring at the top of this file):
      1. Load data
      2. Validate data
      3. Load spaCy model
      4. Add custom labels
      5. Build Examples
      6. Train with best-checkpoint tracking
      7. Save last-epoch model
      8. Restore best checkpoint and save it
    """

    # --- Seed both Python random and numpy for reproducibility. ---
    # Python random controls: data shuffling.
    # numpy controls: spaCy/Thinc weight initialisation and optimiser steps.
    random.seed(seed)
    np.random.seed(seed)
    print(f"  Random seed set: {seed} (Python random + numpy)")

    # --- 1. Load data. ---
    print("\n[1] Loading data ...")
    train_records, dev_records = load_training_data(
        PREANNOTATED, DEV_FILE, TEST_FILE, EVAL_FILE
    )

    if not train_records:
        print("ERROR: No training records loaded. Check PREANNOTATED path.")
        sys.exit(1)
    if not dev_records:
        print("ERROR: No dev records loaded. Check DEV_FILE path.")
        sys.exit(1)

    # --- 2. Validate data. ---
    print("\n[2] Validating data ...")
    train_invalid = validate_records(train_records, "train")
    dev_invalid   = validate_records(dev_records,   "dev")
    if train_invalid + dev_invalid > 0:
        print(f"  WARNING: {train_invalid + dev_invalid} invalid entities found. "
              "They will be included in training (spaCy skips bad spans internally). "
              "Run scripts/validate_data.py for a full report.")

    # --- 3. Load the base model. ---
    print(f"\n[3] Loading base model: {BASE_MODEL} ...")
    nlp = spacy.load(BASE_MODEL)

    # --- 4. Add custom labels. ---
    # en_core_web_sm already has an 'ner' pipe; we add our labels to it.
    ner = nlp.get_pipe("ner")
    for lbl in LABELS:
        ner.add_label(lbl)
    print(f"  Added labels: {LABELS}")

    # --- 5. Build spaCy Example objects. ---
    print("\n[5] Building spaCy Examples ...")
    train_examples = make_examples(nlp, train_records)
    dev_examples   = make_examples(nlp, dev_records)

    # --- 6. Training loop. ---
    print(f"\n[6] Training for {n_iter} epochs (LR={learn_rate}, dropout={dropout}) ...")
    print(f"  {'Epoch':>5}  {'Loss':>10}  {'Dev P':>7}  {'Dev R':>7}  "
          f"{'Dev F1':>7}  {'Best?':>6}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}")

    # Only train the NER component; keep the other pipes frozen.
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    best_dev_f1    = -1.0   # highest dev F1 seen so far
    best_epoch     = 0      # epoch number (1-based) where best was achieved
    best_model_bytes = None # nlp.to_bytes() snapshot of the best state

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        optimizer.learn_rate = learn_rate

        for epoch in range(1, n_iter + 1):
            # Shuffle training data each epoch (important for convergence).
            random.shuffle(train_examples)
            losses = {}

            # Train in small batches that grow from 4 to 32.
            for batch in minibatch(train_examples, size=compounding(4.0, 32.0, 1.001)):
                nlp.update(batch, drop=dropout, losses=losses)

            # Evaluate on the dev set using spaCy's built-in ents_f score.
            # This is fast and consistent with what spaCy uses internally.
            dev_scores  = nlp.evaluate(dev_examples)
            dev_p  = dev_scores.get("ents_p", 0.0)
            dev_r  = dev_scores.get("ents_r", 0.0)
            dev_f1 = dev_scores.get("ents_f", 0.0)
            loss   = losses.get("ner", 0.0)

            # Track the best checkpoint.
            is_best = dev_f1 > best_dev_f1
            if is_best:
                best_dev_f1      = dev_f1
                best_epoch       = epoch
                best_model_bytes = nlp.to_bytes()  # snapshot weights in memory

            marker = " ←BEST" if is_best else ""
            print(f"  {epoch:>5}  {loss:>10.2f}  {dev_p:>7.4f}  {dev_r:>7.4f}  "
                  f"{dev_f1:>7.4f}{marker}")

    print(f"\n  Best dev F1: {best_dev_f1:.4f} at epoch {best_epoch}")

    # --- 7. Save the LAST epoch model. ---
    print(f"\n[7] Saving last-epoch model → {MODEL_LAST} ...")
    # Re-attach the EntityRuler before saving.
    nlp = attach_entity_ruler(nlp)
    MODEL_LAST.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_LAST)
    print(f"  Last-epoch model saved.")

    # --- 8. Restore the BEST checkpoint and save it. ---
    print(f"\n[8] Restoring best checkpoint (epoch {best_epoch}) → {MODEL_BEST} ...")
    if best_model_bytes is None:
        # This should not happen unless n_iter == 0.
        print("  WARNING: No best checkpoint recorded. Saving current state as best.")
        best_model_bytes = nlp.to_bytes()

    # Load a fresh copy from the base model to restore into, so the
    # EntityRuler from the last-epoch save does not carry over.
    nlp_best = spacy.load(BASE_MODEL)
    ner_best = nlp_best.get_pipe("ner")
    for lbl in LABELS:
        ner_best.add_label(lbl)

    # from_bytes restores the trained NER weights into nlp_best.
    nlp_best.from_bytes(best_model_bytes)

    # Re-attach the EntityRuler to the best model before saving.
    nlp_best = attach_entity_ruler(nlp_best)

    MODEL_BEST.mkdir(parents=True, exist_ok=True)
    nlp_best.to_disk(MODEL_BEST)
    print(f"  Best checkpoint saved.")

    # --- 9. Verify both models load and give sensible output. ---
    print("\n[9] Smoke-testing saved models ...")
    smoke_text = "The patient has migraine and is taking Metformin for type 2 diabetes."
    for name, path in [("ner_last", MODEL_LAST), ("ner_best", MODEL_BEST)]:
        nlp_check = spacy.load(path)
        doc = nlp_check(smoke_text)
        preds = [(e.text, e.label_) for e in doc.ents if e.label_ in set(LABELS)]
        print(f"  {name}: {preds}")

    # --- 10. Final report. ---
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  TRAINING COMPLETE
╠══════════════════════════════════════════════════════════════════╣
║  Best dev F1 : {best_dev_f1:.4f}  (epoch {best_epoch} of {n_iter})
║  models/ner_last  → last epoch weights (may not be the best)
║  models/ner_best  → best dev-F1 checkpoint
║
║  models/ner_final → NOT TOUCHED (still the approved model)
║
║  Next step: run compare_and_promote.py to decide whether to
║  promote ner_best to production.
╚══════════════════════════════════════════════════════════════════╝
""")

    return {
        "best_dev_f1": best_dev_f1,
        "best_epoch":  best_epoch,
        "model_last":  str(MODEL_LAST),
        "model_best":  str(MODEL_BEST),
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Medical NER from en_core_web_sm."
    )
    parser.add_argument("--n_iter",  type=int,   default=N_ITER,      help="Training epochs")
    parser.add_argument("--lr",      type=float, default=LEARN_RATE,  help="Learning rate")
    parser.add_argument("--dropout", type=float, default=DROPOUT,     help="Dropout")
    parser.add_argument("--seed",    type=int,   default=SEED,        help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        n_iter   = args.n_iter,
        learn_rate = args.lr,
        dropout  = args.dropout,
        seed     = args.seed,
    )
