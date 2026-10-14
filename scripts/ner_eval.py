"""
scripts/ner_eval.py
===================
Shared evaluation module for the custom Medical NER project.

This module provides ONE canonical implementation of entity-level evaluation
that is used by:
  - scripts/train_iteration.py   (during training and promotion comparisons)
  - scripts/evaluate_confusion.py (standalone analysis tool)
  - tests/test_evaluation.py     (unit tests with known values)

HOW ENTITY MATCHING WORKS
--------------------------
A prediction is correct only if ALL THREE match exactly:
  - start character offset
  - end character offset
  - entity label

This is called an "exact match" or true positive (TP).

Every gold entity and every predicted entity is counted at most once.

ERROR CATEGORIES
----------------
The function returns two groups of numbers:

1. METRIC COUNTS (used in precision / recall / F1 formulas):
   - tp[label]  : exact matches
   - fp[label]  : predictions that are wrong (wrong label or wrong boundary
                  or no gold entity nearby)
   - fn[label]  : gold entities the model missed or got wrong

2. DIAGNOSTIC COUNTS (explain WHY errors happened, for human review):
   - exact_matches   : number of perfectly correct predictions
   - wrong_label     : same span, different label (1 FN + 1 FP in metrics)
   - boundary_errors : overlapping span but different offsets (1 FN + 1 FP)
   - missed          : gold entity with no overlapping prediction (1 FN only)
   - extra_predictions: predicted entity with no gold overlap at all (1 FP only)

IMPORTANT: "extra_predictions" (standalone FP) is DIFFERENT from fp[label].
  fp[label] = wrong_label FP  +  boundary FP  +  standalone FP for that label.
  extra_predictions counts only the standalone category.
  This distinction is documented clearly so reports are not misleading.

FORMULAS
--------
  Precision[label] = tp[label] / (tp[label] + fp[label])
  Recall[label]    = tp[label] / (tp[label] + fn[label])
  F1[label]        = 2 * P * R / (P + R)   (0.0 if P+R == 0)

  Micro Precision  = sum(tp) / (sum(tp) + sum(fp))
  Micro Recall     = sum(tp) / (sum(tp) + sum(fn))
  Micro F1         = 2 * MicroP * MicroR / (MicroP + MicroR)

  Macro F1         = average of per-label F1 values (using unrounded values)
"""

from collections import defaultdict

# The four supported entity labels.
LABELS = ["DRUG", "DISEASE", "SYMPTOM", "CONDITION"]


# ---------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------

def evaluate_entities(gold_records, pred_records):
    """
    Evaluate NER predictions against gold annotations.

    Parameters
    ----------
    gold_records : list of dicts
        Each dict has:
          "text"     : str
          "entities" : list of {"start": int, "end": int, "label": str}

    pred_records : list of dicts
        Same structure as gold_records. The "text" field is matched by
        position (index), not by string comparison.

    Returns
    -------
    dict with keys:
      "per_label"  : dict[label] -> {tp, fp, fn, precision, recall, f1, support}
      "overall"    : aggregate metric and diagnostic counts
      "errors"     : list of individual error records for inspection
    """
    assert len(gold_records) == len(pred_records), (
        f"gold_records ({len(gold_records)}) and pred_records ({len(pred_records)}) "
        "must have the same length."
    )

    # --- Accumulate TP / FP / FN per label ---
    # These are the only numbers used in precision / recall / F1.
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # --- Accumulate diagnostic counters ---
    # These explain the nature of errors but are separate from metric counts.
    n_exact        = 0   # TP: correct span AND correct label
    n_wrong_label  = 0   # same span, wrong label   → 1 FN + 1 FP
    n_boundary     = 0   # overlapping span, any label mismatch → 1 FN + 1 FP
    n_missed       = 0   # gold entity, no overlapping prediction → 1 FN
    n_extra        = 0   # predicted entity, no gold overlap → 1 FP

    # --- Collect individual error records for human review ---
    errors = []

    for idx, (gold_rec, pred_rec) in enumerate(zip(gold_records, pred_records)):
        text = gold_rec["text"]

        # Build sets and dicts for fast lookup.
        # gold_span_map: (start, end) -> label
        # pred_span_map: (start, end) -> label
        gold_entities = [
            (e["start"], e["end"], e["label"])
            for e in gold_rec.get("entities", [])
            if e["label"] in LABELS
        ]
        pred_entities = [
            (e["start"], e["end"], e["label"])
            for e in pred_rec.get("entities", [])
            if e["label"] in LABELS
        ]

        gold_span_map = {(s, e): lbl for s, e, lbl in gold_entities}
        pred_span_map = {(s, e): lbl for s, e, lbl in pred_entities}

        # Track which gold and predicted spans have already been matched
        # so nothing is counted twice.
        matched_gold = set()   # (start, end) keys already accounted for
        matched_pred = set()   # (start, end) keys already accounted for

        # ------------------------------------------------------------------
        # PASS 1: Exact matches (correct span + correct label)
        # ------------------------------------------------------------------
        for (s, e), gold_lbl in gold_span_map.items():
            if (s, e) in pred_span_map and pred_span_map[(s, e)] == gold_lbl:
                tp[gold_lbl] += 1
                n_exact += 1
                matched_gold.add((s, e))
                matched_pred.add((s, e))

        # ------------------------------------------------------------------
        # PASS 2: Wrong-label errors (same span, different label)
        # ------------------------------------------------------------------
        for (s, e), gold_lbl in gold_span_map.items():
            if (s, e) in matched_gold:
                continue  # already counted as exact match
            if (s, e) in pred_span_map:
                pred_lbl = pred_span_map[(s, e)]
                # Same span, different label.
                fn[gold_lbl] += 1
                fp[pred_lbl] += 1
                n_wrong_label += 1
                matched_gold.add((s, e))
                matched_pred.add((s, e))
                errors.append({
                    "type":       "wrong_label",
                    "sentence":   text,
                    "entity":     text[s:e],
                    "gold_label": gold_lbl,
                    "pred_label": pred_lbl,
                    "start": s,
                    "end":   e,
                })

        # ------------------------------------------------------------------
        # PASS 3: Boundary errors and completely missed entities
        # (gold entities that had no exact match and no wrong-label match)
        # ------------------------------------------------------------------
        for s, e, gold_lbl in gold_entities:
            if (s, e) in matched_gold:
                continue  # already handled in Pass 1 or 2

            # Look for any predicted span that overlaps this gold span.
            # Overlap condition: pred starts before gold ends AND pred ends after gold starts.
            boundary_match = None
            for (ps, pe), pred_lbl in pred_span_map.items():
                if (ps, pe) in matched_pred:
                    continue  # this pred span is already matched to something
                if ps < e and pe > s:  # overlap
                    boundary_match = (ps, pe, pred_lbl)
                    break

            if boundary_match is not None:
                ps, pe, pred_lbl = boundary_match
                fn[gold_lbl] += 1
                fp[pred_lbl] += 1
                n_boundary += 1
                matched_gold.add((s, e))
                matched_pred.add((ps, pe))
                errors.append({
                    "type":        "boundary_error",
                    "sentence":    text,
                    "gold_entity": text[s:e],
                    "pred_entity": text[ps:pe],
                    "gold_label":  gold_lbl,
                    "pred_label":  pred_lbl,
                    "gold_start":  s, "gold_end": e,
                    "pred_start":  ps, "pred_end": pe,
                })
            else:
                # Gold entity with no overlapping prediction at all → missed.
                fn[gold_lbl] += 1
                n_missed += 1
                matched_gold.add((s, e))
                errors.append({
                    "type":       "missed",
                    "sentence":   text,
                    "entity":     text[s:e],
                    "gold_label": gold_lbl,
                    "start": s,
                    "end":   e,
                })

        # ------------------------------------------------------------------
        # PASS 4: Extra (standalone) predictions with no gold overlap at all
        # (predicted entities that were not matched to any gold entity)
        # ------------------------------------------------------------------
        for s, e, pred_lbl in pred_entities:
            if (s, e) in matched_pred:
                continue  # already counted in Pass 1, 2, or 3

            # This prediction has no gold overlap at all.
            fp[pred_lbl] += 1
            n_extra += 1
            matched_pred.add((s, e))
            errors.append({
                "type":       "extra_prediction",
                "sentence":   text,
                "entity":     text[s:e],
                "pred_label": pred_lbl,
                "start": s,
                "end":   e,
            })

    # ------------------------------------------------------------------
    # Compute per-label metrics using unrounded values,
    # round only for the final output dict.
    # ------------------------------------------------------------------
    per_label = {}
    total_tp = total_fp = total_fn = 0
    macro_f1_sum = 0.0

    for lbl in LABELS:
        t = tp[lbl]
        f = fp[lbl]
        n = fn[lbl]

        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall    = t / (t + n) if (t + n) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_label[lbl] = {
            "tp":        t,
            "fp":        f,
            "fn":        n,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   t + n,   # total gold entities for this label
        }

        total_tp  += t
        total_fp  += f
        total_fn  += n
        macro_f1_sum += f1  # use unrounded f1 for macro average

    # Macro F1: simple average of per-label F1 values (using unrounded values).
    macro_f1 = macro_f1_sum / len(LABELS)

    # Micro F1: pool all labels together.
    micro_precision = (total_tp / (total_tp + total_fp)
                       if (total_tp + total_fp) > 0 else 0.0)
    micro_recall    = (total_tp / (total_tp + total_fn)
                       if (total_tp + total_fn) > 0 else 0.0)
    micro_f1        = (2 * micro_precision * micro_recall
                       / (micro_precision + micro_recall)
                       if (micro_precision + micro_recall) > 0 else 0.0)

    # Count total gold and predicted entities (filtered to supported labels).
    total_gold_entities = sum(per_label[lbl]["support"] for lbl in LABELS)
    total_pred_entities = total_tp + total_fp

    overall = {
        # ---- Metric counts (used in formulas above) ----
        "total_tp":   total_tp,
        "total_fp":   total_fp,
        "total_fn":   total_fn,
        "total_gold_entities": total_gold_entities,
        "total_pred_entities": total_pred_entities,

        # ---- Aggregate metrics ----
        "micro_precision": round(micro_precision, 4),
        "micro_recall":    round(micro_recall,    4),
        "micro_f1":        round(micro_f1,        4),
        "macro_f1":        round(macro_f1,        4),

        # ---- Diagnostic counts (explain the nature of errors) ----
        # NOTE: These are NOT the same as fp / fn in the metric formulas.
        # Each diagnostic event maps to metric counts as follows:
        #   exact_matches:    +1 TP (no FP, no FN)
        #   wrong_label:      +1 FN for gold label, +1 FP for pred label
        #   boundary_errors:  +1 FN for gold label, +1 FP for pred label
        #   missed:           +1 FN only (no FP)
        #   extra_predictions:+1 FP only (no FN)
        "exact_matches":    n_exact,
        "wrong_label":      n_wrong_label,
        "boundary_errors":  n_boundary,
        "missed":           n_missed,
        "extra_predictions": n_extra,
    }

    return {
        "per_label": per_label,
        "overall":   overall,
        "errors":    errors,
    }


# ---------------------------------------------------------------------------
# CONVENIENCE WRAPPER: evaluate a spaCy model on a JSONL file
# ---------------------------------------------------------------------------

def evaluate_model_on_jsonl(nlp, jsonl_path):
    """
    Load a JSONL evaluation file, run the spaCy model on every sentence,
    and return evaluate_entities() results.

    Parameters
    ----------
    nlp       : loaded spaCy pipeline
    jsonl_path: str or Path to a JSONL file with {"text":..., "entities":[...]}

    Returns
    -------
    Same dict as evaluate_entities().
    """
    import json
    from pathlib import Path

    path = Path(jsonl_path)
    gold_records = []
    pred_records = []

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold_records.append(rec)

            # Run the model and build a pred record in the same format.
            doc = nlp(rec["text"])
            pred_ents = [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                for ent in doc.ents
                if ent.label_ in LABELS
            ]
            pred_records.append({"text": rec["text"], "entities": pred_ents})

    return evaluate_entities(gold_records, pred_records)


# ---------------------------------------------------------------------------
# PRETTY PRINTER
# ---------------------------------------------------------------------------

def print_results(results, title="EVALUATION RESULTS"):
    """Print a human-readable evaluation report to stdout."""
    ov = results["overall"]
    pl = results["per_label"]

    SEP = "─" * 68

    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

    print(f"\n  AGGREGATE METRICS")
    print(f"  {'Micro Precision':<22}: {ov['micro_precision']:.4f}")
    print(f"  {'Micro Recall':<22}: {ov['micro_recall']:.4f}")
    print(f"  {'Micro F1':<22}: {ov['micro_f1']:.4f}")
    print(f"  {'Macro F1':<22}: {ov['macro_f1']:.4f}")
    print(f"  {'Gold entities':<22}: {ov['total_gold_entities']}")
    print(f"  {'Predicted entities':<22}: {ov['total_pred_entities']}")

    print(f"\n  PER-LABEL METRICS")
    print(f"  {'Label':<12} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'Support':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*8}")
    for lbl in LABELS:
        m = pl[lbl]
        print(f"  {lbl:<12} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1']:>7.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
              f"{m['support']:>8}")

    print(f"\n  DIAGNOSTIC COUNTS")
    print(f"  (These explain error types. See ner_eval.py for exact definitions.)")
    print(f"  {'Exact matches':<28}: {ov['exact_matches']}")
    print(f"  {'Wrong-label errors':<28}: {ov['wrong_label']}")
    print(f"  {'Boundary errors':<28}: {ov['boundary_errors']}")
    print(f"  {'Missed (no overlap)':<28}: {ov['missed']}")
    print(f"  {'Extra predictions (no gold)':<28}: {ov['extra_predictions']}")

    # Explain the FP / diagnostic split explicitly.
    print(f"\n  NOTE: total_fp ({ov['total_fp']}) = wrong_label ({ov['wrong_label']}) "
          f"+ boundary ({ov['boundary_errors']}) + extra ({ov['extra_predictions']})")
    print(f"  NOTE: total_fn ({ov['total_fn']}) = wrong_label ({ov['wrong_label']}) "
          f"+ boundary ({ov['boundary_errors']}) + missed ({ov['missed']})")

    if results["errors"]:
        print(f"\n  FIRST 10 ERRORS")
        for err in results["errors"][:10]:
            t = err["type"]
            sent = err["sentence"][:70]
            if t == "exact_match":
                continue
            elif t == "wrong_label":
                print(f"  [wrong_label]   '{err['entity']}' "
                      f"{err['gold_label']}→{err['pred_label']}")
                print(f"    {sent}")
            elif t == "boundary_error":
                print(f"  [boundary]      gold='{err['gold_entity']}' "
                      f"pred='{err['pred_entity']}' "
                      f"{err['gold_label']}→{err['pred_label']}")
                print(f"    {sent}")
            elif t == "missed":
                print(f"  [missed]        '{err['entity']}' ({err['gold_label']})")
                print(f"    {sent}")
            elif t == "extra_prediction":
                print(f"  [extra_pred]    '{err['entity']}' ({err['pred_label']})")
                print(f"    {sent}")

    print(f"\n{SEP}\n")
