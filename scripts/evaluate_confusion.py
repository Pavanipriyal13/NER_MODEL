"""
scripts/evaluate_confusion.py
==============================
Entity-level confusion analysis for the custom Medical NER model.

Evaluates the current best model (models/ner_final) against the
phase6_eval.jsonl dataset and produces a detailed confusion matrix,
per-label metrics, and entity-level error breakdown.

Usage:
    python scripts/evaluate_confusion.py
    python scripts/evaluate_confusion.py --eval data/processed/phase6_eval.jsonl
    python scripts/evaluate_confusion.py --model models/ner_final --output results.json
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spacy

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PROJ_ROOT   = Path(__file__).resolve().parents[1]
LABELS      = ["DRUG", "DISEASE", "SYMPTOM", "CONDITION"]
# Default evaluation file changed from phase6_eval.jsonl (no longer present)
# to realistic_eval.jsonl, which is the current benchmark.
EVAL_FILE   = PROJ_ROOT / "data" / "processed" / "realistic_eval.jsonl"
MODEL_PATH  = PROJ_ROOT / "models" / "ner_final"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_data(path: Path):
    """Load JSONL evaluation data. Returns list of (text, gold_entities)."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"]
            entities = [
                (e["start"], e["end"], e["label"])
                for e in obj.get("entities", [])
            ]
            examples.append((text, entities))
    return examples


def predict(nlp, text: str):
    """Run model and return list of (start, end, label) tuples."""
    doc = nlp(text)
    return [
        (ent.start_char, ent.end_char, ent.label_)
        for ent in doc.ents
        if ent.label_ in LABELS
    ]


def span_to_key(start, end, text):
    """Return a normalised key for a span."""
    return (start, end)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(nlp, examples):
    """
    Entity-level evaluation with confusion analysis.

    Returns a detailed dict containing:
        - confusion_matrix   : {gold_label: {pred_label: count}}
        - per_label          : {label: {tp, fp, fn, precision, recall, f1}}
        - overall            : {exact, wrong_label, missed, false_positive, boundary_error, ...}
        - errors             : list of individual error records
        - missed_terms       : Counter of missed entity texts
        - fp_terms           : Counter of false-positive texts
        - boundary_terms     : list of boundary error records
    """

    # confusion_matrix[gold][pred] = count
    # gold="MISSED" for false-negatives, pred="FP" for false-positives
    confusion = defaultdict(lambda: defaultdict(int))

    # Per-label TP / FP / FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # Global counters
    total_exact         = 0
    total_wrong_label   = 0
    total_missed        = 0
    total_fp            = 0
    total_boundary_err  = 0

    # Error records for detailed analysis
    errors = []

    # Term-level frequency
    missed_terms    = defaultdict(int)
    fp_terms        = defaultdict(int)
    wrong_label_map = []   # list of {text, gold_label, pred_label}

    for text, gold_ents in examples:
        pred_ents = predict(nlp, text)

        gold_set = {(s, e, l) for s, e, l in gold_ents}
        pred_set = {(s, e, l) for s, e, l in pred_ents}

        # ----- exact matches -----
        for span in gold_set & pred_set:
            s, e, lbl = span
            confusion[lbl][lbl] += 1
            tp[lbl] += 1
            total_exact += 1

        # ----- wrong-label matches (same span, different label) -----
        gold_spans = {(s, e): l for s, e, l in gold_ents}
        pred_spans = {(s, e): l for s, e, l in pred_ents}

        for span_key, gold_lbl in gold_spans.items():
            s, e = span_key
            if span_key in pred_spans:
                pred_lbl = pred_spans[span_key]
                if pred_lbl != gold_lbl:
                    # Span matched exactly but wrong label
                    confusion[gold_lbl][pred_lbl] += 1
                    fn[gold_lbl] += 1
                    fp[pred_lbl] += 1
                    total_wrong_label += 1
                    wrong_label_map.append({
                        "text":     text[s:e],
                        "sentence": text,
                        "gold_label": gold_lbl,
                        "pred_label": pred_lbl,
                        "start": s,
                        "end":   e,
                    })
                    errors.append({
                        "type":       "wrong_label",
                        "sentence":   text,
                        "entity":     text[s:e],
                        "gold_label": gold_lbl,
                        "pred_label": pred_lbl,
                        "start": s,
                        "end":   e,
                    })

        # ----- missed entities (false-negatives) -----
        for s, e, gold_lbl in gold_ents:
            span_key = (s, e)
            if span_key not in pred_spans:
                # Check for boundary error: any pred span that overlaps
                boundary_match = None
                for (ps, pe), pred_lbl in pred_spans.items():
                    if ps < e and pe > s:  # overlapping
                        boundary_match = (ps, pe, pred_lbl)
                        break

                if boundary_match:
                    ps, pe, pred_lbl = boundary_match
                    confusion[gold_lbl][pred_lbl] += 1
                    fn[gold_lbl] += 1
                    fp[pred_lbl] += 1
                    total_boundary_err += 1
                    errors.append({
                        "type":        "boundary_error",
                        "sentence":    text,
                        "gold_entity": text[s:e],
                        "pred_entity": text[ps:pe],
                        "gold_label":  gold_lbl,
                        "pred_label":  pred_lbl,
                        "gold_start":  s,
                        "gold_end":    e,
                        "pred_start":  ps,
                        "pred_end":    pe,
                    })
                else:
                    confusion[gold_lbl]["MISSED"] += 1
                    fn[gold_lbl] += 1
                    total_missed += 1
                    missed_terms[text[s:e]] += 1
                    errors.append({
                        "type":       "missed",
                        "sentence":   text,
                        "entity":     text[s:e],
                        "gold_label": gold_lbl,
                        "start": s,
                        "end":   e,
                    })

        # ----- false positives -----
        for s, e, pred_lbl in pred_ents:
            span_key = (s, e)
            if span_key not in gold_spans:
                # Check for boundary overlap
                overlaps_gold = False
                for (gs, ge) in gold_spans:
                    if gs < e and ge > s:
                        overlaps_gold = True
                        break
                if not overlaps_gold:
                    confusion["FP"][pred_lbl] += 1
                    fp[pred_lbl] += 1
                    total_fp += 1
                    fp_terms[text[s:e]] += 1
                    errors.append({
                        "type":       "false_positive",
                        "sentence":   text,
                        "entity":     text[s:e],
                        "pred_label": pred_lbl,
                        "start": s,
                        "end":   e,
                    })

    # ── per-label metrics ─────────────────────────────────────────────────
    per_label = {}
    total_tp = total_fp_all = total_fn_all = 0
    for lbl in LABELS:
        t = tp[lbl]
        f = fp[lbl]
        n = fn[lbl]
        prec = t / (t + f) if (t + f) > 0 else 0.0
        rec  = t / (t + n) if (t + n) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[lbl] = {
            "tp": t, "fp": f, "fn": n,
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   t + n,
        }
        total_tp      += t
        total_fp_all  += f
        total_fn_all  += n

    # macro F1
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(LABELS)

    # micro F1
    micro_prec = total_tp / (total_tp + total_fp_all) if (total_tp + total_fp_all) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn_all) if (total_tp + total_fn_all) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # ── build plain-dict confusion matrix ─────────────────────────────────
    conf_out = {}
    all_keys = LABELS + ["MISSED", "FP"]
    for gold in LABELS + ["FP"]:
        row = {}
        for pred in LABELS + ["MISSED"]:
            v = confusion[gold][pred]
            if v:
                row[pred] = v
        if row:
            conf_out[gold] = row

    # ── rank confusion pairs ───────────────────────────────────────────────
    pair_counts = []
    for gold_lbl in LABELS:
        for pred_lbl in LABELS:
            if gold_lbl != pred_lbl:
                count = confusion[gold_lbl][pred_lbl]
                if count:
                    pair_counts.append({
                        "gold": gold_lbl,
                        "pred": pred_lbl,
                        "count": count,
                        "description": f"gold={gold_lbl} predicted as {pred_lbl}",
                    })
    pair_counts.sort(key=lambda x: x["count"], reverse=True)

    # ── top missed + FP terms ──────────────────────────────────────────────
    top_missed = sorted(missed_terms.items(), key=lambda x: x[1], reverse=True)[:20]
    top_fp     = sorted(fp_terms.items(),    key=lambda x: x[1], reverse=True)[:20]

    return {
        "confusion_matrix":      conf_out,
        "confusion_pairs_ranked": pair_counts,
        "per_label":             per_label,
        "overall": {
            "exact_matches":    total_exact,
            "wrong_label":      total_wrong_label,
            "missed":           total_missed,
            "false_positives":  total_fp,
            "boundary_errors":  total_boundary_err,
            "macro_f1":         round(macro_f1,   4),
            "micro_f1":         round(micro_f1,   4),
            "micro_precision":  round(micro_prec, 4),
            "micro_recall":     round(micro_rec,  4),
        },
        "top_missed_terms": [{"term": t, "count": c} for t, c in top_missed],
        "top_false_positive_terms": [{"term": t, "count": c} for t, c in top_fp],
        "wrong_label_details": wrong_label_map,
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result):
    print("\n" + "=" * 70)
    print("  ENTITY-LEVEL CONFUSION ANALYSIS")
    print("=" * 70)

    ovr = result["overall"]
    print(f"\n  Exact matches  : {ovr['exact_matches']}")
    print(f"  Wrong label    : {ovr['wrong_label']}")
    print(f"  Missed (FN)    : {ovr['missed']}")
    print(f"  False pos (FP) : {ovr['false_positives']}")
    print(f"  Boundary errs  : {ovr['boundary_errors']}")
    print(f"\n  Micro F1       : {ovr['micro_f1']:.4f}")
    print(f"  Micro Precision: {ovr['micro_precision']:.4f}")
    print(f"  Micro Recall   : {ovr['micro_recall']:.4f}")
    print(f"  Macro F1       : {ovr['macro_f1']:.4f}")

    print("\n" + "-" * 70)
    print("  PER-LABEL METRICS")
    print("-" * 70)
    print(f"  {'Label':<12} {'P':>7} {'R':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Sup':>5}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for lbl, m in result["per_label"].items():
        print(f"  {lbl:<12} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}"
              f" {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['support']:>5}")

    print("\n" + "-" * 70)
    print("  CONFUSION MATRIX  (row = gold label, col = predicted label)")
    print("-" * 70)
    pred_cols = LABELS + ["MISSED"]
    gold_pred_label = "gold\\pred"
    header = f"  {gold_pred_label:<14}" + "".join(f"{c:>12}" for c in pred_cols)
    print(header)
    print("  " + "-" * (14 + 12 * len(pred_cols)))
    cm = result["confusion_matrix"]
    for gold in LABELS:
        row_data = cm.get(gold, {})
        row = f"  {gold:<14}" + "".join(f"{row_data.get(c, 0):>12}" for c in pred_cols)
        print(row)
    # False positives row
    fp_row = cm.get("FP", {})
    if fp_row:
        row = f"  {'FP (none→X)':<14}" + "".join(f"{fp_row.get(c, 0):>12}" for c in pred_cols)
        print(row)

    print("\n" + "-" * 70)
    print("  CONFUSION PAIRS RANKED BY ERROR COUNT")
    print("-" * 70)
    pairs = result["confusion_pairs_ranked"]
    if not pairs:
        print("  No cross-label confusion detected.")
    for p in pairs[:10]:
        print(f"  {p['gold']:>10} → {p['pred']:<10}  {p['count']:>4} errors")

    print("\n" + "-" * 70)
    print("  TOP MISSED ENTITIES")
    print("-" * 70)
    for item in result["top_missed_terms"][:10]:
        print(f"  '{item['term']}'  ×{item['count']}")

    print("\n" + "-" * 70)
    print("  TOP FALSE POSITIVES")
    print("-" * 70)
    for item in result["top_false_positive_terms"][:10]:
        print(f"  '{item['term']}'  ×{item['count']}")

    print("\n" + "-" * 70)
    print("  WRONG-LABEL ERRORS (entity text / gold→pred)")
    print("-" * 70)
    for err in result["wrong_label_details"][:20]:
        print(f"  '{err['text']}'  {err['gold_label']} → {err['pred_label']}")
        print(f"    Sentence: {err['sentence']}")

    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entity-level confusion analysis")
    parser.add_argument("--eval",   default=str(EVAL_FILE),  help="Path to evaluation JSONL")
    parser.add_argument("--model",  default=str(MODEL_PATH), help="Path to spaCy model")
    parser.add_argument("--output", default=None,            help="Optional: save JSON results to file")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    nlp = spacy.load(args.model)

    print(f"Loading eval data: {args.eval}")
    examples = load_eval_data(Path(args.eval))
    print(f"  {len(examples)} evaluation sentences loaded.")

    print("Running evaluation...")
    result = evaluate(nlp, examples)

    print_report(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {out_path}")

    return result


if __name__ == "__main__":
    main()
