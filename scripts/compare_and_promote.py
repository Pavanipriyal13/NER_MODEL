"""
scripts/compare_and_promote.py
================================
Compare a candidate model against the approved model (models/ner_final)
and decide whether to promote it.

This script NEVER automatically overwrites models/ner_final.
Promotion requires a --promote flag that you must add explicitly.

Usage:
    # Compare only (safe, no changes):
    python scripts/compare_and_promote.py --candidate models/ner_best

    # Compare and promote if checks pass (you must add --promote):
    python scripts/compare_and_promote.py --candidate models/ner_best --promote

PROMOTION RULES (all must pass)
--------------------------------
  1. Candidate micro F1  > approved micro F1  + MIN_MICRO_DELTA (default 0.01)
     AND
     Candidate macro F1  > approved macro F1  - 0.005   (must not regress)
  
  NOTE: Both conditions must hold (AND, not OR).
  A candidate that improves macro F1 but drops micro F1 is REJECTED.

  2. No label may lose more than MAX_LABEL_REGRESSION (default 0.02) in F1.
     If any label regresses by more than this, promotion is BLOCKED (not just
     a warning — it sets verdict to REJECT).

  3. Boundary errors must not increase by more than MAX_BOUNDARY_INCREASE
     (default 2) compared to the approved model.

WHY THESE RULES
---------------
  The previous OR condition allowed a candidate to be promoted if only one
  aggregate metric improved, hiding per-label regressions. These rules require
  genuine overall improvement without sacrificing any individual label.

  MIN_MICRO_DELTA = 0.01 (~1 entity on an 87-entity eval set) is a meaningful
  minimum improvement. The old 0.005 threshold was below one-entity resolution.
"""

import argparse
import sys
from pathlib import Path

import spacy

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from scripts.ner_eval import evaluate_model_on_jsonl, print_results, LABELS

# =============================================================================
# PROMOTION THRESHOLDS
# =============================================================================

APPROVED_MODEL       = PROJ_ROOT / "models" / "ner_final"
EVAL_FILE            = PROJ_ROOT / "data" / "processed" / "realistic_eval.jsonl"

# Candidate must beat approved micro F1 by at least this much.
MIN_MICRO_DELTA      = 0.01

# Candidate macro F1 must not drop below approved macro F1 by more than this.
MAX_MACRO_REGRESSION = 0.005

# No individual label F1 may drop by more than this.
MAX_LABEL_REGRESSION = 0.02

# Boundary errors may not increase by more than this.
MAX_BOUNDARY_INCREASE = 2


# =============================================================================
# HELPERS
# =============================================================================

def load_and_evaluate(model_path, eval_file, model_name):
    """Load a spaCy model and evaluate it on eval_file."""
    print(f"\n  Evaluating: {model_name} ({model_path})")
    nlp     = spacy.load(model_path)
    results = evaluate_model_on_jsonl(nlp, eval_file)
    print_results(results, title=f"Results: {model_name}")
    return results


def check_promotion(approved_results, candidate_results, candidate_path):
    """
    Apply promotion rules.

    Returns (verdict, reasons) where verdict is "ACCEPT" or "REJECT"
    and reasons is a list of strings explaining the decision.
    """
    ov_a = approved_results["overall"]
    ov_c = candidate_results["overall"]
    pl_a = approved_results["per_label"]
    pl_c = candidate_results["per_label"]

    reasons   = []
    reject    = False

    # --- Rule 1: micro F1 improvement ---
    micro_delta = ov_c["micro_f1"] - ov_a["micro_f1"]
    if micro_delta >= MIN_MICRO_DELTA:
        reasons.append(
            f"PASS  Micro F1 improved by {micro_delta:+.4f} "
            f"({ov_a['micro_f1']:.4f} → {ov_c['micro_f1']:.4f}, "
            f"threshold={MIN_MICRO_DELTA})"
        )
    else:
        reject = True
        reasons.append(
            f"FAIL  Micro F1 delta {micro_delta:+.4f} "
            f"(need ≥ +{MIN_MICRO_DELTA}, "
            f"got {ov_a['micro_f1']:.4f} → {ov_c['micro_f1']:.4f})"
        )

    # --- Rule 1b: macro F1 must not regress ---
    macro_delta = ov_c["macro_f1"] - ov_a["macro_f1"]
    if macro_delta >= -MAX_MACRO_REGRESSION:
        reasons.append(
            f"PASS  Macro F1 delta {macro_delta:+.4f} "
            f"(no regression beyond {MAX_MACRO_REGRESSION})"
        )
    else:
        reject = True
        reasons.append(
            f"FAIL  Macro F1 regressed by {macro_delta:.4f} "
            f"(limit={MAX_MACRO_REGRESSION})"
        )

    # --- Rule 2: per-label F1 regressions ---
    for lbl in LABELS:
        lbl_delta = pl_c[lbl]["f1"] - pl_a[lbl]["f1"]
        if lbl_delta < -MAX_LABEL_REGRESSION:
            reject = True
            reasons.append(
                f"FAIL  {lbl} F1 regressed by {lbl_delta:.4f} "
                f"({pl_a[lbl]['f1']:.4f} → {pl_c[lbl]['f1']:.4f}, "
                f"limit={MAX_LABEL_REGRESSION})"
            )
        elif lbl_delta < 0:
            reasons.append(
                f"WARN  {lbl} F1 minor regression {lbl_delta:.4f} "
                f"(within tolerance)"
            )
        else:
            reasons.append(
                f"PASS  {lbl} F1 delta {lbl_delta:+.4f}"
            )

    # --- Rule 3: boundary errors ---
    boundary_delta = ov_c["boundary_errors"] - ov_a["boundary_errors"]
    if boundary_delta > MAX_BOUNDARY_INCREASE:
        reject = True
        reasons.append(
            f"FAIL  Boundary errors increased by {boundary_delta} "
            f"(limit={MAX_BOUNDARY_INCREASE})"
        )
    else:
        reasons.append(
            f"PASS  Boundary errors delta {boundary_delta:+d} "
            f"(limit={MAX_BOUNDARY_INCREASE})"
        )

    verdict = "REJECT" if reject else "ACCEPT"
    return verdict, reasons


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare a candidate model against the approved model."
    )
    parser.add_argument(
        "--candidate", required=True,
        help="Path to the candidate model directory (e.g. models/ner_best)"
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="If set and all checks pass, copy candidate to models/ner_final."
    )
    parser.add_argument(
        "--eval", default=str(EVAL_FILE),
        help="Path to the evaluation JSONL file."
    )
    args = parser.parse_args()

    candidate_path = PROJ_ROOT / args.candidate
    eval_file      = Path(args.eval)

    if not candidate_path.exists():
        print(f"ERROR: Candidate model not found: {candidate_path}")
        sys.exit(1)
    if not APPROVED_MODEL.exists():
        print(f"ERROR: Approved model not found: {APPROVED_MODEL}")
        sys.exit(1)
    if not eval_file.exists():
        print(f"ERROR: Eval file not found: {eval_file}")
        sys.exit(1)

    print(f"\nApproved model : {APPROVED_MODEL}")
    print(f"Candidate      : {candidate_path}")
    print(f"Eval file      : {eval_file}")

    # --- Evaluate both models on the same file with the same logic. ---
    approved_results  = load_and_evaluate(APPROVED_MODEL,  eval_file, "ner_final (approved)")
    candidate_results = load_and_evaluate(candidate_path,  eval_file, f"candidate ({args.candidate})")

    # --- Apply promotion rules. ---
    verdict, reasons = check_promotion(approved_results, candidate_results, candidate_path)

    print("\n" + "═" * 68)
    print("  PROMOTION DECISION")
    print("═" * 68)
    for r in reasons:
        print(f"  {r}")
    print(f"\n  VERDICT: {verdict}")
    print("═" * 68)

    if verdict == "ACCEPT":
        if args.promote:
            print(f"\n  --promote flag set. Copying candidate → {APPROVED_MODEL} ...")
            import shutil
            # Rename the current ner_final to ner_final_backup before replacing.
            backup_path = PROJ_ROOT / "models" / "ner_final_backup"
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(APPROVED_MODEL, backup_path)
            print(f"  Backup saved → {backup_path}")
            shutil.rmtree(APPROVED_MODEL)
            shutil.copytree(candidate_path, APPROVED_MODEL)
            print(f"  Promoted: {candidate_path} → {APPROVED_MODEL}")
        else:
            print(f"\n  Candidate PASSED all checks. To promote, re-run with --promote.")
    else:
        print(f"\n  Candidate REJECTED. models/ner_final is unchanged.")

    sys.exit(0 if verdict == "ACCEPT" else 2)


if __name__ == "__main__":
    main()
