"""
scripts/validate_data.py
=========================
Data validation and overlap report for all JSONL dataset files.

Checks performed on each file:
  1. Valid JSONL syntax (each line is parseable JSON).
  2. Required fields present: "text" and "entities".
  3. Each entity has "start", "end", "label" fields.
  4. Character offsets are within the text (0 <= start < end <= len(text)).
  5. Entity label is one of the four supported labels.
  6. Stored entity text (if present) matches text[start:end].
  7. No overlapping entity spans within the same sentence.
  8. No duplicate (start, end, label) spans within the same sentence.
  9. spaCy tokeniser can align the entity offsets (Example.from_dict).

Cross-file overlap checks:
  - Exact duplicate sentences (same text string) across splits.
  - Near-duplicate detection: lowercased + whitespace-normalised text.

Usage:
    python scripts/validate_data.py

    # Check a single file:
    python scripts/validate_data.py --file data/processed/dev.jsonl

Exit code 0 if all checks pass, 1 if any errors found.
"""

import argparse
import json
import sys
from pathlib import Path

import spacy
from spacy.training import Example

PROJ_ROOT     = Path(__file__).resolve().parents[1]
VALID_LABELS  = {"DRUG", "DISEASE", "SYMPTOM", "CONDITION"}

# Default files to check and which ones are "held-out" (must not appear in train).
DEFAULT_FILES = {
    "train (preannotated, minus dev/test/eval)": PROJ_ROOT / "data" / "processed" / "preannotated_sentences.jsonl",
    "dev":          PROJ_ROOT / "data" / "processed" / "dev.jsonl",
    "test":         PROJ_ROOT / "data" / "processed" / "test.jsonl",
    "realistic_eval": PROJ_ROOT / "data" / "processed" / "realistic_eval.jsonl",
}


# =============================================================================
# LOAD
# =============================================================================

def load_jsonl(path):
    """Load a JSONL file. Returns (records, n_syntax_errors)."""
    records      = []
    syntax_errors = 0
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [SYNTAX ERROR] {path.name} line {line_num}: {exc}")
                syntax_errors += 1
    return records, syntax_errors


# =============================================================================
# PER-FILE VALIDATION
# =============================================================================

def validate_file(path, nlp=None):
    """
    Validate all records in a JSONL file.

    Parameters
    ----------
    path : Path
    nlp  : optional spaCy model for alignment check; uses en_core_web_sm if None.

    Returns a dict with counts and a list of problem descriptions.
    """
    if not path.exists():
        return {"file": str(path), "exists": False}

    records, syntax_errors = load_jsonl(path)

    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "senter"])

    total           = len(records)
    invalid_records = 0
    problems        = []   # (record_index, description)

    n_offset_errors    = 0
    n_label_errors     = 0
    n_text_mismatches  = 0
    n_overlap_errors   = 0
    n_duplicate_spans  = 0
    n_alignment_errors = 0

    for i, rec in enumerate(records):
        text     = rec.get("text", "")
        entities = rec.get("entities", [])
        rec_problems = []

        if not isinstance(text, str) or not text.strip():
            rec_problems.append("empty or missing 'text' field")

        for j, ent in enumerate(entities):
            # --- Required fields ---
            if not all(k in ent for k in ("start", "end", "label")):
                rec_problems.append(f"entity[{j}] missing required fields")
                continue

            s, e, lbl = ent["start"], ent["end"], ent["label"]

            # --- Offset bounds ---
            if not isinstance(s, int) or not isinstance(e, int) or s < 0 or e > len(text) or e <= s:
                n_offset_errors += 1
                rec_problems.append(
                    f"entity[{j}] bad offset [{s}:{e}] (text len={len(text)})"
                )
                continue

            # --- Label check ---
            if lbl not in VALID_LABELS:
                n_label_errors += 1
                rec_problems.append(f"entity[{j}] unknown label '{lbl}'")

            # --- Text/span mismatch ---
            stored = ent.get("text", "")
            if stored and text[s:e] != stored:
                n_text_mismatches += 1
                rec_problems.append(
                    f"entity[{j}] text mismatch: stored='{stored}' actual='{text[s:e]}'"
                )

        # --- Overlap check ---
        spans = [(ent["start"], ent["end"]) for ent in entities
                 if "start" in ent and "end" in ent]
        for a, (as_, ae) in enumerate(spans):
            for b, (bs, be) in enumerate(spans):
                if b <= a:
                    continue
                if as_ < be and bs < ae:
                    n_overlap_errors += 1
                    rec_problems.append(
                        f"entities[{a}] and [{b}] overlap ([{as_}:{ae}] vs [{bs}:{be}])"
                    )

        # --- Duplicate spans ---
        seen = set()
        for j, ent in enumerate(entities):
            if "start" not in ent:
                continue
            key = (ent["start"], ent["end"], ent.get("label", ""))
            if key in seen:
                n_duplicate_spans += 1
                rec_problems.append(f"entity[{j}] duplicate span {key}")
            seen.add(key)

        # --- spaCy alignment check (only if no other problems) ---
        if not rec_problems:
            try:
                doc   = nlp.make_doc(text)
                ents  = [(ent["start"], ent["end"], ent["label"]) for ent in entities]
                Example.from_dict(doc, {"entities": ents})
            except Exception as exc:
                n_alignment_errors += 1
                rec_problems.append(f"spaCy alignment error: {exc}")

        if rec_problems:
            invalid_records += 1
            problems.append({"index": i, "text_preview": text[:60], "issues": rec_problems})

    return {
        "file":              str(path),
        "exists":            True,
        "total_records":     total,
        "syntax_errors":     syntax_errors,
        "invalid_records":   invalid_records,
        "offset_errors":     n_offset_errors,
        "label_errors":      n_label_errors,
        "text_mismatches":   n_text_mismatches,
        "overlap_errors":    n_overlap_errors,
        "duplicate_spans":   n_duplicate_spans,
        "alignment_errors":  n_alignment_errors,
        "problems":          problems,
        "records":           records,   # kept for overlap analysis
    }


# =============================================================================
# CROSS-FILE OVERLAP CHECK
# =============================================================================

def normalise_text(text):
    """Lower-case and collapse whitespace for near-duplicate detection."""
    return " ".join(text.lower().split())


def check_overlaps(file_results):
    """
    Report exact and near-duplicate sentence overlaps across files.

    Returns a list of overlap descriptions.
    """
    overlaps = []

    # Build exact and normalised text sets per file.
    exact_sets = {}
    norm_sets  = {}
    for name, result in file_results.items():
        if not result.get("exists"):
            continue
        exact_sets[name] = {rec["text"] for rec in result["records"]}
        norm_sets[name]  = {normalise_text(rec["text"]): rec["text"]
                            for rec in result["records"]}

    names = [n for n in file_results if file_results[n].get("exists")]

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            # Exact overlaps.
            exact_common = exact_sets[name_a] & exact_sets[name_b]
            if exact_common:
                overlaps.append(
                    f"EXACT OVERLAP: {len(exact_common)} sentence(s) appear in "
                    f"both '{name_a}' and '{name_b}'"
                )
                for t in list(exact_common)[:3]:
                    overlaps.append(f"  Example: {t[:70]}")

            # Near-duplicate overlaps (normalised text matches but not exact).
            norm_common = set(norm_sets[name_a]) & set(norm_sets[name_b])
            near_dupes = [
                k for k in norm_common
                if norm_sets[name_a][k] != norm_sets[name_b][k]
            ]
            if near_dupes:
                overlaps.append(
                    f"NEAR-DUPLICATE OVERLAP: {len(near_dupes)} sentence(s) are "
                    f"near-duplicates between '{name_a}' and '{name_b}'"
                )
                for k in near_dupes[:2]:
                    overlaps.append(f"  A: {norm_sets[name_a][k][:60]}")
                    overlaps.append(f"  B: {norm_sets[name_b][k][:60]}")

    return overlaps


# =============================================================================
# REPORT PRINTER
# =============================================================================

def print_validation_report(file_results, overlaps):
    SEP = "─" * 68
    print(f"\n{SEP}")
    print("  DATA VALIDATION REPORT")
    print(SEP)

    any_errors = False

    for name, result in file_results.items():
        print(f"\n  [{name}]")
        if not result.get("exists"):
            print(f"    FILE NOT FOUND: {result['file']}")
            continue

        r = result
        status = "PASS" if r["invalid_records"] == 0 and r["syntax_errors"] == 0 else "FAIL"
        if status == "FAIL":
            any_errors = True

        print(f"    File            : {Path(r['file']).name}")
        print(f"    Total records   : {r['total_records']}")
        print(f"    Status          : {status}")
        if r["syntax_errors"]:
            print(f"    Syntax errors   : {r['syntax_errors']}")
        if r["invalid_records"]:
            print(f"    Invalid records : {r['invalid_records']}")
            print(f"      Offset errors   : {r['offset_errors']}")
            print(f"      Label errors    : {r['label_errors']}")
            print(f"      Text mismatches : {r['text_mismatches']}")
            print(f"      Overlap errors  : {r['overlap_errors']}")
            print(f"      Duplicate spans : {r['duplicate_spans']}")
            print(f"      Alignment errors: {r['alignment_errors']}")
            # Print first 5 problems.
            for p in r["problems"][:5]:
                print(f"      Record {p['index']}: {p['text_preview']}")
                for issue in p["issues"]:
                    print(f"        → {issue}")
            if len(r["problems"]) > 5:
                print(f"      ... and {len(r['problems']) - 5} more")

    print(f"\n  CROSS-FILE OVERLAP ANALYSIS")
    print(f"  {SEP}")
    if overlaps:
        for msg in overlaps:
            print(f"  {msg}")
        any_errors = True
    else:
        print("  No unexpected overlaps found.")

    print(f"\n  OVERALL STATUS: {'FAIL — see details above' if any_errors else 'PASS'}")
    print(f"{SEP}\n")
    return any_errors


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate NER dataset files.")
    parser.add_argument(
        "--file", default=None,
        help="Validate a single file instead of all default files."
    )
    args = parser.parse_args()

    # Load spaCy once (expensive); pass it to each validator.
    print("Loading spaCy model for alignment checks ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "senter"])

    if args.file:
        path = Path(args.file)
        result = validate_file(path, nlp)
        file_results = {path.name: result}
        overlaps = []
    else:
        file_results = {}
        for name, path in DEFAULT_FILES.items():
            print(f"Validating {name} ...")
            file_results[name] = validate_file(path, nlp)
        overlaps = check_overlaps(file_results)

    any_errors = print_validation_report(file_results, overlaps)
    sys.exit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
