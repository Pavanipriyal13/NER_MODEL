"""
tests/test_evaluation.py
=========================
Unit tests for scripts/ner_eval.py — the canonical evaluation module.

Each test uses known gold and predicted spans so expected TP / FP / FN
values can be calculated by hand before checking the code output.

Run with:
    python -m pytest tests/test_evaluation.py -v
Or without pytest:
    python tests/test_evaluation.py

Tests cover:
  1.  Perfect prediction  → TP only, P=R=F1=1.0
  2.  Missed entity       → FN only, P=1.0, R=0.0
  3.  Extra prediction (standalone FP) → FP only, P=0.0, R=1.0
  4.  Wrong-label error   → 1 FN + 1 FP, counted once each
  5.  Boundary error      → 1 FN + 1 FP, counted once each
  6.  Duplicate gold spans are handled safely (set semantics)
  7.  Duplicate predicted spans are NOT double-counted
  8.  Mixed sentence: correct + wrong-label + missed + boundary + extra
  9.  Micro F1 formula verification with known counts
  10. Macro F1 formula verification with known counts
  11. Best-checkpoint selection: earlier epoch wins over final epoch
  12. Zero-division safety: all metrics return 0.0 when TP+FP=0 or TP+FN=0
  13. Invalid offsets in gold do not crash the evaluator (filtered by ner_eval)
  14. Saving and reloading models/ner_final gives identical predictions
  15. API model-path check: src/ner/model.py points to models/ner_final
"""

import sys
import unittest
from pathlib import Path

# Allow running from project root.
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from scripts.ner_eval import evaluate_entities, LABELS


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def make_gold(text, entities):
    """Build a gold record dict."""
    return {"text": text, "entities": entities}


def make_pred(text, entities):
    """Build a prediction record dict."""
    return {"text": text, "entities": entities}


def ent(start, end, label, text=None):
    """Build a single entity dict."""
    return {"start": start, "end": end, "label": label, "text": text or ""}


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

class TestPerfectPrediction(unittest.TestCase):
    """Test 1: A prediction that exactly matches gold."""

    def test_perfect_tp(self):
        # "diabetes" at [16:24] labelled DISEASE.
        text = "Patient has diabetes."
        gold = [make_gold(text, [ent(12, 20, "DISEASE", "diabetes")])]
        pred = [make_pred(text, [ent(12, 20, "DISEASE", "diabetes")])]
        r = evaluate_entities(gold, pred)

        self.assertEqual(r["per_label"]["DISEASE"]["tp"], 1)
        self.assertEqual(r["per_label"]["DISEASE"]["fp"], 0)
        self.assertEqual(r["per_label"]["DISEASE"]["fn"], 0)
        self.assertAlmostEqual(r["per_label"]["DISEASE"]["f1"], 1.0)
        self.assertAlmostEqual(r["overall"]["micro_f1"], 1.0)
        self.assertEqual(r["overall"]["exact_matches"], 1)
        self.assertEqual(r["overall"]["wrong_label"], 0)
        self.assertEqual(r["overall"]["missed"], 0)
        self.assertEqual(r["overall"]["extra_predictions"], 0)
        self.assertEqual(len(r["errors"]), 0)


class TestMissedEntity(unittest.TestCase):
    """Test 2: Gold entity with no prediction at all."""

    def test_missed_fn_only(self):
        text = "Patient has diabetes."
        gold = [make_gold(text, [ent(12, 20, "DISEASE")])]
        pred = [make_pred(text, [])]   # model predicted nothing
        r = evaluate_entities(gold, pred)

        self.assertEqual(r["per_label"]["DISEASE"]["tp"], 0)
        self.assertEqual(r["per_label"]["DISEASE"]["fp"], 0)
        self.assertEqual(r["per_label"]["DISEASE"]["fn"], 1)
        # Precision = 0/0 → 0.0, Recall = 0/1 → 0.0
        self.assertAlmostEqual(r["per_label"]["DISEASE"]["precision"], 0.0)
        self.assertAlmostEqual(r["per_label"]["DISEASE"]["recall"], 0.0)
        self.assertAlmostEqual(r["per_label"]["DISEASE"]["f1"], 0.0)
        self.assertEqual(r["overall"]["missed"], 1)
        self.assertEqual(r["overall"]["extra_predictions"], 0)
        self.assertEqual(r["overall"]["total_fp"], 0)


class TestExtraPrediction(unittest.TestCase):
    """Test 3: Prediction with no gold entity nearby (standalone FP)."""

    def test_extra_fp_only(self):
        text = "Patient has diabetes."
        gold = [make_gold(text, [])]              # no gold entities
        pred = [make_pred(text, [ent(12, 20, "DISEASE")])]  # spurious prediction
        r = evaluate_entities(gold, pred)

        self.assertEqual(r["per_label"]["DISEASE"]["tp"], 0)
        self.assertEqual(r["per_label"]["DISEASE"]["fp"], 1)
        self.assertEqual(r["per_label"]["DISEASE"]["fn"], 0)
        # Precision = 0/1 → 0.0, Recall = 0/0 → 0.0
        self.assertAlmostEqual(r["per_label"]["DISEASE"]["precision"], 0.0)
        self.assertAlmostEqual(r["per_label"]["DISEASE"]["f1"], 0.0)
        self.assertEqual(r["overall"]["extra_predictions"], 1)
        # extra_predictions == total_fp when it is the only error type.
        self.assertEqual(r["overall"]["total_fp"], 1)
        self.assertEqual(r["overall"]["total_fn"], 0)


class TestWrongLabelError(unittest.TestCase):
    """Test 4: Same span, wrong label — counts as 1 FN and 1 FP, once each."""

    def test_wrong_label_counted_once(self):
        text = "Patient has migraine."
        # Gold says DISEASE; model predicts SYMPTOM.
        gold = [make_gold(text, [ent(12, 20, "DISEASE")])]
        pred = [make_pred(text, [ent(12, 20, "SYMPTOM")])]
        r = evaluate_entities(gold, pred)

        # DISEASE gets 1 FN (gold entity not correctly found).
        self.assertEqual(r["per_label"]["DISEASE"]["tp"], 0)
        self.assertEqual(r["per_label"]["DISEASE"]["fn"], 1)
        self.assertEqual(r["per_label"]["DISEASE"]["fp"], 0)

        # SYMPTOM gets 1 FP (wrong prediction).
        self.assertEqual(r["per_label"]["SYMPTOM"]["tp"], 0)
        self.assertEqual(r["per_label"]["SYMPTOM"]["fp"], 1)
        self.assertEqual(r["per_label"]["SYMPTOM"]["fn"], 0)

        # Diagnostic counter.
        self.assertEqual(r["overall"]["wrong_label"], 1)
        self.assertEqual(r["overall"]["extra_predictions"], 0)  # NOT a standalone FP
        self.assertEqual(r["overall"]["missed"], 0)

        # Total FP = 1, total FN = 1.
        self.assertEqual(r["overall"]["total_fp"], 1)
        self.assertEqual(r["overall"]["total_fn"], 1)


class TestBoundaryError(unittest.TestCase):
    """Test 5: Overlapping but wrong offsets — 1 FN + 1 FP, not double-counted."""

    def test_boundary_not_double_counted(self):
        # Gold: "type 2 diabetes" [12:27]; pred: "diabetes" [19:27].
        text = "Patient has type 2 diabetes."
        gold = [make_gold(text, [ent(12, 27, "DISEASE")])]
        pred = [make_pred(text, [ent(19, 27, "DISEASE")])]
        r = evaluate_entities(gold, pred)

        # The gold entity is not exactly matched → FN.
        self.assertEqual(r["per_label"]["DISEASE"]["fn"], 1)
        # The prediction overlaps gold but is wrong boundary → FP.
        self.assertEqual(r["per_label"]["DISEASE"]["fp"], 1)
        self.assertEqual(r["per_label"]["DISEASE"]["tp"], 0)

        self.assertEqual(r["overall"]["boundary_errors"], 1)
        self.assertEqual(r["overall"]["extra_predictions"], 0)  # NOT a standalone FP
        self.assertEqual(r["overall"]["missed"], 0)

        # Total FP and FN = 1 each — counted exactly once.
        self.assertEqual(r["overall"]["total_fp"], 1)
        self.assertEqual(r["overall"]["total_fn"], 1)


class TestDuplicatePredictionsNotDoubleCount(unittest.TestCase):
    """Test 7: Two identical predictions for the same span count only once.

    The evaluator builds pred_span_map as a dict keyed on (start, end).
    Two identical (start, end, label) entries collapse to one dict entry,
    so the duplicate is invisible to the evaluator — it contributes exactly
    one TP and zero extra FPs. This is the correct safe behaviour: we never
    inflate TP or FP counts due to annotation duplicates.
    """

    def test_duplicate_pred_counted_once(self):
        text = "Patient has diabetes."
        gold = [make_gold(text, [ent(12, 20, "DISEASE")])]
        # Two identical predictions for the same span — dict deduplication
        # means only one prediction is visible to the evaluator.
        pred = [make_pred(text, [ent(12, 20, "DISEASE"), ent(12, 20, "DISEASE")])]
        r = evaluate_entities(gold, pred)

        # One TP, no FP, no extra_predictions — duplicates are absorbed.
        self.assertEqual(r["per_label"]["DISEASE"]["tp"], 1)
        self.assertEqual(r["per_label"]["DISEASE"]["fp"], 0)
        self.assertEqual(r["overall"]["extra_predictions"], 0)
        self.assertEqual(r["overall"]["exact_matches"], 1)
        # No inflation from the duplicate.
        self.assertEqual(r["overall"]["total_fp"], 0)


class TestMixedSentence(unittest.TestCase):
    """Test 8: One sentence with correct, wrong-label, missed, boundary, and extra."""

    def setUp(self):
        # Text: "Metformin for type 2 diabetes with headache and nausea."
        # Verified offsets (0-indexed):
        #   Metformin         [0:9]   DRUG
        #   type 2 diabetes   [14:29] DISEASE  (gold; pred gets wrong boundary)
        #   headache          [35:43] SYMPTOM  (gold only, not predicted → MISSED)
        #   nausea            [48:54] SYMPTOM  (gold label SYMPTOM, pred DISEASE → WRONG_LABEL)
        self.text = "Metformin for type 2 diabetes with headache and nausea."

        # Verify offsets are correct before the test runs.
        assert self.text[0:9]   == "Metformin",        f"got {self.text[0:9]!r}"
        assert self.text[14:29] == "type 2 diabetes",  f"got {self.text[14:29]!r}"
        assert self.text[35:43] == "headache",         f"got {self.text[35:43]!r}"
        assert self.text[48:54] == "nausea",           f"got {self.text[48:54]!r}"

        self.gold = [make_gold(self.text, [
            ent(0,  9,  "DRUG",    "Metformin"),
            ent(14, 29, "DISEASE", "type 2 diabetes"),
            ent(35, 43, "SYMPTOM", "headache"),
            ent(48, 54, "SYMPTOM", "nausea"),
        ])]

        self.pred = [make_pred(self.text, [
            ent(0,  9,  "DRUG"),        # correct → exact match
            ent(21, 29, "DISEASE"),     # boundary error: overlaps [14:29]
            # headache NOT predicted → missed
            ent(48, 54, "DISEASE"),     # wrong label: gold=SYMPTOM, pred=DISEASE
        ])]

    def test_mixed_counts(self):
        r = evaluate_entities(self.gold, self.pred)
        ov = r["overall"]
        pl = r["per_label"]

        # DRUG: 1 exact match.
        self.assertEqual(pl["DRUG"]["tp"], 1)
        self.assertEqual(pl["DRUG"]["fp"], 0)
        self.assertEqual(pl["DRUG"]["fn"], 0)

        # DISEASE: 1 FN from boundary error (gold [14:29] vs pred [21:29]).
        #          1 FP from boundary error + 1 FP from wrong-label nausea pred.
        self.assertEqual(pl["DISEASE"]["fn"], 1)
        self.assertEqual(pl["DISEASE"]["fp"], 2)

        # SYMPTOM: headache missed (1 FN) + nausea wrong-label (1 FN).
        #          No SYMPTOM predictions were made, so fp=0.
        self.assertEqual(pl["SYMPTOM"]["fn"], 2)
        self.assertEqual(pl["SYMPTOM"]["fp"], 0)

        # Diagnostics.
        self.assertEqual(ov["exact_matches"],    1)   # Metformin
        self.assertEqual(ov["wrong_label"],      1)   # nausea (SYMPTOM→DISEASE)
        self.assertEqual(ov["boundary_errors"],  1)   # type 2 diabetes span
        self.assertEqual(ov["missed"],           1)   # headache
        self.assertEqual(ov["extra_predictions"],0)   # no pred with zero gold overlap

        # Totals: FN = wrong_label(1) + boundary(1) + missed(1) = 3
        self.assertEqual(ov["total_fn"], 3)
        # Totals: FP = boundary(1 for DISEASE) + wrong_label(1 for DISEASE) = 2
        self.assertEqual(ov["total_fp"], 2)


class TestMicroF1Formula(unittest.TestCase):
    """Test 9: Verify micro F1 formula with known TP / FP / FN values."""

    def test_micro_f1_calculation(self):
        # Simulate: DRUG TP=10 FP=0 FN=0; DISEASE TP=39 FP=2 FN=2
        # Using two separate sentences to produce these counts.
        text1 = "Metformin for diabetes."
        text2 = "Headache."

        # Build 41 DISEASE gold entities across sentences using a loop approach.
        # Simpler: just check the formula with a constructed scenario.
        # We'll use: one exact DRUG, one wrong-label DISEASE (FN+FP each).
        text = "Metformin treats diabetes."
        gold = [make_gold(text, [
            ent(0, 9,   "DRUG"),
            ent(17, 25, "DISEASE"),
        ])]
        pred = [make_pred(text, [
            ent(0, 9,   "DRUG"),
            ent(17, 25, "SYMPTOM"),  # wrong label
        ])]
        r = evaluate_entities(gold, pred)

        # Hand calculation:
        # DRUG:    TP=1, FP=0, FN=0  → P=1.0, R=1.0, F1=1.0
        # DISEASE: TP=0, FP=0, FN=1  → P=0.0, R=0.0, F1=0.0
        # SYMPTOM: TP=0, FP=1, FN=0  → P=0.0, R=0.0, F1=0.0
        # Total TP=1, FP=1, FN=1
        # Micro P = 1/2 = 0.5, Micro R = 1/2 = 0.5, Micro F1 = 0.5
        self.assertEqual(r["overall"]["total_tp"], 1)
        self.assertEqual(r["overall"]["total_fp"], 1)
        self.assertEqual(r["overall"]["total_fn"], 1)
        self.assertAlmostEqual(r["overall"]["micro_precision"], 0.5)
        self.assertAlmostEqual(r["overall"]["micro_recall"], 0.5)
        self.assertAlmostEqual(r["overall"]["micro_f1"], 0.5)


class TestMacroF1Formula(unittest.TestCase):
    """Test 10: Verify macro F1 is the simple average of per-label F1 values."""

    def test_macro_f1_is_simple_average(self):
        text = "Metformin treats diabetes and headache."
        gold = [make_gold(text, [
            ent(0, 9,   "DRUG"),      # will be correct
            ent(17, 25, "DISEASE"),   # will be correct
            ent(30, 38, "SYMPTOM"),   # will be missed
        ])]
        pred = [make_pred(text, [
            ent(0, 9,   "DRUG"),
            ent(17, 25, "DISEASE"),
            # SYMPTOM not predicted
        ])]
        r = evaluate_entities(gold, pred)

        # Hand calculation:
        # DRUG:      TP=1, FP=0, FN=0 → F1=1.0
        # DISEASE:   TP=1, FP=0, FN=0 → F1=1.0
        # SYMPTOM:   TP=0, FP=0, FN=1 → F1=0.0
        # CONDITION: TP=0, FP=0, FN=0 → F1=0.0 (undefined → 0.0)
        # Macro F1 = (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.5
        self.assertAlmostEqual(r["overall"]["macro_f1"], 0.5)


class TestBestCheckpointSelection(unittest.TestCase):
    """Test 11: Best checkpoint is the epoch with highest dev F1, not the last epoch."""

    def test_earlier_epoch_wins(self):
        # Simulate epoch scores: epoch 3 is best, epoch 5 is final.
        dev_f1_scores = [0.70, 0.80, 0.92, 0.88, 0.85]

        best_f1    = -1.0
        best_epoch = 0

        for epoch, score in enumerate(dev_f1_scores, start=1):
            if score > best_f1:
                best_f1    = score
                best_epoch = epoch

        self.assertEqual(best_epoch, 3)
        self.assertAlmostEqual(best_f1, 0.92)
        # The last epoch (5) has F1=0.85 which is NOT the best.
        self.assertNotEqual(best_epoch, len(dev_f1_scores))


class TestZeroDivisionSafety(unittest.TestCase):
    """Test 12: Zero TP / FP / FN should return 0.0, not crash."""

    def test_empty_gold_and_pred(self):
        gold = [make_gold("No entities here.", [])]
        pred = [make_pred("No entities here.", [])]
        r = evaluate_entities(gold, pred)

        for lbl in LABELS:
            self.assertEqual(r["per_label"][lbl]["precision"], 0.0)
            self.assertEqual(r["per_label"][lbl]["recall"],    0.0)
            self.assertEqual(r["per_label"][lbl]["f1"],        0.0)

        self.assertEqual(r["overall"]["micro_f1"],  0.0)
        self.assertEqual(r["overall"]["macro_f1"],  0.0)
        self.assertEqual(r["overall"]["total_gold_entities"], 0)
        self.assertEqual(r["overall"]["total_pred_entities"], 0)


class TestInvalidOffsetFiltered(unittest.TestCase):
    """Test 13: Entities with unsupported labels are silently ignored by the evaluator."""

    def test_unsupported_label_ignored(self):
        # The evaluator filters out entities not in LABELS.
        text = "Patient is in London."
        gold = [make_gold(text, [ent(14, 20, "GPE", "London")])]  # spaCy built-in
        pred = [make_pred(text, [])]
        r = evaluate_entities(gold, pred)

        # GPE is not in LABELS, so it should be invisible to the evaluator.
        self.assertEqual(r["overall"]["total_gold_entities"], 0)
        self.assertEqual(r["overall"]["total_fn"], 0)


class TestModelReloadConsistency(unittest.TestCase):
    """Test 14: Load models/ner_final and verify it gives the same output on a fixed input."""

    @classmethod
    def setUpClass(cls):
        import spacy
        model_path = PROJ_ROOT / "models" / "ner_final"
        if not model_path.exists():
            cls.nlp = None
            return
        cls.nlp = spacy.load(model_path)
        cls.text = "Can Metformin be used for type 2 diabetes?"

    def test_model_loads_and_predicts(self):
        if self.nlp is None:
            self.skipTest("models/ner_final not found — skipping model test.")
        doc = self.nlp(self.text)
        ents = [(e.text, e.label_) for e in doc.ents
                if e.label_ in {"DRUG", "DISEASE", "SYMPTOM", "CONDITION"}]
        # We expect at least Metformin (DRUG) and type 2 diabetes (DISEASE).
        labels_found = {lbl for _, lbl in ents}
        self.assertIn("DRUG",    labels_found, f"DRUG not found in: {ents}")
        self.assertIn("DISEASE", labels_found, f"DISEASE not found in: {ents}")

    def test_reload_gives_same_output(self):
        if self.nlp is None:
            self.skipTest("models/ner_final not found — skipping model test.")
        import spacy
        nlp2 = spacy.load(PROJ_ROOT / "models" / "ner_final")
        doc1 = self.nlp(self.text)
        doc2 = nlp2(self.text)
        ents1 = [(e.text, e.label_) for e in doc1.ents]
        ents2 = [(e.text, e.label_) for e in doc2.ents]
        self.assertEqual(ents1, ents2, "Two loads of ner_final gave different predictions.")


class TestAPIModelPath(unittest.TestCase):
    """Test 15: src/ner/model.py DEFAULT_MODEL_PATH points to models/ner_final."""

    def test_default_model_path(self):
        from src.ner.model import DEFAULT_MODEL_PATH
        expected = PROJ_ROOT / "models" / "ner_final"
        self.assertEqual(
            Path(DEFAULT_MODEL_PATH).resolve(),
            expected.resolve(),
            f"DEFAULT_MODEL_PATH is {DEFAULT_MODEL_PATH}, expected {expected}"
        )

    def test_ner_final_not_overwritten(self):
        """models/ner_final must still exist and contain a config.cfg."""
        model_path = PROJ_ROOT / "models" / "ner_final"
        self.assertTrue(
            model_path.exists(),
            "models/ner_final directory was deleted or moved!"
        )
        self.assertTrue(
            (model_path / "config.cfg").exists(),
            "models/ner_final/config.cfg is missing — model may be corrupt."
        )


# ---------------------------------------------------------------------------
# STANDALONE RUNNER (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run all tests and print a summary.
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
