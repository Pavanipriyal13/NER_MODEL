"""
predict.py
==========
Interactive NER prediction script.

Usage:
    python predict.py                   # interactive mode
    python predict.py --demo            # run built-in demo sentences
    python predict.py --text "..."      # predict a single sentence

Examples:
    python predict.py --demo
    python predict.py --text "Can Metformin be used during pregnancy?"
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.ner.model import NERModel

SUPPORTED_LABELS = {"DRUG", "DISEASE", "SYMPTOM", "CONDITION"}

DEMO_SENTENCES = [
    "Can Metformin be used during pregnancy in a patient with type 2 diabetes?",
    "Metformin + Glimepiride may be used for type 2 diabetes.",
    "The patient has severe headache.",
    "The patient has hepatic impairment.",
    "The patient reports urinary pain and fever.",
    "Frusemide + Spironolactone for heart failure — is it safe during pregnancy?",
    "Can Metformin Hydrochloride (SR) + Glimepiride + Voglibose be used in renal impairment?",
    "What medication is available for hypertension with dizziness?",
    "The patient does not currently report nausea.",
    "What are the available treatment options?",
]


def format_output(text: str, entities: list) -> str:
    """Pretty-print NER results."""
    lines = [f"\nInput:\n  {text}", "\nEntities:"]
    if not entities:
        lines.append("  (no entities found)")
    else:
        for ent in entities:
            lines.append(f"  {ent['text']:<45} → {ent['label']}")
    return "\n".join(lines)


def run_demo(ner: NERModel):
    """Run the built-in demo sentences."""
    print("\n" + "=" * 65)
    print("DEMO — Medical NER predictions")
    print("=" * 65)
    for text in DEMO_SENTENCES:
        entities = ner.predict(text)
        print(format_output(text, entities))
        print()


def run_interactive(ner: NERModel):
    """Interactive loop: user types sentences, model predicts."""
    print("\n" + "=" * 65)
    print("Medical NER — Interactive Mode")
    print("Supported labels: DRUG  DISEASE  SYMPTOM  CONDITION")
    print("Type 'quit' or press Ctrl-C to exit.")
    print("=" * 65)
    while True:
        try:
            text = input("\nInput: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if text.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break
        if not text:
            continue
        entities = ner.predict(text)
        if not entities:
            print("  (no entities found)")
        else:
            for ent in entities:
                print(f"  {ent['text']:<45} → {ent['label']}")


def main():
    parser = argparse.ArgumentParser(description="Medical NER prediction")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--demo",  action="store_true", help="Run demo sentences")
    group.add_argument("--text",  type=str,            help="Predict a single sentence")
    args = parser.parse_args()

    print("Loading model...")
    ner = NERModel()
    print("Model ready.\n")

    if args.demo:
        run_demo(ner)
    elif args.text:
        entities = ner.predict(args.text)
        print(format_output(args.text, entities))
    else:
        run_interactive(ner)


if __name__ == "__main__":
    main()
