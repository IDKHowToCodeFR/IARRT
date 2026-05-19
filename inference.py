"""
Interactive inference demo for IARRT.

Run:
    python inference.py

Then enter a German sentence. The script prints detected idioms, retrieved
meanings, baseline mBART translation, and final routed translation.
"""

from models.mbart_wrapper import MBartTranslator
from retrieval.idiom_retrieval import IdiomRetriever
from utils.idiom_detection import detect_idioms
from utils.routing import make_idiom_aware_translation


def main() -> None:
    """Run a small command-line demo."""
    translator = MBartTranslator()
    retriever = IdiomRetriever()

    print("IARRT inference demo. Press Enter on an empty line to quit.")
    while True:
        sentence = input("\nGerman sentence: ").strip()
        if not sentence:
            break

        spans = detect_idioms(sentence)
        retrievals = [retriever.retrieve(idiom, k=3) for idiom, _, _ in spans]
        baseline = translator.translate(sentence)
        final, gate = make_idiom_aware_translation(baseline, spans, retrievals)

        print("\nDetected idioms:")
        if spans:
            for idiom, start, end in spans:
                print(f"- {idiom} (span {start}:{end})")
        else:
            print("- None")

        print("\nRetrieved meanings:")
        if retrievals:
            for items in retrievals:
                top = items[0]
                print(f"- {top['idiom']} -> {top['meaning']} ({top['score']:.3f})")
        else:
            print("- None")

        print(f"\nBaseline translation: {baseline}")
        print(f"Gate score: {gate:.3f}")
        print(f"Final translation: {final}")


if __name__ == "__main__":
    main()
