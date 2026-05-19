"""
IARRT: Idiom-Aware German-to-English Translation.

This script runs a simple research prototype:
1. Load a small OPUS German-English subset.
2. Generate baseline mBART translations.
3. Detect German idioms with a handcrafted dictionary.
4. Retrieve idiom meanings with SentenceTransformers + FAISS.
5. Route between literal and idiom-aware translation.
6. Evaluate and save visual outputs.
"""

import argparse
import json
import os
import random
from typing import List

import pandas as pd
from tqdm import tqdm

from data.load_data import download_and_save_subset
from models.mbart_wrapper import MBartTranslator
from retrieval.idiom_retrieval import IdiomRetriever
from utils.evaluation import compute_bleu, idiom_accuracy
from utils.idiom_detection import IDIOM_DICT, detect_idioms
from utils.routing import gate_score, apply_post_injection
from visualizations.plot_utils import (
    plot_bleu_comparison,
    plot_gate_histogram,
    plot_idiom_accuracy,
    plot_loss_curve,
    plot_retrieval_heatmap,
)


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")


def add_idiom_demo_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few idiom-rich examples so the routing and plots are visible.

    The OPUS subset may not contain our tiny manual idiom list, so these rows
    make the prototype behavior easy to inspect.
    """
    demo_rows = pd.DataFrame(
        [
            {
                "de": "Ich musste ins kalte Wasser springen.",
                "en": "I had to take the plunge.",
            },
            {
                "de": "Sie liess die Katze aus dem Sack.",
                "en": "She let the cat out of the bag.",
            },
            {
                "de": "Er trifft den Nagel auf den Kopf.",
                "en": "He hits the nail on the head.",
            },
            {
                "de": "Bitte rede nicht um den heissen Brei.",
                "en": "Please do not beat around the bush.",
            },
        ]
    )
    return pd.concat([demo_rows, df], ignore_index=True)


def process_sentences(
    german_sentences: List[str],
    translator: MBartTranslator,
    retriever: IdiomRetriever,
    top_k: int,
) -> dict:
    """Translate, detect idioms, retrieve meanings, and route outputs."""
    # First pass: detect and retrieve context
    contexts = []
    gate_scores = []
    idiom_spans_list = []
    retrieval_results_list = []

    for german in tqdm(german_sentences, desc="Detecting Idioms"):
        spans = detect_idioms(german)
        retrievals = [retriever.retrieve(idiom, k=top_k) for idiom, _, _ in spans]
        score = gate_score(spans, retrievals)

        idiom_spans_list.append(spans)
        retrieval_results_list.append(retrievals)
        gate_scores.append(score)

    # Second pass: batch translate (baseline)
    baseline = translator.translate_batch(german_sentences, contexts=[""] * len(german_sentences), batch_size=4)
    
    # Third pass: Research Upgrade - Post-MT Injection
    final_translations = [
        apply_post_injection(base, spans, ret)
        for base, spans, ret in zip(baseline, idiom_spans_list, retrieval_results_list)
    ]

    return {
        "baseline": baseline,
        "final": final_translations,
        "gates": gate_scores,
        "spans": idiom_spans_list,
        "retrievals": retrieval_results_list,
    }


def main(sample_size: int = 4000, eval_size: int = 80, top_k: int = 3, seed: int = 42):
    """Run the complete IARRT prototype."""
    random.seed(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subset_path = download_and_save_subset(sample_size=sample_size, seed=seed)
    df = pd.read_csv(subset_path)
    df = add_idiom_demo_rows(df)
    eval_df = df.head(eval_size).copy()

    print(f"Loaded {len(df)} rows. Evaluating {len(eval_df)} rows.")
    print(f"Manual idiom dictionary size: {len(IDIOM_DICT)}")

    translator = MBartTranslator()
    retriever = IdiomRetriever()

    results = process_sentences(
        eval_df["de"].tolist(),
        translator=translator,
        retriever=retriever,
        top_k=top_k,
    )

    bleu_baseline = compute_bleu(eval_df["en"].tolist(), results["baseline"])
    bleu_idiom = compute_bleu(eval_df["en"].tolist(), results["final"])
    idiom_acc = idiom_accuracy(
        results["final"],
        results["spans"],
        results["retrievals"],
    )

    # Calculate per-sentence research metrics
    semantic_shifts = []
    confidence_levels = []
    for base, final, gate in zip(results["baseline"], results["final"], results["gates"]):
        # Simple research proxy for 'Semantic Shift': normalized edit distance
        import difflib
        shift = 1.0 - difflib.SequenceMatcher(None, base, final).ratio()
        semantic_shifts.append(shift)
        
        # Categorize confidence
        if gate > 0.8: level = "High (Intervene)"
        elif gate > 0.4: level = "Medium (Verify)"
        else: level = "Low (Baseline Stable)"
        confidence_levels.append(level)

    output_df = eval_df.copy()
    output_df["Detected Idioms"] = [
        "; ".join(idiom for idiom, _, _ in spans) for spans in results["spans"]
    ]
    output_df["Baseline Translation"] = results["baseline"]
    output_df["IARRT Translation (Ours)"] = results["final"]
    output_df["Gating Confidence"] = results["gates"]
    output_df["Confidence Category"] = confidence_levels
    output_df["Semantic Shift"] = semantic_shifts
    
    # Save a more professional CSV
    output_df.to_csv(os.path.join(OUTPUT_DIR, "translation_results.csv"), index=False)

    summary = {
        "sample_size": sample_size,
        "eval_size": len(eval_df),
        "bleu_baseline": bleu_baseline,
        "bleu_idiom_aware": bleu_idiom,
        "idiom_accuracy": idiom_acc,
    }
    with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), "w") as file:
        json.dump(summary, file, indent=2)

    retrieval_samples = []
    for sentence_retrievals in results["retrievals"]:
        retrieval_samples.extend(sentence_retrievals)
        if len(retrieval_samples) >= 5:
            break

    plot_loss_curve(OUTPUT_DIR)
    plot_bleu_comparison(bleu_baseline, bleu_idiom, OUTPUT_DIR)
    plot_idiom_accuracy(idiom_acc, OUTPUT_DIR)
    plot_gate_histogram(results["gates"], OUTPUT_DIR)
    plot_retrieval_heatmap(retrieval_samples, OUTPUT_DIR)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the IARRT prototype.")
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--eval-size", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        sample_size=args.sample_size,
        eval_size=args.eval_size,
        top_k=args.top_k,
        seed=args.seed,
    )
