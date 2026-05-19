"""
Formal baseline comparison for IARRT research paper.
Generates results for:
1. Base mBART (Zero-shot)
2. mBART + Context Prompt (RAG)
3. IARRT (LoRA + RAG)
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm

from data.load_data import download_and_save_subset
from models.mbart_wrapper import MBartTranslator
from retrieval.idiom_retrieval import IdiomRetriever
from utils.evaluation import (
    compute_bleu,
    compute_chrf,
    compute_meteor,
    compute_bertscore,
    compute_litter,
    idiom_accuracy
)
from utils.idiom_detection import detect_idioms
from utils.routing import make_idiom_aware_translation


def run_eval(name, hypotheses, references, spans_list, retrievals_list):
    print(f"\nEvaluating {name}...")
    metrics = {
        "Model": name,
        "BLEU ↑": compute_bleu(references, hypotheses),
        "chrF++ ↑": compute_chrf(references, hypotheses),
        "METEOR ↑": compute_meteor(references, hypotheses),
        "BERTScore ↑": compute_bertscore(references, hypotheses),
        "LitTER ↓": compute_litter(hypotheses, spans_list),
        "Idiom Acc ↑": idiom_accuracy(hypotheses, spans_list, retrievals_list)
    }
    return metrics


def main(eval_size: int = 20, seed: int = 42):
    subset_path = download_and_save_subset(sample_size=4000, seed=seed)
    df = pd.read_csv(subset_path)
    
    # Ensure some idioms are present for evaluation
    from main import add_idiom_demo_rows
    df = add_idiom_demo_rows(df)
    eval_df = df.head(eval_size).copy()
    
    german_sentences = eval_df["de"].tolist()
    references = eval_df["en"].tolist()
    
    retriever = IdiomRetriever()
    
    # 1. Detect and Retrieve
    all_spans = []
    all_retrievals = []
    all_contexts = []
    for de in tqdm(german_sentences, desc="Preprocessing"):
        spans = detect_idioms(de)
        retrievals = [retriever.retrieve(idiom) for idiom, _, _ in spans]
        context = make_idiom_aware_translation(spans, retrievals)
        all_spans.append(spans)
        all_retrievals.append(retrievals)
        all_contexts.append(context)

    results = []

    # Pipeline A: Base Zero-Shot (Force No LoRA)
    print("\nRunning Baseline 1: Zero-Shot mBART...")
    translator_base = MBartTranslator(use_lora=False)
    base_hyps = translator_base.translate_batch(german_sentences, contexts=[""] * eval_size)
    results.append(run_eval("Base mBART", base_hyps, references, all_spans, all_retrievals))

    # Pipeline B: RAG Prompting (Base model + Context)
    print("\nRunning Baseline 2: RAG Prompting (Base + Context)...")
    rag_hyps = translator_base.translate_batch(german_sentences, contexts=all_contexts)
    results.append(run_eval("mBART + RAG (Prompt)", rag_hyps, references, all_spans, all_retrievals))

    # Pipeline C: IARRT (LoRA + RAG)
    print("\nRunning Pipeline 3: IARRT (LoRA + RAG)...")
    translator_lora = MBartTranslator(use_lora=True)
    iarrt_hyps = translator_lora.translate_batch(german_sentences, contexts=all_contexts)
    results.append(run_eval("IARRT (LoRA+RAG)", iarrt_hyps, references, all_spans, all_retrievals))

    # Display Results Table
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RESEARCH PAPER RESULTS TABLE")
    print("="*50)
    print(results_df.to_markdown(index=False))
    print("="*50)
    
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "baseline_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(eval_size=args.eval_size, seed=args.seed)
