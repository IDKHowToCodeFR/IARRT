"""
Formal baseline comparison for IARRT research paper.
Generates results for:
1. Base mBART (Zero-shot)
2. mBART + Context Prompt (RAG)
3. IARRT (Proposed: Post-MT Injection)
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
from utils.routing import apply_post_injection


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
    for de in tqdm(german_sentences, desc="Preprocessing"):
        spans = detect_idioms(de)
        retrievals = [retriever.retrieve(idiom) for idiom, _, _ in spans]
        all_spans.append(spans)
        all_retrievals.append(retrievals)

    results = []
    
    # Pipeline A: Base Zero-Shot (Force No LoRA)
    print("\nRunning Baseline 1: Zero-Shot mBART...")
    translator_base = MBartTranslator(use_lora=False)
    base_hyps = translator_base.translate_batch(german_sentences, contexts=[""] * len(german_sentences))
    results.append(run_eval("Base mBART", base_hyps, references, all_spans, all_retrievals))
    print(f"  Sample Hyp: {base_hyps[0]}")

    # Pipeline B: RAG Prompting (Base model + Context)
    print("\nRunning Baseline 2: RAG Prompting (Base + Context)...")
    # Format context strings for prompting
    all_contexts = []
    for spans, retrievals in zip(all_spans, all_retrievals):
        meanings = [r[0]["meaning"] for r in retrievals if r]
        all_contexts.append("; ".join(meanings))
        
    rag_hyps = translator_base.translate_batch(german_sentences, contexts=all_contexts)
    results.append(run_eval("mBART + RAG (Prompt)", rag_hyps, references, all_spans, all_retrievals))
    print(f"  Sample Hyp: {rag_hyps[0]}")

    # Pipeline C: IARRT (Proposed: Post-MT Injection)
    print("\nRunning Pipeline 3: IARRT (Post-MT Injection)...")
    iarrt_hyps = [
        apply_post_injection(base, spans, ret) 
        for base, spans, ret in zip(base_hyps, all_spans, all_retrievals)
    ]
    results.append(run_eval("IARRT (Ours)", iarrt_hyps, references, all_spans, all_retrievals))
    print(f"  Sample Hyp: {iarrt_hyps[0]}")

    # Display Results Table
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RESEARCH PAPER RESULTS TABLE")
    print("="*50)
    print(results_df.to_markdown(index=False))
    print("="*50)
    
    print("\nDEBUG: Final Translations (IARRT Ours):")
    for i in range(len(iarrt_hyps)):
        print(f"  REF: {references[i]}")
        print(f"  OUR: {iarrt_hyps[i]}")
    
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
