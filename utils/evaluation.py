"""Evaluation helpers for BLEU and idiom accuracy."""

from typing import Dict, List, Tuple

import sacrebleu
import evaluate
import nltk
from bert_score import score as bert_score_fn

# Initialize evaluate metrics
_METEOR = None
try:
    _METEOR = evaluate.load("meteor")
except Exception as e:
    print(f"ERROR: Could not load METEOR: {e}")


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus BLEU using sacreBLEU."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return float(bleu.score)


def compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    """Compute chrF++ score."""
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return float(chrf.score)


from nltk.translate.meteor_score import meteor_score

def compute_meteor(references: List[str], hypotheses: List[str]) -> float:
    """Compute METEOR score using NLTK."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        # NLTK expects tokens or a list of reference tokens
        score = meteor_score([ref.split()], hyp.split())
        scores.append(score)
    return (sum(scores) / len(scores)) * 100 if scores else 0.0


def compute_bertscore(references: List[str], hypotheses: List[str]) -> float:
    """Compute BERTScore (F1)."""
    import logging
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    P, R, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
    return float(F1.mean()) * 100


def compute_litter(
    hypotheses: List[str],
    idiom_spans_list: List[List[Tuple[str, int, int]]],
) -> float:
    """
    Literal Translation Error Rate (LitTER).
    Higher is WORSE. Measures how often literal words of a German idiom appear in English.
    """
    error_count = 0
    total_idioms = 0

    for translation, spans in zip(hypotheses, idiom_spans_list):
        if not spans:
            continue
        total_idioms += 1
        for idiom, _, _ in spans:
            # Heuristic: if more than 2 words of the German idiom appear in English,
            # it's likely a literal translation error (e.g., "cold water" instead of "plunge").
            # This is a simple research proxy for LitTER.
            words = idiom.lower().split()
            found = sum(1 for w in words if w in translation.lower() and len(w) > 3)
            if found >= 1:
                error_count += 1
                break

    return 100.0 * error_count / total_idioms if total_idioms else 0.0


def idiom_accuracy(
    final_translations: List[str],
    idiom_spans_list: List[List[Tuple[str, int, int]]],
    retrieval_results_list: List[List[List[Dict]]],
) -> float:
    """
    Measure how often a detected idiom's retrieved meaning appears in output.

    Sentences with no detected idiom are skipped from the denominator.
    """
    correct = 0
    total = 0

    for translation, spans, retrievals in zip(
        final_translations, idiom_spans_list, retrieval_results_list
    ):
        if not spans:
            continue
        total += 1
        expected = [items[0]["meaning"] for items in retrievals if items]
        if any(meaning.lower() in translation.lower() for meaning in expected):
            correct += 1

    return 100.0 * correct / total if total else 0.0
