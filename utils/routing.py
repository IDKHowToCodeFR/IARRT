"""Simple routing and idiom-aware translation utilities."""

from typing import Dict, List, Tuple


def gate_score(spans: List[Tuple[str, int, int]], retrievals: List[List[Dict]]) -> float:
    """Compute a simple confidence score from detection and retrieval results."""
    if not spans:
        return 0.0

    top_scores = [items[0]["score"] for items in retrievals if items]
    if not top_scores:
        return 0.5

    # Blend "idiom was detected" with the average top retrieval similarity.
    return min(1.0, 0.5 + 0.5 * (sum(top_scores) / len(top_scores)))


def make_idiom_aware_translation(
    spans: List[Tuple[str, int, int]],
    retrievals: List[List[Dict]],
) -> str:
    """
    Format the context for the idiom-aware translator.
    """
    if not spans:
        return ""

    meanings = []
    for items in retrievals:
        if items and items[0]["meaning"] not in meanings:
            meanings.append(items[0]["meaning"])

    if not meanings:
        return ""

    return "; ".join(meanings)
