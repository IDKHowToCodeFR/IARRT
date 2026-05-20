"""Plotting utilities for IARRT outputs."""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


def _style() -> None:
    sns.set_theme(style="whitegrid", palette="Set2")


def plot_loss_curve(output_dir: str) -> str:
    """Save a simulated loss curve because this prototype does not train."""
    _style()
    epochs = list(range(1, 9))
    loss = [2.4, 2.0, 1.72, 1.49, 1.33, 1.24, 1.18, 1.15]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, loss, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Simulated loss")
    plt.title("Training Loss (Simulated)")
    path = os.path.join(output_dir, "training_loss.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_bleu_comparison(baseline: float, idiom_aware: float, output_dir: str) -> str:
    """Save baseline vs idiom-aware BLEU barplot."""
    _style()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Baseline", "Idiom-aware"], y=[baseline, idiom_aware])
    plt.ylabel("BLEU")
    plt.title("BLEU Score Comparison")
    path = os.path.join(output_dir, "bleu_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_idiom_accuracy(accuracy: float, output_dir: str) -> str:
    """Save idiom accuracy barplot."""
    _style()
    plt.figure(figsize=(5, 4))
    sns.barplot(x=["Idiom-aware"], y=[accuracy])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Idiom Accuracy")
    path = os.path.join(output_dir, "idiom_accuracy.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_gate_histogram(gate_scores: List[float], output_dir: str) -> str:
    """Save histogram of routing gate scores."""
    _style()
    plt.figure(figsize=(6, 4))
    sns.histplot(gate_scores, bins=10, kde=False)
    plt.xlabel("Gate score")
    plt.title("Routing/Gating Scores")
    path = os.path.join(output_dir, "routing_gate_histogram.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_retrieval_heatmap(sample_retrievals: List[List[Dict]], output_dir: str) -> str:
    """Save a heatmap of top retrieval similarities."""
    _style()
    rows = []
    labels = []
    max_len = max((len(items) for items in sample_retrievals), default=0)

    for idx, items in enumerate(sample_retrievals):
        labels.append(f"Query {idx + 1}")
        row = [item["score"] for item in items]
        row += [0.0] * (max_len - len(row))
        rows.append(row)

    if not rows:
        rows = [[0.0]]
        labels = ["No idiom"]

    plt.figure(figsize=(7, 4))
    sns.heatmap(
        rows,
        annot=True,
        fmt=".2f",
        xticklabels=[f"Top {i + 1}" for i in range(len(rows[0]))],
        yticklabels=labels,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
    )
    plt.title("Retrieval Similarity Heatmap")
    path = os.path.join(output_dir, "retrieval_similarity_heatmap.png")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path
