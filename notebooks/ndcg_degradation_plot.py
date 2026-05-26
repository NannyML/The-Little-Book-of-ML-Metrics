"""
nDCG degradation under random swaps, Tufte-style.

Apply random adjacent-pair swaps to a perfectly-ranked list and recompute
nDCG. The descent is noisy — most swaps barely move the score, occasional
top-of-list swaps drop it sharply. The shape is what generalizes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, save_figure


def ndcg(rels):
    rels = np.asarray(rels, dtype=float)
    positions = np.arange(1, len(rels) + 1)
    dcg = np.sum(rels / np.log2(positions + 1))
    ideal = np.sort(rels)[::-1]
    idcg = np.sum(ideal / np.log2(positions + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    np.random.seed(7)
    # Graded relevance scores; sorted descending = perfect ranking.
    rels = np.array([5, 4, 3, 3, 2, 2, 1, 1, 0, 0], dtype=float)
    n_swaps = 20

    scores = [ndcg(rels)]
    current = rels.copy()
    for _ in range(n_swaps):
        i, j = np.random.choice(len(current), size=2, replace=False)
        current[i], current[j] = current[j], current[i]
        scores.append(ndcg(current))

    swaps = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    ax.plot(swaps, scores, color=NML_CYAN, linewidth=2.2, zorder=2)
    ax.scatter(swaps, scores, color=NML_CYAN, s=22, zorder=3)

    # Annotate the largest drop
    deltas = np.diff(scores)
    worst_step = int(np.argmin(deltas)) + 1
    ax.annotate(
        "biggest single drop:\na rank-1 item demoted",
        xy=(worst_step, scores[worst_step]),
        xytext=(worst_step + 1.5, scores[worst_step] - 0.18),
        fontsize=11,
        color="dimgray",
        ha="left",
        arrowprops=dict(arrowstyle="-", color="dimgray", linewidth=0.7),
    )

    ax.axhline(1.0, color="lightgray", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(
        n_swaps + 0.1, 1.0, "perfect ranking",
        va="center", ha="left", fontsize=10, color="dimgray",
    )

    # Range frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0.4, 1.0)
    ax.spines["bottom"].set_bounds(0, n_swaps)

    ax.set_xlim(-1, n_swaps + 7)
    ax.set_ylim(0.35, 1.06)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])

    ax.set_xlabel("number of random pair swaps")
    ax.set_ylabel("nDCG")

    fig.tight_layout()
    save_figure(fig, "nDCG_degradation")


if __name__ == "__main__":
    main()
