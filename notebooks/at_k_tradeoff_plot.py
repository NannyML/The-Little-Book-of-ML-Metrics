"""
Precision@K vs Recall@K trade-off plot for the @K Metrics page.

The figure shows the classic shape: as K grows, Precision@K falls and Recall@K
rises. The crossover point (where the two curves meet) is a useful anchor for
picking K.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_RED, save_figure


def main():
    # Simulate a single query: 20 relevant items in a catalog.
    # Probability that the item at rank k is relevant declines with k.
    np.random.seed(7)

    K_values = np.arange(1, 51)
    total_relevant = 20

    # Hits at each rank: realistic "good" recommender — first few are usually
    # relevant, declining tail.
    is_relevant = np.array(
        [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
         0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    )
    assert is_relevant.sum() == total_relevant, is_relevant.sum()
    cum_relevant = np.cumsum(is_relevant)
    precision_at_k = cum_relevant / K_values
    recall_at_k = cum_relevant / total_relevant

    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.plot(K_values, precision_at_k, color=NML_CYAN, linewidth=2.5,
            label="Precision@K")
    ax.plot(K_values, recall_at_k, color=NML_RED, linewidth=2.5,
            label="Recall@K")

    # Range frame (Tufte): only show axis ranges that contain data.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, 1.0)
    ax.spines["bottom"].set_bounds(1, 50)

    ax.set_xlim(0, 52)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([1, 5, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel("K (size of top-K list)")
    ax.set_ylabel("score")

    # Inline labels at the end of each curve.
    ax.text(50.5, precision_at_k[-1], "Precision@K", color=NML_CYAN,
            va="center", ha="left", fontsize=13)
    ax.text(50.5, recall_at_k[-1], "Recall@K", color=NML_RED,
            va="center", ha="left", fontsize=13)

    # Mark the crossover.
    cross_idx = int(np.argmin(np.abs(precision_at_k - recall_at_k)))
    ax.scatter([K_values[cross_idx]], [precision_at_k[cross_idx]],
               color="black", zorder=5, s=30)
    ax.annotate(
        f"crossover at K={K_values[cross_idx]}",
        xy=(K_values[cross_idx], precision_at_k[cross_idx]),
        xytext=(K_values[cross_idx] + 4, precision_at_k[cross_idx] - 0.18),
        fontsize=12,
        color="black",
        arrowprops=dict(arrowstyle="-", color="black", linewidth=0.8),
    )

    fig.tight_layout()
    save_figure(fig, "at_K_precision_recall")


if __name__ == "__main__":
    main()
