"""
MRR ranked-lists plot for the MRR page.

Small multiples: each row is one query's ranked list of K items. The first
relevant item (filled circle) is what MRR scores; subsequent relevant items
(open circle) are visible but ignored. The reciprocal rank sits on the right;
MRR averages them at the bottom.

This makes the three core facts about MRR visible at once:
  1. RR per query = 1 / rank of the first filled circle.
  2. MRR is the mean of those reciprocals.
  3. Open circles never enter the sum; an empty row contributes 0.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_RED, save_figure


def main():
    K = 10  # show top-10 positions
    # Each tuple: (first_relevant_rank, subsequent_relevant_ranks)
    # None = no relevant item in top-K.
    queries = [
        (1, []),
        (2, [5]),
        (3, [7]),
        (5, [10]),
        (10, []),
        (None, []),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))

    # Heavy/light marker sizes
    SIZE_FIRST = 180
    SIZE_OTHER = 120
    SIZE_IRRELEVANT = 12

    rrs = []
    for row, (first, others) in enumerate(queries):
        y = len(queries) - 1 - row  # top-to-bottom

        # Background dots for every rank (irrelevant by default)
        ax.scatter(
            np.arange(1, K + 1), [y] * K,
            s=SIZE_IRRELEVANT, color="lightgray", zorder=1,
        )

        # Open circles for subsequent relevant items
        if others:
            ax.scatter(
                others, [y] * len(others),
                s=SIZE_OTHER, facecolors="none",
                edgecolors=NML_CYAN, linewidths=2.2, zorder=2,
            )

        # Filled circle for the first relevant item (MRR sees this one)
        if first is not None:
            ax.scatter(
                [first], [y],
                s=SIZE_FIRST, color=NML_CYAN, zorder=3,
            )
            rr = 1.0 / first
        else:
            rr = 0.0
        rrs.append(rr)

        # Query label on the left
        ax.text(
            -0.5, y, f"Q{row+1}",
            ha="right", va="center", fontsize=14,
        )

        # RR value on the right
        rr_text = f"RR = {rr:.2f}" if first is not None else "RR = 0.00  (no hit)"
        ax.text(
            K + 0.8, y, rr_text,
            ha="left", va="center", fontsize=14,
            color="black" if first is not None else NML_RED,
        )

    # Rank header above the grid
    for r in range(1, K + 1):
        ax.text(
            r, len(queries) - 0.3, str(r),
            ha="center", va="bottom", fontsize=12, color="dimgray",
        )
    ax.text(
        (K + 1) / 2, len(queries) + 0.2,
        "rank →",
        ha="center", va="bottom", fontsize=12, color="dimgray",
        style="italic",
    )

    # MRR summary at the bottom
    mrr = float(np.mean(rrs))
    ax.text(
        (K + 1) / 2, -1.6,
        f"MRR = mean of RRs = {mrr:.3f}",
        ha="center", va="center", fontsize=15, fontweight="bold",
    )

    # Legend strip (compact and well-spaced)
    legend_y = -2.7
    ax.scatter([0.6], [legend_y], s=SIZE_FIRST, color=NML_CYAN)
    ax.text(1.0, legend_y, "first relevant (MRR counts this)",
            va="center", fontsize=11)
    ax.scatter([6.6], [legend_y], s=SIZE_OTHER, facecolors="none",
               edgecolors=NML_CYAN, linewidths=2.2)
    ax.text(6.9, legend_y, "later relevant (ignored)",
            va="center", fontsize=11)
    ax.scatter([11.6], [legend_y], s=SIZE_IRRELEVANT, color="lightgray")
    ax.text(11.85, legend_y, "irrelevant",
            va="center", fontsize=11)

    # Strip all axis chartjunk
    ax.set_xlim(-1.5, K + 5)
    ax.set_ylim(-3.3, len(queries) + 0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    save_figure(fig, "MRR_ranked_lists")


if __name__ == "__main__":
    main()
