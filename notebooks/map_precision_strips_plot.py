"""
MAP good-vs-poor plot, redesigned as running-precision strips.

Two ranked lists with the SAME number of relevant items but in different
positions. At each relevant rank, annotate the running precision — that is
exactly what Average Precision averages. The figure makes the integral
visible: same recall, very different AP.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_RED, save_figure


def main():
    K = 10
    # Same number of relevant items, very different positions.
    good_ranks = [1, 2, 3, 6]
    poor_ranks = [4, 7, 9, 10]

    def running_precisions(ranks):
        return [(r, (i + 1) / r) for i, r in enumerate(ranks)]

    good = running_precisions(good_ranks)
    poor = running_precisions(poor_ranks)
    ap_good = np.mean([p for _, p in good])
    ap_poor = np.mean([p for _, p in poor])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    SIZE_RELEVANT = 200
    SIZE_IRRELEVANT = 14

    rows = [
        ("Good ranking", good_ranks, good, ap_good, 1),
        ("Poor ranking", poor_ranks, poor, ap_poor, 0),
    ]

    for label, ranks, prec_pairs, ap, y in rows:
        # Background dots
        ax.scatter(
            np.arange(1, K + 1), [y] * K,
            s=SIZE_IRRELEVANT, color="lightgray", zorder=1,
        )
        # Relevant items
        ax.scatter(
            ranks, [y] * len(ranks),
            s=SIZE_RELEVANT, color=NML_CYAN, zorder=3,
        )
        # Running-precision annotations below each relevant item
        for r, p in prec_pairs:
            ax.text(
                r, y - 0.28, f"{p:.2f}",
                ha="center", va="top", fontsize=11,
                color="black",
            )
        # Row label on the left
        ax.text(
            -0.3, y, label,
            ha="right", va="center", fontsize=14,
        )
        # AP value on the right
        terms = " + ".join(f"{p:.2f}" for _, p in prec_pairs)
        ax.text(
            K + 0.6, y + 0.15,
            f"AP = mean({terms})",
            ha="left", va="center", fontsize=11, color="dimgray",
        )
        ax.text(
            K + 0.6, y - 0.18,
            f"    = {ap:.2f}",
            ha="left", va="center", fontsize=13,
        )

    # Rank header
    for r in range(1, K + 1):
        ax.text(
            r, 1.55, str(r),
            ha="center", va="bottom", fontsize=11, color="dimgray",
        )
    ax.text(
        (K + 1) / 2, 1.85, "rank →",
        ha="center", va="bottom", fontsize=11, color="dimgray",
        style="italic",
    )

    # Legend strip
    legend_y = -0.95
    ax.scatter([1.0], [legend_y], s=SIZE_RELEVANT, color=NML_CYAN)
    ax.text(1.4, legend_y, "relevant item",
            va="center", fontsize=11)
    ax.scatter([4.5], [legend_y], s=SIZE_IRRELEVANT, color="lightgray")
    ax.text(4.75, legend_y, "irrelevant",
            va="center", fontsize=11)
    ax.text(7.5, legend_y,
            "number below relevant dot = running precision at that rank",
            va="center", fontsize=11, color="dimgray", style="italic")

    # Strip axis chartjunk
    ax.set_xlim(-1.8, K + 5.5)
    ax.set_ylim(-1.4, 2.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    save_figure(fig, "MAP_precision_strips")


if __name__ == "__main__":
    main()
