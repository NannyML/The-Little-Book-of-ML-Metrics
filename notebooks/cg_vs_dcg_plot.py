"""
CG vs DCG plot, Tufte-style.

Both curves score the same ranked list of items. CG just sums; DCG sums after
applying the log discount. The shaded area between them is exactly the cost of
ignoring position — what CG would have counted but DCG discounts away.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_RED, save_figure


def main():
    # Same ranked list of relevance scores. Hand-picked so the gap is readable.
    relevances = np.array([3, 2, 3, 0, 1, 2, 0, 0, 1, 0])
    K = len(relevances)
    positions = np.arange(1, K + 1)
    discount = 1.0 / np.log2(positions + 1)

    cg = np.cumsum(relevances)
    dcg = np.cumsum(relevances * discount)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    # Shaded gap = discount cost
    ax.fill_between(
        positions, dcg, cg,
        color=NML_RED, alpha=0.10, zorder=1,
    )

    ax.plot(positions, cg, color=NML_RED, linewidth=2.4, zorder=3)
    ax.plot(positions, dcg, color=NML_CYAN, linewidth=2.4, zorder=3)

    # Inline labels at the right edge
    ax.text(
        K + 0.2, cg[-1], f"CG = {cg[-1]:.1f}",
        va="center", ha="left", fontsize=13, color=NML_RED,
    )
    ax.text(
        K + 0.2, dcg[-1], f"DCG = {dcg[-1]:.2f}",
        va="center", ha="left", fontsize=13, color=NML_CYAN,
    )

    # Annotation for the gap
    mid = K // 2
    ax.annotate(
        "shaded area = relevance\nlost to position discount",
        xy=(mid + 0.5, (cg[mid] + dcg[mid]) / 2),
        xytext=(mid - 3, cg[-1] - 1.5),
        fontsize=11,
        color="dimgray",
        ha="left",
        arrowprops=dict(arrowstyle="-", color="dimgray", linewidth=0.7),
    )

    # Range frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, cg[-1])
    ax.spines["bottom"].set_bounds(1, K)

    ax.set_xlim(0.5, K + 3)
    ax.set_ylim(-0.3, cg[-1] + 1)
    ax.set_xticks(positions)
    ax.set_yticks([0, 3, 6, 9, 12])

    ax.set_xlabel("rank (K)")
    ax.set_ylabel("cumulative score")

    fig.tight_layout()
    save_figure(fig, "CG_vs_DCG")


if __name__ == "__main__":
    main()
