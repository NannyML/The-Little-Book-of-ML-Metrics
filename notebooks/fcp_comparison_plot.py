"""
FCP comparison plot, Tufte-style.

Two scatter panels: predicted rank vs true preference. Concordant pairs sit
along the diagonal; discordant pairs sit off-diagonal. The good ranking is
tight to the diagonal (FCP = 0.90), the poor ranking scatters away from it
(FCP = 0.20).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_RED, save_figure


def main():
    # 5 items: their true ranks are 1..5.
    true_ranks = np.array([1, 2, 3, 4, 5])

    # Good ranking: nearly diagonal (one small inversion between items 2 and 3)
    good_pred = np.array([1, 3, 2, 4, 5])
    # Poor ranking: mostly reversed
    poor_pred = np.array([4, 2, 5, 3, 1])

    def fcp(pred):
        n = len(pred)
        concordant = 0
        comparable = 0
        for i in range(n):
            for j in range(i + 1, n):
                comparable += 1
                # true says i < j (i is more preferred). Concordant if pred agrees.
                if pred[i] < pred[j]:
                    concordant += 1
        return concordant / comparable

    fcp_good = fcp(good_pred)
    fcp_poor = fcp(poor_pred)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6),
                             gridspec_kw={"wspace": 0.3})

    for ax, (label, pred, color, score) in zip(
        axes,
        [
            ("Good ranking", good_pred, NML_CYAN, fcp_good),
            ("Poor ranking", poor_pred, NML_RED, fcp_poor),
        ],
    ):
        # Diagonal reference
        ax.plot([0.5, 5.5], [0.5, 5.5], color="lightgray",
                linewidth=1.0, linestyle="--", zorder=1)

        # Scatter
        ax.scatter(true_ranks, pred, color=color, s=120, zorder=3)

        ax.text(
            0.5, 5.7,
            f"{label} — FCP = {score:.2f}",
            ha="left", va="bottom",
            fontsize=13, color=color,
        )

        ax.set_xlim(0.3, 5.7)
        ax.set_ylim(0.3, 5.7)
        ax.set_xticks(range(1, 6))
        ax.set_yticks(range(1, 6))
        ax.set_xlabel("true preference rank")
        ax.set_ylabel("predicted rank")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_bounds(1, 5)
        ax.spines["bottom"].set_bounds(1, 5)
        ax.set_aspect("equal")

    fig.tight_layout()
    save_figure(fig, "FCP_comparison")


if __name__ == "__main__":
    main()
