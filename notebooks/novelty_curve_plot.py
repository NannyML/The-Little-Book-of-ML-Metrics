"""
Novelty curve plot, Tufte-style.

Novelty as -log_2(P) over item popularity. The steepness near P = 0 is the
key insight: rare items have wildly different novelty scores, mainstream
items all look similar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, save_figure


def main():
    P = np.linspace(0.005, 1.0, 300)
    novelty = -np.log2(P)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(P, novelty, color=NML_CYAN, linewidth=2.4, zorder=2)

    # Annotate two reference points
    for p, label, off in [
        (0.01, "rare item\nP = 0.01 → 6.6 bits", (15, -10)),
        (0.5, "mainstream item\nP = 0.5 → 1 bit", (15, 10)),
    ]:
        n = -np.log2(p)
        ax.scatter([p], [n], color=NML_CYAN, s=50, zorder=3)
        ax.annotate(
            label, xy=(p, n), xytext=off,
            textcoords="offset points",
            fontsize=11, color="black",
            arrowprops=dict(arrowstyle="-", color="dimgray", linewidth=0.7),
        )

    # Half-popularity marker
    ax.axvline(0.5, color="lightgray", linewidth=0.8,
               linestyle="--", zorder=1)

    # Range frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, novelty.max())
    ax.spines["bottom"].set_bounds(0, 1.0)

    ax.set_xlim(-0.04, 1.05)
    ax.set_ylim(-0.3, novelty.max() + 0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 2, 4, 6])

    ax.set_xlabel("item popularity  P(i)")
    ax.set_ylabel("Novelty  ($-\\log_2 P$, bits)")

    fig.tight_layout()
    save_figure(fig, "Novelty_curve")


if __name__ == "__main__":
    main()
