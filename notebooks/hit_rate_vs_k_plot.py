"""
Hit Rate vs K plot, Tufte-style.

The original showed the same data but with grid, full bounding box, and big
markers. This version strips chartjunk and annotates the endpoints inline so
the curve can be read without a legend.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, save_figure


def main():
    K_values = np.array([1, 3, 5, 10, 20, 50])
    hit_rate = np.array([0.15, 0.35, 0.52, 0.72, 0.88, 0.96])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(K_values, hit_rate, color=NML_CYAN, linewidth=2.5, zorder=2)
    ax.scatter(K_values, hit_rate, color=NML_CYAN, s=30, zorder=3)

    # Inline annotations for endpoints + one middle point
    for k, hr, ha, off in [
        (1, 0.15, "left", (4, -6)),
        (10, 0.72, "left", (4, -6)),
        (50, 0.96, "right", (-4, -12)),
    ]:
        ax.annotate(
            f"K={k} → {int(hr*100)}%",
            xy=(k, hr),
            xytext=off,
            textcoords="offset points",
            ha=ha, va="top",
            fontsize=12,
            color="black",
        )

    # Range frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, 1.0)
    ax.spines["bottom"].set_bounds(1, 50)

    ax.set_xlim(-1, 53)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xticks([1, 3, 5, 10, 20, 50])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])

    ax.set_xlabel("K (size of top-K list)")
    ax.set_ylabel("Hit Rate")

    fig.tight_layout()
    save_figure(fig, "Hit_Rate_vs_K")


if __name__ == "__main__":
    main()
