"""
Serendipity scatter plot, Tufte-style.

Each point is one recommended item: x = relevance, y = unexpectedness.
Serendipity is the product of the two — making it the area swept under the
point from the origin. Quadrant labels show the four corner cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_PURPLE, save_figure


def main():
    np.random.seed(5)
    n = 60
    relevance = np.random.beta(2.0, 2.5, n)
    unexp = np.random.beta(2.0, 2.5, n)

    serendipity = relevance * unexp

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Background contour of constant serendipity = rel * unexp
    g = np.linspace(0.01, 1.0, 200)
    R, U = np.meshgrid(g, g)
    S = R * U
    ax.contourf(R, U, S, levels=8, cmap="Purples", alpha=0.35, zorder=1)

    # Points
    sizes = 80 + serendipity * 250
    ax.scatter(relevance, unexp, s=sizes,
               color=NML_PURPLE, alpha=0.85, zorder=3,
               edgecolors="white", linewidth=1.0)

    # Quadrant labels
    label_props = dict(fontsize=11, color="dimgray", ha="center", va="center")
    ax.text(0.85, 0.92, "high serendipity\n(relevant + surprising)",
            fontsize=12, color="black", ha="center", va="center",
            fontweight="bold")
    ax.text(0.15, 0.92, "surprising\nbut irrelevant",
            **label_props)
    ax.text(0.15, 0.08, "irrelevant\n& obvious",
            **label_props)
    ax.text(0.85, 0.08, "relevant\nbut obvious",
            **label_props)

    # Range frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, 1.0)
    ax.spines["bottom"].set_bounds(0, 1.0)

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel("relevance")
    ax.set_ylabel("unexpectedness")

    fig.tight_layout()
    save_figure(fig, "Serendipity_scatter")


if __name__ == "__main__":
    main()
