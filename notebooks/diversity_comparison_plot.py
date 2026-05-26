"""
Diversity comparison plot, Tufte-style.

Two recommendation lists shown as horizontal strips of category-colored
rectangles. High diversity = many different colors; low diversity = a single
block of one color. ILD scores annotated on the right.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import save_figure


# Distinct categorical palette — Tableau-like, readable.
GENRE_COLORS = {
    "Action":  "#4E79A7",
    "Comedy":  "#F28E2B",
    "Drama":   "#E15759",
    "Sci-Fi":  "#76B7B2",
    "Horror":  "#59A14F",
    "Romance": "#B07AA1",
}


def ild(genres):
    """Category-distance diversity: avg pairwise (1 if different category)."""
    n = len(genres)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 0 if genres[i] == genres[j] else 1
            count += 1
    return total / count if count else 0.0


def main():
    high = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance"]
    low = ["Action"] * 6

    ild_high = ild(high)
    ild_low = ild(low)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    def draw_strip(y, genres):
        for i, g in enumerate(genres):
            ax.add_patch(Rectangle(
                (i, y), 1, 1,
                facecolor=GENRE_COLORS[g],
                edgecolor="white", linewidth=1.5,
            ))
            ax.text(
                i + 0.5, y + 0.5, g[0],
                ha="center", va="center",
                fontsize=14, color="white", fontweight="bold",
            )

    draw_strip(1.4, high)
    draw_strip(0.0, low)

    # Row labels
    ax.text(-0.3, 1.9, "High diversity",
            ha="right", va="center", fontsize=14)
    ax.text(-0.3, 0.5, "Low diversity",
            ha="right", va="center", fontsize=14)

    # ILD annotations on right
    ax.text(6.3, 1.9, f"ILD = {ild_high:.2f}",
            ha="left", va="center", fontsize=13)
    ax.text(6.3, 0.5, f"ILD = {ild_low:.2f}",
            ha="left", va="center", fontsize=13)

    # Category legend underneath
    legend_y = -1.0
    keys = list(GENRE_COLORS.keys())
    for i, g in enumerate(keys):
        x = 0.4 + i * 1.6
        ax.add_patch(Rectangle(
            (x, legend_y), 0.4, 0.4,
            facecolor=GENRE_COLORS[g], edgecolor="none",
        ))
        ax.text(x + 0.55, legend_y + 0.2, g,
                va="center", fontsize=11)

    ax.set_xlim(-2.6, 10)
    ax.set_ylim(-1.5, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")

    fig.tight_layout()
    save_figure(fig, "Diversity_comparison")


if __name__ == "__main__":
    main()
