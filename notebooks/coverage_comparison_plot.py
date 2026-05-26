"""
Coverage comparison plot, Tufte-style.

Three systems shown as catalog grids: 100 small squares per system, each
square is one item in the catalog. Filled cyan = recommended at least once;
gray = never recommended. Coverage is just the visible proportion.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import NML_CYAN, save_figure


def main():
    np.random.seed(0)
    catalog_size = 100  # 10x10 grid
    cols = 20
    rows = 5

    systems = [
        ("System A — popular only", 0.12),
        ("System B — moderate spread", 0.48),
        ("System C — broad utilization", 0.89),
    ]

    fig, axes = plt.subplots(
        len(systems), 1, figsize=(10, 5.5),
        gridspec_kw={"hspace": 0.55},
    )

    for ax, (label, coverage) in zip(axes, systems):
        n_recommended = int(round(catalog_size * coverage))
        recommended_idx = set(
            np.random.choice(catalog_size, n_recommended, replace=False)
        )

        for k in range(catalog_size):
            r, c = divmod(k, cols)
            color = NML_CYAN if k in recommended_idx else "lightgray"
            ax.add_patch(Rectangle(
                (c, rows - 1 - r), 0.85, 0.85,
                facecolor=color, edgecolor="none",
            ))

        ax.text(
            -0.5, rows / 2, label,
            ha="right", va="center", fontsize=13,
        )
        ax.text(
            cols + 0.5, rows / 2,
            f"Coverage = {coverage:.2f}",
            ha="left", va="center", fontsize=14,
        )

        ax.set_xlim(-7, cols + 5)
        ax.set_ylim(-0.5, rows + 0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)

    # Caption-style note
    axes[-1].text(
        cols / 2, -0.9,
        "each square = one item in a 100-item catalog",
        ha="center", va="top", fontsize=11,
        color="dimgray", style="italic",
    )

    save_figure(fig, "Coverage_comparison")


if __name__ == "__main__":
    main()
