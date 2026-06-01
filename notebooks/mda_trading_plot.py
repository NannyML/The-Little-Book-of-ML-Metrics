"""
MDA direction confusion matrix.

MDA collapses each timestep into a binary direction: up or down. So the
metric is really an accuracy on a 2x2 confusion matrix: predicted-direction
× actual-direction. Showing it that way makes the structure obvious and
separates the "got direction right" signal from anything magnitude-based.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import NML_CYAN, NML_RED, save_figure


def main():
    np.random.seed(11)
    n = 20
    prices = 100 + np.cumsum(np.random.randn(n) * 2)

    pred = prices + np.random.randn(n) * 5
    for i in range(1, n):
        if np.sign(prices[i] - prices[i - 1]) != np.sign(pred[i] - prices[i - 1]):
            if np.random.rand() < 0.8:
                pred[i] = prices[i - 1] + np.sign(prices[i] - prices[i - 1]) * abs(
                    np.random.randn() * 3
                )

    actual_dir = np.sign(np.diff(prices))
    pred_dir = np.sign(np.diff(pred))

    # Build 2x2 counts: rows = actual, cols = predicted.
    up_up   = int(np.sum((actual_dir > 0) & (pred_dir > 0)))
    up_dn   = int(np.sum((actual_dir > 0) & (pred_dir < 0)))
    dn_up   = int(np.sum((actual_dir < 0) & (pred_dir > 0)))
    dn_dn   = int(np.sum((actual_dir < 0) & (pred_dir < 0)))

    total = up_up + up_dn + dn_up + dn_dn
    mda = (up_up + dn_dn) / total

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    # Cell positions: (col, row) with (0,0) bottom-left.
    cells = {
        (0, 1): ("↑ ↑", up_up,  NML_CYAN, "correct"),
        (1, 1): ("↑ ↓", up_dn,  NML_RED,  "missed up"),
        (0, 0): ("↓ ↑", dn_up,  NML_RED,  "missed down"),
        (1, 0): ("↓ ↓", dn_dn,  NML_CYAN, "correct"),
    }

    for (c, r), (arrows, count, color, note) in cells.items():
        alpha = 0.18 if count > 0 else 0.08
        ax.add_patch(Rectangle(
            (c, r), 1, 1, facecolor=color, alpha=alpha,
            edgecolor=color, linewidth=2,
        ))
        # Count in the center, large
        ax.text(c + 0.5, r + 0.6, str(count),
                ha="center", va="center",
                fontsize=36, color=color, fontweight="bold")
        # Note below
        ax.text(c + 0.5, r + 0.22, note,
                ha="center", va="center",
                fontsize=11, color="dimgray")
        # Arrow pair in top corner
        ax.text(c + 0.5, r + 0.88, arrows,
                ha="center", va="center", fontsize=13,
                color="dimgray")

    # Axis labels (use text, not native axis labels — cleaner).
    ax.text(-0.18, 1.5, "actual ↑", ha="right", va="center", fontsize=13)
    ax.text(-0.18, 0.5, "actual ↓", ha="right", va="center", fontsize=13)
    ax.text(0.5, 2.15, "predicted ↑", ha="center", va="center", fontsize=13)
    ax.text(1.5, 2.15, "predicted ↓", ha="center", va="center", fontsize=13)

    # MDA summary
    ax.text(
        1.0, -0.32,
        f"MDA = ({up_up} + {dn_dn}) / {total} = {mda:.2f}",
        ha="center", va="center", fontsize=14,
    )

    ax.set_xlim(-1.4, 2.3)
    ax.set_ylim(-0.6, 2.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)

    fig.tight_layout()
    save_figure(fig, "MDA_trading_example")


if __name__ == "__main__":
    main()
