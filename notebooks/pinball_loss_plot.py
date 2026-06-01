"""
Pinball Loss plot, Tufte-style.

Show the asymmetric V-shape for three quantiles. The slope on each side
reflects the quantile choice: at q=0.75, under-predictions (right side) are
penalized 3x more than over-predictions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_PURPLE, NML_RED, save_figure


def pinball(error, q):
    return np.where(error >= 0, q * error, (q - 1) * error)


def main():
    err = np.linspace(-10, 10, 401)

    fig, ax = plt.subplots(figsize=(9, 4.4))

    for q, color in [
        (0.25, NML_CYAN),
        (0.50, NML_PURPLE),
        (0.75, NML_RED),
    ]:
        loss = pinball(err, q)
        ax.plot(err, loss, color=color, linewidth=2.4, zorder=2)
        # Inline label slightly inset from the curve endpoint so it does not
        # touch the axis spine.
        if q == 0.75:
            ax.text(11, loss[-1], f"q = {q}", va="center", ha="left",
                    color=color, fontsize=12)
        else:
            ax.text(-11, loss[0], f"q = {q}", va="center", ha="right",
                    color=color, fontsize=12)

    # Zero crossings — minimal axes
    ax.axhline(0, color="lightgray", linewidth=0.7, zorder=1)
    ax.axvline(0, color="lightgray", linewidth=0.7, zorder=1)

    # The asymmetry between q values is self-evident from the curve slopes;
    # no separate annotation needed.

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, 8)
    ax.spines["bottom"].set_bounds(-10, 10)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-0.6, 8.5)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([0, 2, 4, 6, 8])

    ax.set_xlabel("$Y - \\hat{Y}$")
    ax.set_ylabel("loss")

    fig.tight_layout()
    save_figure(fig, "Pinball_Loss")


if __name__ == "__main__":
    main()
