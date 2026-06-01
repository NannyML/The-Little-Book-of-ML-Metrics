"""
EVS vs R² across three model regimes.

Three small multiples — each is one diagnostic case. Together they map out
when EVS adds information beyond R² and when it doesn't:

  1. Good model        — both metrics high.  Trust it.
  2. Biased model      — EVS high, R² low.   Fixable: shift by mean residual.
  3. Noisy model       — both metrics equal and low.  Retrain — pattern wrong.

The middle panel shows the bias-corrected predictions alongside the raw
predictions, making the actionable insight ("EVS = R² after bias correction")
visible directly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_PURPLE, NML_RED, save_figure


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def evs(y, yhat):
    return 1 - np.var(y - yhat) / np.var(y)


def main():
    np.random.seed(42)
    n = 80
    y_true = np.random.randn(n) * 3 + 10

    # Three models.
    y_good   = y_true + np.random.randn(n) * 0.6
    y_biased = y_true + 5 + np.random.randn(n) * 0.6   # constant +5 offset
    y_noisy  = y_true + np.random.randn(n) * 2.5

    # Bias correction for the biased model.
    bias = (y_biased - y_true).mean()
    y_biased_corrected = y_biased - bias

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6),
                             gridspec_kw={"wspace": 0.25})

    lims = (0, 22)

    # ---- Panel 1: Good ----
    ax = axes[0]
    ax.plot(lims, lims, color="lightgray", linewidth=1.0,
            linestyle="--", zorder=1)
    ax.scatter(y_true, y_good, color=NML_CYAN, s=22, alpha=0.85, zorder=3)
    ax.set_title("good model", fontsize=13, color=NML_CYAN, loc="left")
    ax.text(0.5, 20.5,
            f"EVS = {evs(y_true, y_good):.2f}\nR²  = {r2(y_true, y_good):.2f}",
            fontsize=11, color=NML_CYAN, va="top")
    ax.text(0.5, -3.5,
            "no bias — both metrics agree",
            fontsize=10, color="dimgray", style="italic")

    # ---- Panel 2: Biased + corrected ----
    ax = axes[1]
    ax.plot(lims, lims, color="lightgray", linewidth=1.0,
            linestyle="--", zorder=1)
    ax.scatter(y_true, y_biased, color=NML_RED, s=22, alpha=0.85, zorder=3)
    ax.scatter(y_true, y_biased_corrected, color=NML_CYAN,
               s=22, alpha=0.85, zorder=3)
    ax.set_title("biased model", fontsize=13, color=NML_RED, loc="left")

    # Annotations in the bottom-right whitespace, away from both point clouds.
    ax.text(13, 6.5,
            f"EVS = {evs(y_true, y_biased):.2f}",
            fontsize=11, color="black", va="top")
    ax.text(13, 4.5,
            f"R² = {r2(y_true, y_biased):.2f}  raw",
            fontsize=11, color=NML_RED, va="top")
    ax.text(13, 2.5,
            f"R² = {r2(y_true, y_biased_corrected):.2f}  corrected",
            fontsize=11, color=NML_CYAN, va="top")
    ax.text(0.5, -3.5,
            "EVS predicted what bias correction would recover",
            fontsize=10, color="dimgray", style="italic")

    # ---- Panel 3: Noisy ----
    ax = axes[2]
    ax.plot(lims, lims, color="lightgray", linewidth=1.0,
            linestyle="--", zorder=1)
    ax.scatter(y_true, y_noisy, color=NML_PURPLE, s=22, alpha=0.85, zorder=3)
    ax.set_title("noisy model", fontsize=13, color=NML_PURPLE, loc="left")
    ax.text(0.5, 20.5,
            f"EVS = {evs(y_true, y_noisy):.2f}\nR²  = {r2(y_true, y_noisy):.2f}",
            fontsize=11, color=NML_PURPLE, va="top")
    ax.text(0.5, -3.5,
            "both low — no offset can fix this",
            fontsize=10, color="dimgray", style="italic")

    for ax in axes:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("actual")
        if ax is axes[0]:
            ax.set_ylabel("predicted")
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_bounds(0, 22)
        ax.spines["bottom"].set_bounds(0, 22)
        ax.set_aspect("equal")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.18)
    save_figure(fig, "EVS_vs_R2_three_models")


if __name__ == "__main__":
    main()
