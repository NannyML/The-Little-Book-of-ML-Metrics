"""
D² explained geometrically — parallels the R² explained plot.

D² = 1 - SAD_res / SAD_tot, where
  SAD_res = sum of absolute residuals against the model
  SAD_tot = sum of absolute deviations from the median baseline.

Bars (height = |residual|) replace R²'s squares (area = residual²). The
median replaces the mean as the baseline because absolute deviations are
minimized by the median.

Includes one outlier so the reader can also see why D² is outlier-robust:
the outlier's bar is just tall — not visually explosive like an R² square
would be.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_PURPLE, NML_RED, save_figure


def main():
    # Same 5 base points as the R² explained plot, with one outlier added
    # at the end. The regression line is fit on the 5 clean points (the same
    # line as the R² explained plot), so the outlier shows up as a clear
    # bad residual rather than dragging the line toward itself.
    np.random.seed(0)
    x_clean = np.array([1.5, 3.0, 5.0, 6.5, 8.0])
    y_clean = 1.0 * x_clean + 1.0 + np.array([0.6, -1.0, -0.5, 1.6, 1.0])

    # Outlier appended.
    x = np.append(x_clean, 4.0)
    y_true = np.append(y_clean, 12.0)

    # Fit on clean points only — same line as R² plot.
    slope, intercept = np.polyfit(x_clean, y_clean, 1)
    y_pred_model = slope * x + intercept
    y_median = float(np.median(y_true))

    sad_res = float(np.sum(np.abs(y_true - y_pred_model)))
    sad_tot = float(np.sum(np.abs(y_true - y_median)))
    d2 = 1 - sad_res / sad_tot

    # R² on the same data, for the side note.
    ss_res = float(np.sum((y_true - y_pred_model) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4),
                             gridspec_kw={"wspace": 0.18})

    pad = 0.5
    line_x = np.linspace(x.min() - pad, x.max() + pad, 100)
    line_y_model = slope * line_x + intercept

    # ---- Left: model ----
    ax = axes[0]
    ax.plot(line_x, line_y_model, color="black", linewidth=1.5, zorder=3)
    for xi, yi, ypi in zip(x, y_true, y_pred_model):
        ax.plot([xi, xi], [ypi, yi],
                color=NML_PURPLE, linewidth=7, alpha=0.65, zorder=2,
                solid_capstyle="butt")
    ax.scatter(x[:-1], y_true[:-1], color="black", s=30, zorder=4)
    ax.scatter([x[-1]], [y_true[-1]], color=NML_RED, s=45, zorder=5)
    ax.set_title("model: linear fit",
                 fontsize=13, color=NML_PURPLE, loc="left")
    ax.text(0.4, 13.3,
            f"$SAD_{{res}}$ = sum of purple bar heights = {sad_res:.1f}",
            fontsize=11, color=NML_PURPLE)

    # ---- Right: baseline (predict the median) ----
    ax = axes[1]
    ax.plot([x.min() - pad, x.max() + pad],
            [y_median, y_median],
            color="black", linewidth=1.5, zorder=3)
    for xi, yi in zip(x, y_true):
        ax.plot([xi, xi], [y_median, yi],
                color=NML_RED, linewidth=7, alpha=0.55, zorder=2,
                solid_capstyle="butt")
    ax.scatter(x[:-1], y_true[:-1], color="black", s=30, zorder=4)
    ax.scatter([x[-1]], [y_true[-1]], color=NML_RED, s=45, zorder=5)
    ax.set_title("baseline: predict the median",
                 fontsize=13, color=NML_RED, loc="left")
    ax.text(0.4, 13.3,
            f"$SAD_{{tot}}$ = sum of red bar heights = {sad_tot:.1f}",
            fontsize=11, color=NML_RED)

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_bounds(0, 13)
        ax.spines["bottom"].set_bounds(0, 10)
        ax.set_aspect("equal")

    fig.text(
        0.5, 0.07,
        f"$D^2 = 1 - {sad_res:.1f}\\,/\\,{sad_tot:.1f} = {d2:.2f}$",
        ha="center", va="bottom", fontsize=15,
    )
    fig.text(
        0.5, 0.015,
        f"on the same data, $R^2 = {r2:.2f}$ — the outlier inflates $SS_{{res}}$ "
        f"quadratically, $SAD_{{res}}$ only linearly",
        ha="center", va="bottom", fontsize=10, color="dimgray",
    )

    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.16)
    save_figure(fig, "D2_abs_comparison")


if __name__ == "__main__":
    main()
