"""
R² explained geometrically.

R² = 1 - SS_res / SS_tot. The two terms are sums of squared residuals against
the model (left) and against the mean baseline (right). Each residual becomes
a square whose AREA is the squared error. The ratio of the total purple area
to the total red area is the second term in the R² formula.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import NML_PURPLE, NML_RED, save_figure


def draw_residual_squares(ax, x, y, y_pred, color, alpha=0.45):
    """For each point, draw a square with side = |y - y_pred|."""
    for xi, yi, ypi in zip(x, y, y_pred):
        side = yi - ypi  # signed; square extends below the prediction line
        x0 = xi
        y0 = min(yi, ypi)
        ax.add_patch(Rectangle(
            (x0, y0), abs(side), abs(side),
            facecolor=color, alpha=alpha, edgecolor="none", zorder=2,
        ))


def main():
    np.random.seed(0)
    x = np.array([1.5, 3.0, 5.0, 6.5, 8.0])
    y_true = 1.0 * x + 1.0 + np.array([0.6, -1.0, -0.5, 1.6, 1.0])

    # Model: linear fit.
    slope, intercept = np.polyfit(x, y_true, 1)
    y_pred_model = slope * x + intercept

    # Baseline: predict the mean.
    y_mean = np.full_like(x, y_true.mean())

    ss_res = np.sum((y_true - y_pred_model) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"wspace": 0.18},
    )

    # ---- left: model ----
    ax = axes[0]
    # Regression line spans the data range with a small margin only.
    pad = 0.5
    line_x = np.linspace(x.min() - pad, x.max() + pad, 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color="black", linewidth=1.5, zorder=3)
    draw_residual_squares(ax, x, y_true, y_pred_model, NML_PURPLE)
    ax.scatter(x, y_true, color="black", s=30, zorder=4)
    ax.set_title("model: linear fit", fontsize=13, color=NML_PURPLE, loc="left")
    ax.text(0.4, 11.5, f"$SS_{{res}}$ = sum of purple-square areas = {ss_res:.1f}",
            fontsize=11, color=NML_PURPLE)

    # ---- right: baseline (predict the mean) ----
    ax = axes[1]
    pad = 0.5
    ax.plot([x.min() - pad, x.max() + pad],
            [y_true.mean(), y_true.mean()],
            color="black", linewidth=1.5, zorder=3)
    draw_residual_squares(ax, x, y_true, y_mean, NML_RED)
    ax.scatter(x, y_true, color="black", s=30, zorder=4)
    ax.set_title("baseline: predict the mean", fontsize=13, color=NML_RED, loc="left")
    ax.text(0.4, 11.5, f"$SS_{{tot}}$ = sum of red-square areas = {ss_tot:.1f}",
            fontsize=11, color=NML_RED)

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(-1, 12.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel("$y$", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_bounds(0, 11)
        ax.spines["bottom"].set_bounds(0, 10)
        ax.set_aspect("equal")

    # Bottom-center summary
    fig.text(
        0.5, 0.01,
        f"$R^2 = 1 - {ss_res:.1f}\\,/\\,{ss_tot:.1f} = {r2:.2f}$",
        ha="center", va="bottom", fontsize=15,
    )

    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.12)
    save_figure(fig, "R2_explained")


if __name__ == "__main__":
    main()
