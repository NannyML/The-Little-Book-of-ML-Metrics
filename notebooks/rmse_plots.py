"""
Three RMSE-related plots, Tufte-style:

1. RMSE_sensitivity_outliers_plot — two-panel scatter showing how a single
   outlier inflates RMSE.
2. RMSE_comparison_MSE — MSE (quadratic) vs RMSE (linear) on the same axes.
3. RMSLE_comparison_MSLE — MSLE vs RMSLE with the asymmetric under-prediction
   penalty visible.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, NML_RED, NML_PURPLE, save_figure


def rmse_outliers_plot():
    np.random.seed(3)
    x = np.array([10, 15, 20, 25, 30, 40, 50], dtype=float)
    y_clean = np.array([3.5, 2.2, 4.2, 7.5, 4.8, 6.6, 7.8])
    y_out = y_clean.copy()
    y_out[-1] = 20.0  # introduce outlier

    def fit_line(xs, ys):
        slope, intercept = np.polyfit(xs, ys, 1)
        return slope, intercept

    def predict(xs, m, b):
        return m * xs + b

    def rmse(ys, yhat):
        return float(np.sqrt(np.mean((ys - yhat) ** 2)))

    m1, b1 = fit_line(x, y_clean)
    m2, b2 = fit_line(x, y_out)
    yp1 = predict(x, m1, b1)
    yp2 = predict(x, m2, b2)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                             gridspec_kw={"wspace": 0.18})

    line_x = np.linspace(x.min() - 2, x.max() + 2, 100)

    for ax, (xs, ys, yp, color, title, rmse_val, has_outlier) in zip(
        axes,
        [
            (x, y_clean, yp1, NML_CYAN,  "without outlier", rmse(y_clean, yp1), False),
            (x, y_out,   yp2, NML_RED,   "with outlier",    rmse(y_out, yp2),   True),
        ],
    ):
        # Fit line
        m, b = fit_line(xs, ys)
        ax.plot(line_x, m * line_x + b, color=color, linewidth=2.0, zorder=2)

        # Residuals as dashed segments
        for xi, yi, ypi in zip(xs, ys, yp):
            ax.plot([xi, xi], [yi, ypi], color=color, linewidth=0.9,
                    linestyle=":", zorder=2, alpha=0.7)

        # Points
        pt_colors = [color] * len(xs)
        if has_outlier:
            pt_colors[-1] = NML_RED  # highlight
        ax.scatter(xs, ys, color=pt_colors, s=55, zorder=4, edgecolors="none")

        ax.text(11, 21.5, title, fontsize=13, color=color)
        ax.text(11, 19, f"RMSE = {rmse_val:.2f}",
                fontsize=12, color=color)

        # Range frame
        ax.set_xlim(8, 53)
        ax.set_ylim(0, 23)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_bounds(0, 20)
        ax.spines["bottom"].set_bounds(10, 50)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    fig.tight_layout()
    save_figure(fig, "RMSE_sensitivity_outliers_plot")
    plt.close(fig)


def rmse_vs_mse_plot():
    err = np.linspace(-10, 10, 401)
    rmse_single = np.abs(err)
    mse_single = err ** 2

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(err, mse_single, color=NML_CYAN, linewidth=2.4)
    ax.plot(err, rmse_single, color=NML_RED, linewidth=2.4)

    # Inline labels
    ax.text(10.3, mse_single[-1], "MSE = error$^2$",
            va="center", ha="left", color=NML_CYAN, fontsize=12)
    ax.text(10.3, rmse_single[-1], "RMSE = |error|",
            va="center", ha="left", color=NML_RED, fontsize=12)

    ax.set_xlim(-13, 18)
    ax.set_ylim(-5, 110)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, 100)
    ax.spines["bottom"].set_bounds(-10, 10)
    ax.set_xlabel("$Y - \\hat{Y}$")
    ax.set_ylabel("error magnitude")

    fig.tight_layout()
    save_figure(fig, "RMSE_comparison_MSE")
    plt.close(fig)


def rmsle_vs_msle_plot():
    # Use single-observation form: MSLE = (log(1+y) - log(1+yhat))^2
    # Plot against signed prediction error at a fixed y=10.
    y_true = 10.0
    delta = np.linspace(-9, 30, 401)
    y_pred = y_true + delta
    sle = (np.log1p(y_true) - np.log1p(y_pred)) ** 2
    rsle = np.sqrt(sle)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(delta, sle, color=NML_CYAN, linewidth=2.4)
    ax.plot(delta, rsle, color=NML_RED, linewidth=2.4)

    # Inline labels at end — show per-observation contribution
    ax.text(30.5, sle[-1], "MSLE", va="center", ha="left",
            color=NML_CYAN, fontsize=12)
    ax.text(30.5, rsle[-1], "RMSLE", va="center", ha="left",
            color=NML_RED, fontsize=12)

    # Single concise annotation above the chart's data region.
    ax.axvline(0, color="lightgray", linewidth=0.7)
    ax.text(
        -12, 3.25,
        "asymmetric: under-prediction ($\\hat{Y} < Y$) penalized more",
        fontsize=10, color="dimgray", ha="left", va="center",
    )

    ax.set_xlim(-13, 37)
    ax.set_ylim(-0.2, max(sle.max(), rsle.max()) * 1.2)
    ax.set_xticks([-5, 0, 10, 20, 30])
    ax.set_xticklabels(["−5", "0", "+10", "+20", "+30"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, sle.max())
    ax.spines["bottom"].set_bounds(-9, 30)
    ax.set_xlabel("$\\hat{Y} - Y$")
    ax.set_ylabel("loss")

    fig.tight_layout()
    save_figure(fig, "RMSLE_comparison_MSLE")
    plt.close(fig)


if __name__ == "__main__":
    rmse_outliers_plot()
    rmse_vs_mse_plot()
    rmsle_vs_msle_plot()
