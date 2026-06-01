"""
Tufte-style cross-section plots for the regression chapter.

Each metric has a 3D surface (brand) and a 2D cross-section showing the slice
through error space. The 2D plot is the 1D companion to the 3D — same
mechanism, easier to read.

Cleanup applied to all:
- range frame (spines bound to data)
- no top/right spines
- inline metric labels where useful
- minimal axis decoration
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style import NML_CYAN, save_figure


def _setup_range_frame(ax, xlim, ylim, xtick, ytick, xlabel, ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(ylim[0], ylim[1])
    ax.spines["bottom"].set_bounds(xlim[0], xlim[1])
    ax.set_xlim(xlim[0] - (xlim[1] - xlim[0]) * 0.05,
                xlim[1] + (xlim[1] - xlim[0]) * 0.05)
    ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.05,
                ylim[1] * 1.05)
    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def mae_cross():
    err = np.linspace(-40, 40, 401)
    loss = np.abs(err)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(err, loss, color=NML_CYAN, linewidth=2.4)
    _setup_range_frame(ax, (-40, 40), (0, 40),
                       [-40, -20, 0, 20, 40], [0, 10, 20, 30, 40],
                       "$Y - \\hat{Y}$", "MAE contribution")
    fig.tight_layout()
    save_figure(fig, "MAE_cross_section")
    plt.close(fig)


def mse_cross():
    err = np.linspace(-40, 40, 401)
    loss = err ** 2
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(err, loss, color=NML_CYAN, linewidth=2.4)
    _setup_range_frame(ax, (-40, 40), (0, 1600),
                       [-40, -20, 0, 20, 40], [0, 400, 800, 1200, 1600],
                       "$Y - \\hat{Y}$", "MSE contribution")
    fig.tight_layout()
    save_figure(fig, "MSE_cross_section")
    plt.close(fig)


def msle_cross():
    """MSLE vs predicted value at fixed Y. Vertical dashed line marks Y."""
    y_true = 10.0
    y_pred = np.linspace(0.5, 40, 400)
    loss = (np.log1p(y_true) - np.log1p(y_pred)) ** 2
    y_max = 4
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(y_pred, loss, color=NML_CYAN, linewidth=2.4)
    ax.axvline(y_true, color="lightgray", linewidth=0.8, linestyle="--")
    ax.text(y_true + 0.6, y_max * 0.95,
            f"$Y = {y_true:.0f}$",
            fontsize=11, color="dimgray", ha="left", va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, y_max)
    ax.spines["bottom"].set_bounds(0, 40)
    ax.set_xlim(-2, 42)
    ax.set_ylim(-0.1, y_max + 0.4)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("$\\hat{Y}$")
    ax.set_ylabel("MSLE contribution")
    # Asymmetry annotation — placed in the upper-right clear region.
    ax.text(25, 3.6,
            "asymmetric:\nunder-prediction\ncosts more",
            fontsize=10, color="dimgray", ha="left", va="center")
    fig.tight_layout()
    save_figure(fig, "MSLE_cross_section")
    plt.close(fig)


def mape_cross():
    """MAPE blows up as the true Y approaches zero.

    Three curves at fixed |error|, each in a different NML color so the
    reader can match label → curve unambiguously. Labels placed above each
    curve in clearly empty regions.
    """
    from style import NML_PURPLE, NML_RED
    Y = np.linspace(0.5, 100, 600)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    # Three errors, each its own color. Labels sit above the curve at an
    # x where there is vertical room.
    curves = [
        (10.0, NML_RED,    12,  60),
        (5.0,  NML_PURPLE, 40,  45),
        (1.0,  NML_CYAN,   70,  30),
    ]
    for err, color, label_x, label_y in curves:
        mape = err / Y * 100
        ax.plot(Y, mape, color=color, linewidth=2.0)
        # Place the inline label just above the curve at label_x.
        curve_y = err / label_x * 100
        ax.text(label_x, curve_y + label_y,
                f"|error| = {int(err)}",
                fontsize=11, color=color, va="bottom", ha="left")

    # Blow-up annotation
    ax.annotate(
        "MAPE → ∞ as Y → 0\n(undefined at Y = 0)",
        xy=(1.2, 430),
        xytext=(30, 600),
        fontsize=11, color="dimgray", ha="left",
        arrowprops=dict(arrowstyle="-", color="dimgray", linewidth=0.7),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, 700)
    ax.spines["bottom"].set_bounds(0, 100)

    ax.set_xlim(-2, 125)
    ax.set_ylim(-30, 720)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 200, 400, 600])
    ax.set_xlabel("true value $Y$")
    ax.set_ylabel("MAPE (%)")
    fig.tight_layout()
    save_figure(fig, "MAPE_cross_section")
    plt.close(fig)


def smape_cross():
    """sMAPE vs predicted value at fixed Y. Vertical dashed line marks Y."""
    y_true = 100.0
    y_pred = np.linspace(0.01, 400, 800)
    loss = 200 * np.abs(y_true - y_pred) / (abs(y_true) + np.abs(y_pred))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(y_pred, loss, color=NML_CYAN, linewidth=2.4)
    ax.axvline(y_true, color="lightgray", linewidth=0.8, linestyle="--")
    ax.text(y_true + 6, 60,
            f"$Y = {y_true:.0f}$",
            fontsize=11, color="dimgray", ha="left", va="center")
    ax.axhline(200, color="lightgray", linewidth=0.8, linestyle=":")
    ax.text(280, 195, "capped at 200%",
            fontsize=10, color="dimgray", ha="left", va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, 200)
    ax.spines["bottom"].set_bounds(0, 400)
    ax.set_xlim(-20, 420)
    ax.set_ylim(-10, 215)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlabel("$\\hat{Y}$")
    ax.set_ylabel("sMAPE (%)")
    fig.tight_layout()
    save_figure(fig, "sMAPE_cross_section")
    plt.close(fig)


def mpd_cross():
    """MPD vs predicted value at fixed Y. Vertical dashed line marks Y."""
    y_true = 10.0
    y_pred = np.linspace(1, 30, 400)
    loss = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
    y_max = 30
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(y_pred, loss, color=NML_CYAN, linewidth=2.4)
    ax.axvline(y_true, color="lightgray", linewidth=0.8, linestyle="--")
    ax.text(y_true + 0.5, y_max * 0.95,
            f"$Y = {y_true:.0f}$",
            fontsize=11, color="dimgray", ha="left", va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, y_max)
    ax.spines["bottom"].set_bounds(1, 30)
    ax.set_xlim(-1, 32)
    ax.set_ylim(-1.5, y_max + 1)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("$\\hat{Y}$")
    ax.set_ylabel("Poisson deviance")
    # Asymmetry annotation — placed in the upper-right clear region.
    ax.text(18, 22,
            "asymmetric:\nunder-prediction\ncosts more",
            fontsize=10, color="dimgray", ha="left", va="center")
    fig.tight_layout()
    save_figure(fig, "MPD_cross_section")
    plt.close(fig)


def mgd_cross():
    """MGD vs predicted value at fixed Y. Vertical dashed line marks Y."""
    y_true = 10.0
    y_pred = np.linspace(1, 30, 400)
    loss = 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)
    y_max = 15
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(y_pred, loss, color=NML_CYAN, linewidth=2.4)
    ax.axvline(y_true, color="lightgray", linewidth=0.8, linestyle="--")
    ax.text(y_true + 0.5, y_max * 0.95,
            f"$Y = {y_true:.0f}$",
            fontsize=11, color="dimgray", ha="left", va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_bounds(0, y_max)
    ax.spines["bottom"].set_bounds(1, 30)
    ax.set_xlim(-1, 32)
    ax.set_ylim(-0.5, y_max + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("$\\hat{Y}$")
    ax.set_ylabel("Gamma deviance")
    ax.text(18, 11,
            "asymmetric:\nunder-prediction\ncosts more",
            fontsize=10, color="dimgray", ha="left", va="center")
    fig.tight_layout()
    save_figure(fig, "MGD_cross_section")
    plt.close(fig)


if __name__ == "__main__":
    mae_cross()
    mse_cross()
    msle_cross()
    mape_cross()
    smape_cross()
    mpd_cross()
    mgd_cross()
