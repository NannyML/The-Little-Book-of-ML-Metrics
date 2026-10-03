"""RMSE and RMSLE figures for the Regression chapter.

RMSE_sensitivity_outliers_plot: one observation is moved and the line refitted.
RMSLE_comparison_MSLE: one-observation MSLE and RMSLE as the prediction varies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from style import *  # noqa: F401,F403

X = np.array([10, 15, 20, 25, 30, 40, 50], dtype=float)
Y_CLEAN = np.array([3.5, 2.2, 4.2, 7.5, 4.8, 6.6, 7.8])


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def rmse_outliers_plot():
    y_out = Y_CLEAN.copy()
    y_out[-1] = 20.0                      # the one changed observation
    fig, axes = book_figure(1.0, 1.95, 1, 2, sharey=True)
    line_x = np.linspace(8, 52, 50)
    panels = [(Y_CLEAN, NML_CYAN, 'original data'), (y_out, NML_RED, 'last point moved')]
    for ax, (ys, color, title) in zip(axes, panels):
        m, b = np.polyfit(X, ys, 1)
        yhat = m * X + b
        ax.plot(line_x, m * line_x + b, color=color, lw=LW, zorder=2)
        for xi, yi, yp in zip(X, ys, yhat):
            ax.plot([xi, xi], [yi, yp], color=color, lw=LW_THIN, ls=(0, (1.5, 1.5)), zorder=2)
        ax.scatter(X, ys, s=14, color=INK, zorder=4, linewidths=0)
        ax.set_title(f'{title}\nRMSE = {num(rmse(ys, yhat))}', loc='left', color=color,
                     fontsize=TEXT_PT, linespacing=1.3)
        ax.set_xlim(7, 53)
        ax.set_ylim(-0.5, 21.5)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_xlabel('$x$')
        tidy_axes(ax)
    axes[0].set_ylabel('$y$')
    # Mark the moved observation: its original position, and where it went.
    right = axes[1]
    right.scatter([X[-1]], [y_out[-1]], s=22, color=NML_RED, zorder=5, linewidths=0)
    right.scatter([X[-1]], [Y_CLEAN[-1]], s=22, facecolors='white', edgecolors=REF,
                  linewidths=0.8, zorder=5)
    right.annotate('', xy=(X[-1], y_out[-1] - 0.9), xytext=(X[-1], Y_CLEAN[-1] + 0.9),
                   arrowprops=dict(arrowstyle='-|>', color=REF, lw=LW_THIN, mutation_scale=6))
    note(right, X[-1] - 2.5, 19.2, 'moved from 7.8', ha='right', va='center')
    save_figure(fig, 'RMSE_sensitivity_outliers_plot')
    plt.close(fig)


def rmsle_vs_msle_plot():
    """One observation with Y = 10; MSLE and RMSLE against the prediction."""
    y_true = 10.0
    yhat = np.linspace(1, 40, 600)
    sle = (np.log1p(y_true) - np.log1p(yhat)) ** 2
    rsle = np.sqrt(sle)
    fig, ax = book_figure(0.82, 2.05)
    ax.axvline(y_true, color=REF, lw=LW_THIN, ls='--', zorder=1)
    ax.text(y_true + 0.6, 2.85, '$Y$ = 10', color=MUTED, va='top')
    ax.plot(yhat, sle, color=NML_CYAN)
    ax.plot(yhat, rsle, color=NML_RED)
    label_end(ax, yhat[-1], sle[-1], 'MSLE', NML_CYAN)
    label_end(ax, yhat[-1], rsle[-1], 'RMSLE', NML_RED)
    note(ax, 17, 2.75, 'under-prediction ($\\hat{Y} < Y$)\nis penalized more', ha='left', va='top')
    ax.set_xlim(0, 41)
    ax.set_ylim(0, 3)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel('prediction $\\hat{Y}$')
    ax.set_ylabel('loss for one observation')
    tidy_axes(ax)
    save_figure(fig, 'RMSLE_comparison_MSLE')
    plt.close(fig)


if __name__ == '__main__':
    rmse_outliers_plot()
    rmsle_vs_msle_plot()
