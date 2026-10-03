"""R-squared as a ratio of squared-error areas (Regression chapter).

Each residual becomes a square whose area is the squared error: against the
fitted line on the left (SS_res) and against the mean on the right (SS_tot).
The five points are spaced so that no two squares overlap and every square is
fully inside its panel; both panels share one scale.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle
from style import *  # noqa: F401,F403

X = np.array([1.0, 4.5, 8.0, 11.5, 15.0])
Y = np.array([3.1, 3.0, 5.5, 9.1, 10.0])
SLOPE, INTERCEPT = np.polyfit(X, Y, 1)


def squares(ax, y_ref, color):
    for xi, yi, ri in zip(X, Y, y_ref):
        side = abs(yi - ri)
        ax.add_patch(Rectangle((xi, min(yi, ri)), side, side, facecolor=color, alpha=0.45,
                               edgecolor=color, lw=0.5, zorder=2))


def main():
    fit = SLOPE * X + INTERCEPT
    mean = np.full_like(X, Y.mean())
    ss_res = float(((Y - fit) ** 2).sum())
    ss_tot = float(((Y - mean) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    fig, axes = book_figure(1.0, 2.05, 1, 2)
    line_x = np.array([0.3, 15.7])
    panels = [(axes[0], fit, SLOPE * line_x + INTERCEPT, NML_PURPLE,
               f'model: linear fit\n$SS_{{res}}$ = {num(ss_res, 1)}'),
              (axes[1], mean, np.full(2, Y.mean()), NML_RED,
               f'baseline: predict the mean\n$SS_{{tot}}$ = {num(ss_tot, 1)}')]
    for ax, ref, line_y, color, title in panels:
        ax.plot(line_x, line_y, color=INK, lw=LW_THIN + 0.3, zorder=3)
        squares(ax, ref, color)
        ax.scatter(X, Y, s=10, color=INK, zorder=4, linewidths=0)
        ax.set_title(title, loc='left', color=color, fontsize=TEXT_PT, linespacing=1.35)
        ax.set_xlim(0, 19.2)
        ax.set_ylim(0, 11)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        tidy_axes(ax)
    fig.supxlabel(f'$R^2$ = 1 − {num(ss_res, 1)} / {num(ss_tot, 1)} = {num(r2)}',
                  fontsize=TEXT_PT, y=0.0)
    save_figure(fig, 'R2_explained')
    plt.close(fig)
    print(f'R2: SS_res {ss_res:.3f} SS_tot {ss_tot:.3f} R2 {r2:.3f}')


if __name__ == '__main__':
    main()
