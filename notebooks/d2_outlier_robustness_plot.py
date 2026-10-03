"""D-squared as a ratio of absolute-error bar heights (Regression chapter).

Mirrors R2_explained: the same five points and fitted line, plus one outlier
(red) added after fitting. Bars measure absolute errors against the line (left,
SAD_res) and against the median (right, SAD_tot).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from style import *  # noqa: F401,F403
from r2_explained_plot import X as X5, Y as Y5, SLOPE, INTERCEPT

X = np.append(X5, 6.25)          # outlier between the second and third points
Y = np.append(Y5, 12.0)


def main():
    fit = SLOPE * X + INTERCEPT
    median = float(np.median(Y))
    sad_res = float(np.abs(Y - fit).sum())
    sad_tot = float(np.abs(Y - median).sum())
    d2 = 1 - sad_res / sad_tot
    r2 = 1 - float(((Y - fit) ** 2).sum()) / float(((Y - Y.mean()) ** 2).sum())
    clean_fit = SLOPE * X5 + INTERCEPT
    d2_clean = 1 - np.abs(Y5 - clean_fit).sum() / np.abs(Y5 - np.median(Y5)).sum()
    fig, axes = book_figure(1.0, 2.2, 1, 2)
    line_x = np.array([0.3, 15.7])
    panels = [(axes[0], fit, SLOPE * line_x + INTERCEPT, NML_PURPLE,
               f'model: linear fit\n$SAD_{{res}}$ = {num(sad_res, 1)}'),
              (axes[1], np.full_like(X, median), np.full(2, median), NML_RED,
               f'baseline: predict the median\n$SAD_{{tot}}$ = {num(sad_tot, 1)}')]
    for ax, ref, line_y, color, title in panels:
        ax.plot(line_x, line_y, color=INK, lw=LW_THIN + 0.3, zorder=3)
        for xi, yi, ri in zip(X, Y, ref):
            ax.plot([xi, xi], [ri, yi], color=color, lw=3.2, alpha=0.6, zorder=2,
                    solid_capstyle='butt')
        ax.scatter(X[:-1], Y[:-1], s=10, color=INK, zorder=4, linewidths=0)
        ax.scatter([X[-1]], [Y[-1]], s=14, color=NML_RED, zorder=5, linewidths=0)
        ax.set_title(title, loc='left', color=color, fontsize=TEXT_PT, linespacing=1.35)
        ax.set_xlim(0, 16.5)
        ax.set_ylim(0, 13)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        tidy_axes(ax)
    fig.supxlabel(f'$D^2$ = 1 − {num(sad_res, 1)} / {num(sad_tot, 1)} = {num(d2)}',
                  fontsize=TEXT_PT, y=0.0)
    save_figure(fig, 'D2_abs_comparison')
    plt.close(fig)
    print(f'D2: SAD_res {sad_res:.3f} SAD_tot {sad_tot:.3f} D2 {d2:.3f}; R2 with outlier {r2:.3f}; '
          f'clean D2 {d2_clean:.3f}')


if __name__ == '__main__':
    main()
