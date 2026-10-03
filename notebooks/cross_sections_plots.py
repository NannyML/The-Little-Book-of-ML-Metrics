"""One-observation deviance curves for the Regression chapter (Y fixed at 10).

MPD_cross_section and MGD_cross_section share the layout of
RMSLE_comparison_MSLE: the prediction on the horizontal axis and the actual
value marked by a dashed line. The points compared in each caption are marked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from style import *  # noqa: F401,F403

Y_TRUE = 10.0
YHAT = np.linspace(1, 30, 600)


def _curve(name, loss_fn, ylim, yticks, points, note_xy, note_text):
    fig, ax = book_figure(0.82, 2.05)
    ax.axvline(Y_TRUE, color=REF, lw=LW_THIN, ls='--', zorder=1)
    ax.text(Y_TRUE + 0.5, ylim * 0.97, '$Y$ = 10', color=MUTED, va='top')
    ax.plot(YHAT, loss_fn(YHAT), color=NML_CYAN, clip_on=True)
    for x, dx, dy, ha in points:
        y = float(loss_fn(np.array([x]))[0])
        ax.scatter([x], [y], s=16, color=NML_CYAN, edgecolors='white', linewidths=0.6, zorder=4)
        ax.annotate(num(y, 2), (x, y), xytext=(dx, dy),
                    textcoords='offset points', ha=ha, va='center', color=INK)
    note(ax, *note_xy, note_text, ha='left', va='top')
    ax.set_xlim(0, 31)
    ax.set_ylim(0, ylim)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_yticks(yticks)
    ax.set_xlabel('prediction $\\hat{Y}$')
    ax.set_ylabel('deviance for one observation')
    tidy_axes(ax)
    save_figure(fig, name)
    plt.close(fig)


def mpd_cross():
    loss = lambda p: 2 * (Y_TRUE * np.log(Y_TRUE / p) - (Y_TRUE - p))
    _curve('MPD_cross_section', loss, 10, [0, 2, 4, 6, 8, 10],
           [(5, -5, 0, 'right'), (15, 5, -3, 'left')], (20, 4.2),
           'both miss by 5;\nunder-prediction is\npenalized more')


def mgd_cross():
    loss = lambda p: 2 * (np.log(p / Y_TRUE) + Y_TRUE / p - 1)
    _curve('MGD_cross_section', loss, 2.5, [0, 0.5, 1.0, 1.5, 2.0, 2.5],
           [(5, -5, 0, 'right'), (20, 0, 8, 'center')], (19, 2.42),
           'half vs double the\nactual value: the half\nis penalized more')


if __name__ == '__main__':
    mpd_cross()
    mgd_cross()
