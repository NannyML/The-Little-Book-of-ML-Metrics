"""3D metric surfaces for the Regression and Classification chapters.

All thirteen surfaces share one layout: printed size, viewpoint per chapter,
tick labels clear of the box, and a colorbar whose range and ticks equal the
vertical axis. Run: uv run python notebooks/surfaces.py [NAME ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
from style import *  # noqa: F401,F403
from matplotlib import ticker

YHAT = r'$\hat{Y}$'


def surface(name, X, Y, Z, *, xlabel, ylabel, cbar_label, xticks, yticks, zticks,
            cmap, view, xlim=None, ylim=None, zero_contour=False, height=2.62,
            fmt=None, zfmt=None, pads=(0, 1, 4), zoom=0.84, corner=None):
    fig = book_canvas(0.82, height)
    ax = fig.add_axes([-0.06, -0.02, 0.92, 1.04], projection='3d')
    cax = fig.add_axes([0.875, 0.22, 0.026, 0.58])
    lo, hi = zticks[0], zticks[-1]
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, vmin=lo, vmax=hi, linewidth=0,
                           antialiased=False, rcount=120, ccount=120, zorder=1)
    if zero_contour:
        # Where the score changes sign, projected on the floor and labeled.
        ax.contour(X, Y, Z, levels=[0], zdir='z', offset=lo, colors=[INK],
                   linewidths=0.8, linestyles='--')
        x0, y0 = zero_contour
        ax.text(x0, y0, lo, 'MCC = 0', color=INK, fontsize=TEXT_PT, ha='left', va='center',
                zdir=None)
    ax.set_xlim(*(xlim or (xticks[0], xticks[-1])))
    ax.set_ylim(*(ylim or (yticks[0], yticks[-1])))
    ax.set_zlim(lo, hi)
    fmt = fmt or (lambda v: num(v, 0) if float(v).is_integer() else num(v, 1))
    zfmt = zfmt or fmt
    ax.set_xticks(xticks, labels=[fmt(t) for t in xticks])
    # The two floor axes meet at the front corner; label the shared corner
    # value once (on the front axis) so the two numbers do not run together.
    ax.set_yticks(yticks, labels=['' if t == corner else fmt(t) for t in yticks])
    ax.set_zticks(zticks, labels=[zfmt(t) for t in zticks])
    ax.set_xlabel(xlabel, labelpad=pads[1])
    ax.set_ylabel(ylabel, labelpad=pads[1])
    ax.set_zlabel('')
    ax.tick_params(axis='x', pad=pads[0])
    ax.tick_params(axis='y', pad=pads[0])
    ax.tick_params(axis='z', pad=pads[2])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor('#f4f4f4')
        axis.pane.set_edgecolor(LIGHT)
        axis._axinfo['grid'].update(color='#dddddd', linewidth=0.4)
        axis.line.set_linewidth(0.6)
        axis._axinfo['tick'].update(inward_factor=0.0, outward_factor=0.25)
        axis._axinfo['tick']['linewidth'] = {True: 0.6, False: 0.4}
    ax.view_init(*view)
    ax.set_box_aspect((1, 1, 0.78), zoom=zoom)
    cb = fig.colorbar(surf, cax=cax, ticks=zticks)
    cb.ax.set_yticklabels([zfmt(t) for t in zticks])
    cb.set_label(cbar_label, labelpad=4)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(width=0.6, length=2.5, pad=2)
    save_figure(fig, name, crop=True)
    plt.close(fig)


def regression():
    y = np.linspace(10, 50, 161)
    Yg, Pg = np.meshgrid(y, y)          # x: actual value, y: prediction
    common = dict(xlabel='actual value (Y)', ylabel=f'predicted value ({YHAT})',
                  xticks=[10, 20, 30, 40, 50], yticks=[10, 20, 30, 40, 50],
                  cmap=nml_cmap, view=(15, 25), corner=50)
    yield 'MAE_3d_surface', lambda: surface('MAE_3d_surface', Yg, Pg, np.abs(Yg - Pg),
        cbar_label='absolute error', zticks=[0, 10, 20, 30, 40], **common)
    yield 'MSE_3d_surface', lambda: surface('MSE_3d_surface', Yg, Pg, (Yg - Pg) ** 2,
        cbar_label='squared error', zticks=[0, 400, 800, 1200, 1600],
        fmt=lambda v: num(v, 0, thousands=True), **common)
    yield 'MAPE_3d_surface', lambda: surface('MAPE_3d_surface', Yg, Pg, 100 * np.abs(Yg - Pg) / Yg,
        cbar_label='absolute percentage error (%)', zticks=[0, 100, 200, 300, 400], **common)
    yield 'sMAPE_3d_surface', lambda: surface('sMAPE_3d_surface', Yg, Pg,
        200 * np.abs(Yg - Pg) / (np.abs(Yg) + np.abs(Pg)),
        cbar_label='sMAPE term (%)', zticks=[0, 50, 100, 150], **common)
    # MSLE starts at 1 so the caption's example (Y = 10, predictions 5 and 15)
    # and the steep penalty near zero are on the surface.
    v = np.linspace(1, 50, 197)
    Yl, Pl = np.meshgrid(v, v)
    msle = (np.log1p(Yl) - np.log1p(Pl)) ** 2
    yield 'MSLE_3d_surface', lambda: surface('MSLE_3d_surface', Yl, Pl, msle,
        cbar_label='squared log error', zticks=[0, 3, 6, 9, 12],
        xlim=(1, 50), ylim=(1, 50), **{**common, 'xticks': [10, 30, 50]})


def classification():
    c = np.linspace(1, 100, 100)   # counts start at 1: a 0/0 rate is undefined
    A, B = np.meshgrid(c, c)
    unit = [0, 0.5, 1.0]
    cnt = [0, 25, 50, 75, 100]
    good = nml_cmap.reversed()
    view = (20, -65)
    rate = dict(xticks=cnt, yticks=cnt, zticks=unit, view=view,
                ylim=(100, 0), xlim=(0, 100), corner=100,
                fmt=lambda v: num(v, 0), zfmt=lambda v: num(v, 1))
    yield 'FPR_3d_surface', lambda: surface('FPR_3d_surface', A, B, A / (A + B),
        xlabel='false positives (FP)', ylabel='true negatives (TN)', cbar_label='FPR',
        cmap=nml_cmap, **rate)
    yield 'FNR_3d_surface', lambda: surface('FNR_3d_surface', A, B, A / (A + B),
        xlabel='false negatives (FN)', ylabel='true positives (TP)', cbar_label='FNR',
        cmap=nml_cmap, **rate)
    yield 'Recall_3d_surface', lambda: surface('Recall_3d_surface', A, B, A / (A + B),
        xlabel='true positives (TP)', ylabel='false negatives (FN)', cbar_label='recall (TPR)',
        cmap=good, **rate)
    yield 'TNR_3d_surface', lambda: surface('TNR_3d_surface', A, B, A / (A + B),
        xlabel='true negatives (TN)', ylabel='false positives (FP)', cbar_label='TNR',
        cmap=good, **rate)
    yield 'Precision_3d_surface', lambda: surface('Precision_3d_surface', A, B, A / (A + B),
        xlabel='true positives (TP)', ylabel='false positives (FP)', cbar_label='precision',
        cmap=good, **rate)
    # Accuracy with FP + FN fixed at 50 (caption); TP and TN on the same 0-100
    # range as the other rate surfaces and the color scale on the full 0-1 range.
    acc = (A + B) / (A + B + 50)
    yield 'Accuracy_3d_surface', lambda: surface('Accuracy_3d_surface', A, B, acc,
        xlabel='true positives (TP)', ylabel='true negatives (TN)', cbar_label='accuracy',
        cmap=good, xticks=cnt, yticks=cnt, zticks=unit, view=view, corner=0,
        xlim=(0, 100), ylim=(0, 100), fmt=lambda v: num(v, 0), zfmt=lambda v: num(v, 1))
    p = np.linspace(0.01, 1, 100)
    P, R = np.meshgrid(p, p)
    yield 'F1_3d_surface', lambda: surface('F1_3d_surface', P, R, 2 * P * R / (P + R),
        xlabel='precision', ylabel='recall', cbar_label='F1-score', cmap=good,
        xticks=unit, yticks=unit, zticks=unit, view=view, xlim=(0, 1), ylim=(0, 1),
        corner=0, fmt=lambda v: num(v, 1))
    # MCC with FN = 20 and TN = 80; symmetric color scale so 0 (no association)
    # sits in the middle of the colormap, and the zero contour is drawn.
    FN, TN = 20, 80
    mcc = (A * TN - B * FN) / np.sqrt((A + B) * (A + FN) * (TN + B) * (TN + FN))
    yield 'MCC_3d_surface', lambda: surface('MCC_3d_surface', A, B, mcc,
        xlabel='true positives (TP)', ylabel='false positives (FP)', cbar_label='MCC',
        cmap=good, xticks=cnt, yticks=cnt, zticks=[-1, -0.5, 0, 0.5, 1.0], view=view,
        xlim=(0, 100), ylim=(100, 0), zero_contour=(34, 92), corner=100,
        fmt=lambda v: num(v, 0), zfmt=lambda v: num(v, 1))


if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for group in (regression(), classification()):
        for name, make in group:
            if not wanted or name in wanted:
                make()
