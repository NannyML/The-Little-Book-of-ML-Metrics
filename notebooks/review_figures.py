"""Figures rebuilt after Santiago's first review pass (2026-09-06).

Every figure follows the print-size rule in style.py: 8 in wide, fonts >= 14 pt,
so text prints at >= 6.5 pt when the figure spans the text width.  Every number
printed on a figure is computed here.  Run from notebooks/:

    uv run python review_figures.py
"""
import sys

sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
from style import *
from matplotlib.patches import Rectangle

DARK = '#2a2a2a'
MID = '#6f6f6f'
GREY = '#c9c9c9'
CORRECT = '#edf7ec'      # the book's strengths/weaknesses box tints
WRONG = '#fbecec'
CORRECT_EDGE = '#4EB046'
WRONG_EDGE = '#DD4040'
F = BOOK_FONT            # 15


def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


# ===========================================================================
# 1. MDA: direction right or wrong, day by day
# ===========================================================================
def fig_mda():
    rng = np.random.default_rng(7)
    T = 20
    t = np.arange(T)
    actual = 100 + np.cumsum(rng.normal(0, 2.0, T))
    # forecast for day t made at t-1: right direction most days, wrong size often
    forecast = actual.copy()
    for i in range(1, T):
        move = actual[i] - actual[i - 1]
        sign = np.sign(move) if rng.random() < 0.7 else -np.sign(move)
        forecast[i] = actual[i - 1] + sign * abs(rng.normal(0, 3.0))
    a_dir = np.sign(np.diff(actual))
    f_dir = np.sign(forecast[1:] - actual[:-1])
    hit = a_dir == f_dir
    mda = hit.mean()
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(FIG_W, 5.6), sharex=True,
                                   gridspec_kw={'height_ratios': [3.2, 1.25], 'hspace': 0.12})
    axT.plot(t, actual, color=DARK, lw=2.0, marker='o', ms=5, zorder=4)
    axT.plot(t[1:], forecast[1:], color=start_color, lw=0, marker='o', ms=6, zorder=3)
    for i in range(1, T):
        axT.plot([i - 1, i], [actual[i - 1], forecast[i]], color=start_color, lw=1.3, alpha=0.7, zorder=2)
    axT.text(0.02, 0.06, 'actual', color=DARK, fontsize=F, transform=axT.transAxes)
    axT.text(0.02, 0.16, 'forecast, made the day before', color=start_color, fontsize=F, transform=axT.transAxes)
    axT.set_ylabel('price', fontsize=F)
    axT.set_yticks([90, 95, 100])
    axT.tick_params(labelsize=F - 1)
    axT.tick_params(axis='x', length=0)
    despine(axT, keep=('left',))
    # direction strip: arrows for actual and forecast, colored by agreement
    for i in range(1, T):
        col = CORRECT_EDGE if hit[i - 1] else WRONG_EDGE
        axB.add_patch(Rectangle((i - 0.45, 0.1), 0.9, 1.8, facecolor=CORRECT if hit[i - 1] else WRONG, edgecolor='none'))
        axB.text(i, 1.45, '\u2191' if a_dir[i - 1] > 0 else '\u2193', ha='center', va='center', fontsize=F + 2, color=DARK)
        axB.text(i, 0.55, '\u2191' if f_dir[i - 1] > 0 else '\u2193', ha='center', va='center', fontsize=F + 2, color=col)
    axB.text(0.4, 1.45, 'actual move', ha='right', va='center', fontsize=F - 1, color=DARK)
    axB.text(0.4, 0.55, 'forecast move', ha='right', va='center', fontsize=F - 1, color=start_color)
    axB.set_ylim(0, 2.0)
    axB.set_xlim(-3.6, T + 0.2)
    axB.set_xticks(range(0, T, 5))
    axB.set_xlabel('day', fontsize=F)
    axB.tick_params(labelsize=F - 1, length=0)
    axB.set_yticks([])
    despine(axB, keep=())
    axT.set_title(f'MDA = {hit.sum()} of {T - 1} moves in the right direction = {mda:.2f}', loc='left', fontsize=F + 1, color=DARK, pad=10)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.12)
    save_figure(fig, 'MDA_direction')
    plt.close()
    # a day where the size was badly wrong but the direction right
    size_err = np.abs(forecast[1:] - actual[1:])
    worst_ok = int(np.argmax(np.where(hit, size_err, -1))) + 1
    print(f'1. MDA hits {hit.sum()}/{T-1} = {mda:.3f}; misses on days {[int(i)+1 for i in np.nonzero(~hit)[0]]}; '
          f'day {worst_ok}: direction right, size off by {size_err[worst_ok-1]:.1f}')
    for d in np.nonzero(~hit)[0] + 1:
        print(f'   miss day {d}: actual {actual[d-1]:.1f}->{actual[d]:.1f} ({actual[d]-actual[d-1]:+.1f}), forecast {forecast[d]:.1f} ({forecast[d]-actual[d-1]:+.1f})')


# ===========================================================================
# 2. Pinball loss: the tilted V, three quantiles
# ===========================================================================
def fig_pinball():
    u = np.linspace(-10, 10, 401)
    qs = [(0.1, start_color), (0.5, middle_color), (0.9, end_color)]
    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    for q, col in qs:
        loss = np.where(u >= 0, q * u, (q - 1) * u)
        ax.plot(u, loss, color=col, lw=2.4, zorder=3)
        ax.text(10.3, q * 10, f'q = {q}', color=col, fontsize=F, va='center')
        ax.text(-10.3, (1 - q) * 10, f'q = {q}', color=col, fontsize=F, va='center', ha='right')
    ax.text(-5, 9.6, 'over-predicted:  $\\hat{Y} > Y$\nslope $1 - q$', ha='center', va='top', fontsize=F - 1, color=MID)
    ax.text(5, 9.6, 'under-predicted:  $Y > \\hat{Y}$\nslope $q$', ha='center', va='top', fontsize=F - 1, color=MID)
    ax.axvline(0, color=GREY, lw=0.8, zorder=1)
    ax.set_xlim(-13, 13)
    ax.set_ylim(0, 10)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([0, 5, 10])
    ax.spines['left'].set_bounds(0, 10)
    ax.spines['bottom'].set_bounds(-10, 10)
    despine(ax)
    ax.tick_params(labelsize=F - 1)
    ax.set_xlabel('$Y - \\hat{Y}$', fontsize=F)
    ax.set_ylabel('loss', fontsize=F)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.16)
    save_figure(fig, 'Pinball_Loss')
    plt.close()
    print('2. Pinball: q=0.9 penalizes an under-prediction 9x an over-prediction of the same size')


# ===========================================================================
# 3. Explained variance vs R^2: three models on the same targets
# ===========================================================================
def fig_evs():
    rng = np.random.default_rng(3)
    n = 80
    x = rng.uniform(0, 10, n)
    y = 2 * x + rng.normal(0, 1.5, n)
    models = {
        'good': y + rng.normal(0, 1.2, n),
        'biased: shifted up': y + 6 + rng.normal(0, 1.2, n),
        'noisy': y + rng.normal(0, 5.0, n),
    }

    def r2(y, yh):
        return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def evs(y, yh):
        return 1 - np.var(y - yh) / np.var(y)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.6), sharey=True, gridspec_kw={'wspace': 0.12})
    out = {}
    lim = (-2, 30)
    for ax, (name, yh), col in zip(axes, models.items(), [start_color, end_color, middle_color]):
        e, r = evs(y, yh), r2(y, yh)
        out[name] = (e, r)
        ax.plot(lim, lim, color=GREY, lw=1, zorder=1)
        ax.scatter(y, yh, s=16, color=col, alpha=0.85, zorder=3, linewidths=0)
        ax.set_title(name, fontsize=F - 1, color=col, loc='left', pad=6)
        ax.text(0.05, 0.97, f'EVS = {e:.2f}\n$R^2$ = {r:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=F, color=DARK, linespacing=1.4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([0, 10, 20, 30])
        ax.tick_params(labelsize=F - 2)
        ax.set_xlabel('actual', fontsize=F - 1)
        despine(ax)
    axes[0].set_ylabel('predicted', fontsize=F - 1)
    bias = (models['biased: shifted up'] - y).mean()
    axes[1].annotate('', xy=(24, 24), xytext=(24, 24 + bias), arrowprops=dict(arrowstyle='<->', color=end_color, lw=1.2))
    axes[1].text(25, 24 + bias / 2, f'+{bias:.1f}', color=end_color, fontsize=F - 1, va='center')
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.18)
    save_figure(fig, 'EVS_vs_R2_three_models')
    plt.close()
    print('3. EVS/R2', {k: (round(e, 2), round(r, 2)) for k, (e, r) in out.items()}, 'bias', round(bias, 1))


# ===========================================================================
# 4. Confusion matrix: ten predictions become four cells, and the cells become the metrics
# ===========================================================================
def fig_confusion():
    actual = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0])
    kind = np.where(actual == 1, np.where(pred == 1, 'TP', 'FN'), np.where(pred == 1, 'FP', 'TN'))
    tp, fn, fp, tn = [(kind == k).sum() for k in ('TP', 'FN', 'FP', 'TN')]
    n = len(actual)
    fig = plt.figure(figsize=(FIG_W, 4.9))
    axL = fig.add_axes([0.02, 0.04, 0.34, 0.82])
    axR = fig.add_axes([0.40, 0.04, 0.58, 0.82])
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([]); despine(ax, keep=())
    # left: the ten predictions
    axL.set_xlim(0, 3.6); axL.set_ylim(-0.6, n + 1.2)
    for j, txt in enumerate(['actual', 'predicted', 'cell']):
        axL.text([0.55, 1.65, 2.9][j], n + 0.45, txt, ha='center', va='center', fontsize=F - 1, color=MID)
    for i in range(n):
        yy = n - 1 - i
        ok = actual[i] == pred[i]
        axL.add_patch(Rectangle((0.05, yy - 0.42), 3.5, 0.84, facecolor=CORRECT if ok else WRONG, edgecolor='none'))
        axL.text(0.55, yy, str(actual[i]), ha='center', va='center', fontsize=F, color=DARK)
        axL.text(1.65, yy, str(pred[i]), ha='center', va='center', fontsize=F, color=DARK)
        axL.text(2.9, yy, kind[i], ha='center', va='center', fontsize=F, color=CORRECT_EDGE if ok else WRONG_EDGE, fontweight='bold')
    # right: the matrix
    axR.set_xlim(-1.9, 4.2); axR.set_ylim(-1.75, 3.5)
    cells = {(0, 1): ('TP', tp, True), (1, 1): ('FN', fn, False), (0, 0): ('FP', fp, False), (1, 0): ('TN', tn, True)}
    for (cx, cy), (name, cnt, ok) in cells.items():
        axR.add_patch(Rectangle((cx, cy), 1, 1, facecolor=CORRECT if ok else WRONG, edgecolor='white', lw=3))
        axR.text(cx + 0.5, cy + 0.62, str(cnt), ha='center', va='center', fontsize=F + 9, color=CORRECT_EDGE if ok else WRONG_EDGE, fontweight='bold')
        axR.text(cx + 0.5, cy + 0.22, name, ha='center', va='center', fontsize=F - 1, color=MID)
    axR.text(1.0, 2.62, 'predicted', ha='center', va='center', fontsize=F - 1, color=DARK)
    axR.text(0.5, 2.22, '1', ha='center', va='center', fontsize=F - 1, color=DARK)
    axR.text(1.5, 2.22, '0', ha='center', va='center', fontsize=F - 1, color=DARK)
    axR.text(-0.15, 1.5, 'actual 1', ha='right', va='center', fontsize=F - 1, color=DARK)
    axR.text(-0.15, 0.5, 'actual 0', ha='right', va='center', fontsize=F - 1, color=DARK)
    # the reads: rows -> recall / FPR, columns -> precision, diagonal -> accuracy
    kw = dict(fontsize=F - 1, color=DARK, va='center')
    axR.annotate('', xy=(2.08, 1.5), xytext=(2.08, 1.5), arrowprops=dict(arrowstyle='-'))
    axR.plot([2.1, 2.1], [1.06, 1.94], color=MID, lw=1)
    axR.text(2.22, 1.5, f'across the row:\nrecall = TP / (TP + FN) = {tp}/{tp+fn} = {tp/(tp+fn):.2f}', ha='left', **kw)
    axR.plot([2.1, 2.1], [0.06, 0.94], color=MID, lw=1)
    axR.text(2.22, 0.5, f'across the row:\nFPR = FP / (FP + TN) = {fp}/{fp+tn} = {fp/(fp+tn):.2f}', ha='left', **kw)
    axR.plot([0.06, 0.94], [-0.12, -0.12], color=MID, lw=1)
    axR.text(0.5, -0.32, f'down the column:\nprecision = TP / (TP + FP) = {tp}/{tp+fp} = {tp/(tp+fp):.2f}', ha='center', va='top', fontsize=F - 1, color=DARK)
    axR.text(1.5, -1.25, f'diagonal over everything:\naccuracy = (TP + TN) / {n} = {(tp+tn)/n:.2f}', ha='center', va='top', fontsize=F - 1, color=DARK)
    fig.text(0.02, 0.96, f'{n} predictions become four counts, and every metric is a different read of them', fontsize=F - 1, color=MID, va='top')
    save_figure(fig, 'Confusion_Matrix_cells')
    plt.close()
    print(f'4. Confusion: TP {tp} FN {fn} FP {fp} TN {tn}; recall {tp/(tp+fn):.2f} precision {tp/(tp+fp):.2f} FPR {fp/(fp+tn):.2f} accuracy {(tp+tn)/n:.2f}')


if __name__ == '__main__':
    fig_mda(); fig_pinball(); fig_evs(); fig_confusion()
