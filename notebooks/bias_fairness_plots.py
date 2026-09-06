"""Generate figures for the Bias & Fairness chapter.

One population, five lenses.  Every figure draws the same two groups of 100
people as grids of squares coloured by confusion-matrix cell, and outlines the
subset of people the metric conditions on.  The model is the same on every
page: a single threshold that catches 5 of every 6 qualified people (TPR = 5/6)
and wrongly flags 1 in 10 unqualified people (FPR = 1/10) in BOTH groups, applied
to groups whose base rates differ (60% and 30% qualified).  All counts and rates
are computed from those definitions; the calibration curves come from a
simulated score model with the same error rates.  Nothing is typed in by hand.

    TP  = qualified and flagged        (filled cyan)
    FN  = qualified but missed         (cyan outline)
    FP  = unqualified but flagged      (filled red)
    TN  = unqualified and not flagged  (grey)

Run from notebooks/:  uv run python bias_fairness_plots.py
"""
import sys
from fractions import Fraction

sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
from style import *
from matplotlib.patches import Rectangle
from scipy.stats import norm
from scipy.optimize import brentq

GREY = '#d6d6d6'
GREY_LINE = '#9a9a9a'
DARK = '#2a2a2a'
MID = '#6f6f6f'
RNG = np.random.default_rng(11)

# ---------------------------------------------------------------------------
# The shared population and model
# ---------------------------------------------------------------------------
N = 100
BASE = {'A': Fraction(60, 100), 'B': Fraction(30, 100)}
TPR = Fraction(5, 6)
FPR = Fraction(1, 10)


def counts(group, tpr=TPR, fpr=FPR):
    """Confusion-matrix counts for a group of N people (exact integers)."""
    pos = BASE[group] * N
    neg = N - pos
    tp, fp = tpr * pos, fpr * neg
    assert tp.denominator == 1 and fp.denominator == 1 and pos.denominator == 1
    tp, fp, pos, neg = int(tp), int(fp), int(pos), int(neg)
    return dict(TP=tp, FN=pos - tp, FP=fp, TN=neg - fp, pos=pos, neg=neg)


# score model with the same class-conditional distributions in both groups,
# chosen so that a threshold at z = 0 (score 0.5) gives exactly TPR and FPR
MU1 = norm.ppf(float(TPR))          # z ~ N(MU1, 1) for qualified
MU0 = norm.ppf(float(FPR))          # z ~ N(MU0, 1) for unqualified


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
STYLE = {
    'TP': dict(facecolor=start_color, edgecolor=start_color),
    'FN': dict(facecolor='white', edgecolor=start_color),
    'FP': dict(facecolor=end_color, edgecolor=end_color),
    'TN': dict(facecolor=GREY, edgecolor=GREY),
}
COLS = 10


def draw_grid(ax, c, order, highlight, x0=0, y0=0, size=1.0, gap=0.18, dim=0.22):
    """Draw N people row by row in the given cell order.  Cells whose kind is
    in `highlight` are drawn at full strength, the rest are dimmed.  Returns a
    dict kind -> list of (row, col) positions and the outline of the highlight."""
    kinds = []
    for k in order:
        kinds += [k] * c[k]
    assert len(kinds) == N
    pos = {}
    in_set = np.zeros((N // COLS, COLS), dtype=bool)
    for i, k in enumerate(kinds):
        r, col = divmod(i, COLS)
        x = x0 + col * (size + gap)
        y = y0 - r * (size + gap)
        alpha = 1.0 if k in highlight else dim
        st = STYLE[k]
        ax.add_patch(Rectangle((x, y), size, size, facecolor=st['facecolor'], edgecolor=st['edgecolor'],
                               linewidth=1.6, alpha=alpha, zorder=2))
        pos.setdefault(k, []).append((r, col))
        in_set[r, col] = k in highlight
    # outline around the highlighted block (row-major contiguous block)
    n_hi = int(in_set.sum())
    if 0 < n_hi < N:
        first = kinds.index(next(k for k in order if k in highlight))
        idx = [i for i, k in enumerate(kinds) if k in highlight]
        assert idx == list(range(idx[0], idx[0] + len(idx))), 'highlight must be contiguous'
        _outline_block(ax, idx[0], idx[-1], x0, y0, size, gap)
    elif n_hi == N:
        _outline_block(ax, 0, N - 1, x0, y0, size, gap)
    return pos


def _outline_block(ax, i0, i1, x0, y0, size, gap, lw=2.2):
    """Thick outline around cells i0..i1 (inclusive) of a row-major grid."""
    step = size + gap
    pad = gap / 2
    r0, c0 = divmod(i0, COLS)
    r1, c1 = divmod(i1, COLS)

    def X(col):
        return x0 + col * step - pad

    def Y(row):
        return y0 - row * step + size + pad

    def Yb(row):
        return y0 - row * step - pad

    if r0 == r1:
        pts = [(X(c0), Y(r0)), (X(c1 + 1), Y(r0)), (X(c1 + 1), Yb(r0)), (X(c0), Yb(r0))]
    else:
        pts = [(X(c0), Y(r0)), (X(COLS), Y(r0)), (X(COLS), Yb(r1 - 1))]
        if c1 < COLS - 1:
            pts += [(X(c1 + 1), Yb(r1 - 1)), (X(c1 + 1), Yb(r1))]
        else:
            pts += [(X(COLS), Yb(r1))]
        pts += [(X(0), Yb(r1)), (X(0), Y(r0 + 1))]
        if c0 > 0:
            pts += [(X(c0), Y(r0 + 1))]
    pts.append(pts[0])
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=DARK, lw=lw, solid_joinstyle='miter', zorder=4)


def setup(ax, ncols, size=1.0, gap=0.18, top_pad=2.4, bottom_pad=2.6):
    step = size + gap
    ax.set_xlim(-0.6, ncols * step + 0.2)
    ax.set_ylim(-(N // COLS) * step - bottom_pad, size + top_pad)
    ax.set_aspect('equal')
    ax.axis('off')


def legend(fig, y=0.03, x=0.5):
    items = [('TP', 'qualified, flagged'), ('FN', 'qualified, missed'), ('FP', 'unqualified, flagged'),
             ('TN', 'unqualified, not flagged')]
    handles = [Rectangle((0, 0), 1, 1, facecolor=STYLE[k]['facecolor'], edgecolor=STYLE[k]['edgecolor'], lw=1.6)
               for k, _ in items]
    fig.legend(handles, [t for _, t in items], loc='lower center', bbox_to_anchor=(x, y), ncol=4, frameon=False,
               fontsize=11.5, handlelength=1.0, handleheight=1.0, columnspacing=1.6)


def pct(num, den):
    return f'{100 * num / den:.0f}%'


def grid_title(ax, x, text, sub=None, size=1.0, gap=0.18):
    step = size + gap
    ax.text(x + (COLS * step - gap) / 2, size + 1.55, text, ha='center', va='bottom', fontsize=14, color=DARK)
    if sub:
        ax.text(x + (COLS * step - gap) / 2, size + 0.55, sub, ha='center', va='bottom', fontsize=11, color=MID)


def grid_result(ax, x, text, color=DARK, size=1.0, gap=0.18, dy=0):
    step = size + gap
    ax.text(x + (COLS * step - gap) / 2, -(N // COLS) * step - 0.55 + dy, text, ha='center', va='top', fontsize=13.5,
            color=color)


cA, cB = counts('A'), counts('B')
print('A:', cA, ' B:', cB)


# ===========================================================================
# 1. Demographic Parity — who is flagged, regardless of truth
#    Grids: A and B under one threshold (parity fails), then B with its
#    threshold lowered until it flags as many as A (parity holds, merit differs).
# ===========================================================================
def fig_dp():
    order = ['TP', 'FP', 'FN', 'TN']       # flagged people first
    hi = {'TP', 'FP'}
    selA = cA['TP'] + cA['FP']
    selB = cB['TP'] + cB['FP']
    # lower B's threshold until it selects selA people (score model, then round)
    pB = float(BASE['B'])
    t = brentq(lambda z: pB * norm.sf(z - MU1) + (1 - pB) * norm.sf(z - MU0) - selA / N, -6, 6)
    tpB2 = int(round(cB['pos'] * norm.sf(t - MU1)))
    fpB2 = selA - tpB2
    cB2 = dict(TP=tpB2, FN=cB['pos'] - tpB2, FP=fpB2, TN=cB['neg'] - fpB2, pos=cB['pos'], neg=cB['neg'])
    fig, ax = plt.subplots(figsize=(13, 5.4))
    step = 1.18
    xs = [0, 14.6, 29.2]
    setup(ax, 3 * COLS + 2 * 2)
    ax.set_xlim(-0.6, xs[2] + COLS * step + 0.2)
    draw_grid(ax, cA, order, hi, x0=xs[0])
    draw_grid(ax, cB, order, hi, x0=xs[1])
    draw_grid(ax, cB2, order, hi, x0=xs[2])
    grid_title(ax, xs[0], 'Group A', f'{cA["pos"]} of 100 qualified')
    grid_title(ax, xs[1], 'Group B', f'{cB["pos"]} of 100 qualified')
    grid_title(ax, xs[2], 'Group B, threshold lowered', 'until it flags as many as A')
    grid_result(ax, xs[0], f'flagged {selA} of 100 = {pct(selA, N)}')
    grid_result(ax, xs[1], f'flagged {selB} of 100 = {pct(selB, N)}')
    grid_result(ax, xs[2], f'flagged {selA} of 100 = {pct(selA, N)}')
    grid_result(ax, xs[0], f'{cA["TP"]} qualified, {cA["FP"]} not', color=MID, dy=-1.1)
    grid_result(ax, xs[1], f'{cB["TP"]} qualified, {cB["FP"]} not', color=MID, dy=-1.1)
    grid_result(ax, xs[2], f'{cB2["TP"]} qualified, {cB2["FP"]} not', color=end_color, dy=-1.1)
    mid1 = (xs[0] + COLS * step - 0.18 + xs[1]) / 2
    mid2 = (xs[1] + COLS * step - 0.18 + xs[2]) / 2
    ax.text(mid1, -4.8, 'parity\nfails', ha='center', va='center', fontsize=12, color=end_color)
    ax.text(mid2, -4.8, 'parity\nholds', ha='center', va='center', fontsize=12, color=start_color)
    legend(fig, y=0.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.08)
    save_figure(fig, 'Demographic_Parity')
    plt.close()
    print(f'1. DP: sel A={selA} B={selB} | B lowered: threshold z={t:.2f} -> TP={tpB2} FP={fpB2} '
          f'(PPV {tpB2 / selA:.2f}, TPR {tpB2 / cB["pos"]:.2f}, FPR {fpB2 / cB["neg"]:.2f})')
    return cB2


# ===========================================================================
# 2. Equality of Opportunity — restrict to the qualified, compare TPR
# ===========================================================================
def fig_eopp():
    order = ['TP', 'FN', 'FP', 'TN']       # qualified people first
    hi = {'TP', 'FN'}
    fig, ax = plt.subplots(figsize=(13, 5.4))
    step = 1.18
    xs = [0, 16.6]
    setup(ax, 2 * COLS + 2)
    ax.set_xlim(-0.6, xs[1] + COLS * step + 0.2)
    draw_grid(ax, cA, order, hi, x0=xs[0])
    draw_grid(ax, cB, order, hi, x0=xs[1])
    grid_title(ax, xs[0], 'Group A', f'{cA["pos"]} qualified, outlined')
    grid_title(ax, xs[1], 'Group B', f'{cB["pos"]} qualified, outlined')
    grid_result(ax, xs[0], f'caught {cA["TP"]} of {cA["pos"]}  →  TPR = {pct(cA["TP"], cA["pos"])}')
    grid_result(ax, xs[1], f'caught {cB["TP"]} of {cB["pos"]}  →  TPR = {pct(cB["TP"], cB["pos"])}')
    grid_result(ax, xs[0], f'missed {cA["FN"]}', color=MID, dy=-1.1)
    grid_result(ax, xs[1], f'missed {cB["FN"]}', color=MID, dy=-1.1)
    mid = (xs[0] + COLS * step - 0.18 + xs[1]) / 2
    ax.text(mid, -4.8, 'equal\nTPR ✓', ha='center', va='center', fontsize=12.5, color=start_color)
    legend(fig, y=0.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.08)
    save_figure(fig, 'Equality_of_Opportunity')
    plt.close()
    print(f'2. EOpp: TPR A={cA["TP"] / cA["pos"]:.3f} B={cB["TP"] / cB["pos"]:.3f}')


# ===========================================================================
# 3. Equality of Odds — both blocks: TPR among the qualified, FPR among the rest
# ===========================================================================
def fig_eodds():
    order = ['TP', 'FN', 'FP', 'TN']
    fig, ax = plt.subplots(figsize=(13, 5.4))
    step = 1.18
    xs = [0, 16.6]
    setup(ax, 2 * COLS + 2)
    ax.set_xlim(-0.6, xs[1] + COLS * step + 0.2)
    for x0, c, name in [(xs[0], cA, 'A'), (xs[1], cB, 'B')]:
        draw_grid(ax, c, order, {'TP', 'FN', 'FP', 'TN'}, x0=x0)   # nothing dimmed; two outlines instead
        # outline the qualified block and the unqualified block separately
        _outline_block(ax, 0, c['pos'] - 1, x0, 0, 1.0, 0.18)
        _outline_block(ax, c['pos'], N - 1, x0, 0, 1.0, 0.18)
        grid_title(ax, x0, f'Group {name}', f'{c["pos"]} qualified (top block), {c["neg"]} not (bottom block)')
        grid_result(ax, x0, f'TPR = {c["TP"]} of {c["pos"]} = {pct(c["TP"], c["pos"])}')
        grid_result(ax, x0, f'FPR = {c["FP"]} of {c["neg"]} = {pct(c["FP"], c["neg"])}', dy=-1.15)
    mid = (xs[0] + COLS * step - 0.18 + xs[1]) / 2
    ax.text(mid, -4.8, 'same\n(FPR, TPR)\npoint ✓', ha='center', va='center', fontsize=12, color=start_color)
    legend(fig, y=0.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.08)
    save_figure(fig, 'Equality_of_Odds')
    plt.close()
    print(f'3. EOdds: A=({cA["FP"] / cA["neg"]:.2f},{cA["TP"] / cA["pos"]:.2f}) '
          f'B=({cB["FP"] / cB["neg"]:.2f},{cB["TP"] / cB["pos"]:.2f})')


# ===========================================================================
# 4. Predictive Parity — restrict to the flagged, compare PPV
# ===========================================================================
def fig_pp():
    order = ['TP', 'FP', 'FN', 'TN']       # flagged people first
    hi = {'TP', 'FP'}
    fig, ax = plt.subplots(figsize=(13, 5.4))
    step = 1.18
    xs = [0, 16.6]
    setup(ax, 2 * COLS + 2)
    ax.set_xlim(-0.6, xs[1] + COLS * step + 0.2)
    draw_grid(ax, cA, order, hi, x0=xs[0])
    draw_grid(ax, cB, order, hi, x0=xs[1])
    selA, selB = cA['TP'] + cA['FP'], cB['TP'] + cB['FP']
    grid_title(ax, xs[0], 'Group A', f'{selA} flagged, outlined')
    grid_title(ax, xs[1], 'Group B', f'{selB} flagged, outlined')
    grid_result(ax, xs[0], f'{cA["TP"]} of {selA} truly qualified  →  PPV = {pct(cA["TP"], selA)}')
    grid_result(ax, xs[1], f'{cB["TP"]} of {selB} truly qualified  →  PPV = {pct(cB["TP"], selB)}')
    grid_result(ax, xs[0], f'{cA["FP"]} flagged by mistake', color=MID, dy=-1.1)
    grid_result(ax, xs[1], f'{cB["FP"]} flagged by mistake', color=MID, dy=-1.1)
    mid = (xs[0] + COLS * step - 0.18 + xs[1]) / 2
    ax.text(mid, -4.8, f'{100 * cA["TP"] / selA - 100 * cB["TP"] / selB:.0f}-point\ngap', ha='center', va='center',
            fontsize=12.5, color=end_color)
    legend(fig, y=0.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=1.0, bottom=0.08)
    save_figure(fig, 'Predictive_Parity')
    plt.close()
    print(f'4. PP: PPV A={cA["TP"] / selA:.3f} B={cB["TP"] / selB:.3f}')


# ===========================================================================
# 5. Calibration within Groups — for each score bin, how many turn out
#    qualified?  Same score model as above, simulated for many people.
# ===========================================================================
def fig_cal():
    n = 200_000
    bins = np.linspace(0, 1, 11)
    frac = {}
    for g in ('A', 'B'):
        p = float(BASE[g])
        y = (RNG.random(n) < p).astype(int)
        z = np.where(y == 1, RNG.normal(MU1, 1, n), RNG.normal(MU0, 1, n))
        s = sigmoid(z)
        idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
        frac[g] = np.array([y[idx == b].mean() for b in range(10)])
    fig, ax = plt.subplots(figsize=(13, 5.0))
    size, gap = 0.62, 0.12
    col_step = size + gap
    bin_w = 3.2
    for b in range(10):
        xb = b * bin_w
        for j, (g, color) in enumerate([('A', start_color), ('B', middle_color)]):
            x = xb + j * (col_step + 0.15)
            k = int(round(10 * frac[g][b]))
            for i in range(10):
                filled = i < k
                ax.add_patch(Rectangle((x, i * col_step), size, size,
                                       facecolor=color if filled else 'white', edgecolor=color if filled else GREY_LINE,
                                       linewidth=1.3, zorder=2))
            ax.text(x + size / 2, 10 * col_step + 0.15, f'{100 * frac[g][b]:.0f}', ha='center', va='bottom',
                    fontsize=10.5, color=color)
        # where a calibrated score would put the fill: the bin's midpoint
        mid_s = (bins[b] + bins[b + 1]) / 2
        yline = mid_s * 10 * col_step - gap / 2
        ax.plot([xb - 0.25, xb + 2 * col_step + 0.15 + 0.1], [yline, yline], color=DARK, lw=1.3, zorder=3)
        ax.text(xb + col_step + 0.05, -0.7, f'{bins[b]:.1f}–{bins[b + 1]:.1f}', ha='center', va='top', fontsize=10.5,
                color=MID)
    ax.set_xlim(-0.8, 10 * bin_w - 0.8)
    ax.set_ylim(-2.0, 10 * col_step + 2.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(10 * bin_w / 2 - 0.8, -1.55, 'model score', ha='center', va='top', fontsize=12.5, color=DARK)
    ax.text(-0.6, 10 * col_step + 1.45, 'Of 10 people with a given score, how many turn out to be qualified?',
            ha='left', va='bottom', fontsize=12.5, color=DARK)
    # legend
    hA = Rectangle((0, 0), 1, 1, facecolor=start_color, edgecolor=start_color)
    hB = Rectangle((0, 0), 1, 1, facecolor=middle_color, edgecolor=middle_color)
    hL = plt.Line2D([0], [0], color=DARK, lw=1.3)
    fig.legend([hA, hB, hL], ['Group A', 'Group B', 'what a calibrated score would give (the score itself)'],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=11.5,
               handlelength=1.0, handleheight=1.0, columnspacing=1.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12)
    save_figure(fig, 'Calibration_within_Groups')
    plt.close()
    b5 = 5   # the 0.5-0.6 bin
    print(f'5. Cal: bin {bins[b5]:.1f}-{bins[b5 + 1]:.1f}: A={frac["A"][b5]:.2f} B={frac["B"][b5]:.2f}; '
          f'bins A={np.round(frac["A"], 2)} B={np.round(frac["B"], 2)}')


if __name__ == '__main__':
    fig_dp()
    fig_eopp()
    fig_eodds()
    fig_pp()
    fig_cal()
    print('\nAll bias & fairness figures regenerated.')
