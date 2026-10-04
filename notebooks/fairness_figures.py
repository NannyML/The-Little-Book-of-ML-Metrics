"""Figures for the Bias & Fairness chapter, drawn at printed size.

One population, five lenses. Two groups of 100 people are drawn as grids of
squares colored by confusion-matrix cell; the outline marks the people the
metric conditions on and everyone else is faded. The model is the same on every
page: one threshold that selects 5 of every 6 qualified people (TPR = 5/6) and
1 in 10 unqualified people (FPR = 1/10) in both groups, whose base rates differ
(60% and 30% qualified). All counts are computed from those definitions.

Calibration within Groups uses the same class-conditional score distributions,
mapped to a probability that is calibrated for the two groups combined, so the
group-level miscalibration is the only effect shown.
Run: uv run python notebooks/fairness_figures.py [NAME ...]
"""
import sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.optimize import brentq
from scipy.stats import norm
from style import *  # noqa: F401,F403

N, COLS = 100, 10
BASE = {'A': Fraction(60, 100), 'B': Fraction(30, 100)}
TPR, FPR = Fraction(5, 6), Fraction(1, 10)
MU1, MU0 = norm.ppf(float(TPR)), norm.ppf(float(FPR))   # z | qualified, z | not; threshold at z = 0
SIZE, GAP = 1.0, 0.2
STEP = SIZE + GAP
FADE = 0.22
STYLE = {'TP': (NML_CYAN, NML_CYAN), 'FN': ('white', NML_CYAN), 'FP': (NML_RED, NML_RED), 'TN': (LIGHT, LIGHT)}
LABELS = {'TP': 'qualified, selected', 'FN': 'qualified, missed', 'FP': 'unqualified, selected',
          'TN': 'unqualified, not selected'}


def counts(group, tpr=TPR, fpr=FPR):
    pos = BASE[group] * N
    neg = N - pos
    tp, fp = tpr * pos, fpr * neg
    assert tp.denominator == 1 and fp.denominator == 1
    tp, fp, pos, neg = int(tp), int(fp), int(pos), int(neg)
    return dict(TP=tp, FN=pos - tp, FP=fp, TN=neg - fp, pos=pos, neg=neg)


cA, cB = counts('A'), counts('B')


def pct(a, b):
    return f'{100 * a / b:.0f}%'


def outline(ax, i0, i1):
    """Outline cells i0..i1 (row-major, inclusive)."""
    pad = GAP / 2
    r0, c0 = divmod(i0, COLS)
    r1, c1 = divmod(i1, COLS)
    X = lambda c: c * STEP - pad
    Yt = lambda r: -r * STEP + SIZE + pad
    Yb = lambda r: -r * STEP - pad
    if r0 == r1:
        pts = [(X(c0), Yt(r0)), (X(c1 + 1), Yt(r0)), (X(c1 + 1), Yb(r0)), (X(c0), Yb(r0))]
    else:
        pts = [(X(c0), Yt(r0)), (X(COLS), Yt(r0)), (X(COLS), Yb(r1 - 1))]
        pts += [(X(c1 + 1), Yb(r1 - 1)), (X(c1 + 1), Yb(r1))] if c1 < COLS - 1 else [(X(COLS), Yb(r1))]
        pts += [(X(0), Yb(r1)), (X(0), Yt(r0 + 1))]
        if c0 > 0:
            pts += [(X(c0), Yt(r0 + 1))]
    pts.append(pts[0])
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=INK, lw=1.0, solid_joinstyle='miter', zorder=4)


def grid(ax, c, order, highlight, outlines=None):
    kinds = [k for k in order for _ in range(c[k])]
    for i, k in enumerate(kinds):
        r, col = divmod(i, COLS)
        face, edge = STYLE[k]
        ax.add_patch(Rectangle((col * STEP, -r * STEP), SIZE, SIZE, facecolor=face, edgecolor=edge, lw=0.7,
                               alpha=1.0 if k in highlight else FADE, zorder=2))
    idx = [i for i, k in enumerate(kinds) if k in highlight]
    for i0, i1 in (outlines or ([(idx[0], idx[-1])] if len(idx) < N else [])):
        outline(ax, i0, i1)
    ax.set_xlim(-0.4, COLS * STEP)
    ax.set_ylim(-(N // COLS - 1) * STEP - 0.4, SIZE + 0.4)
    ax.set_aspect('equal')
    bare_axes(ax)


def legend(fig, faded=True, ncol=2):
    h = [Rectangle((0, 0), 1, 1, facecolor=STYLE[k][0], edgecolor=STYLE[k][1], lw=0.7, label=LABELS[k])
         for k in ('TP', 'FN', 'FP', 'TN')]
    if faded:
        h.append(Rectangle((0, 0), 1, 1, facecolor=NML_CYAN, alpha=FADE, edgecolor='none',
                           label='faded: outside the outline'))
    fig.legend(handles=h, loc='outside lower center', ncol=ncol, handlelength=0.9, handleheight=0.9,
               columnspacing=0.8, handletextpad=0.35)


def panels(width, n):
    fig, axes = book_figure(width, 3.0 if n == 2 else 2.45, 1, n)
    return fig, axes


# ===========================================================================
def demographic_parity():
    order, hi = ['TP', 'FP', 'FN', 'TN'], {'TP', 'FP'}
    selA, selB = cA['TP'] + cA['FP'], cB['TP'] + cB['FP']
    pB = float(BASE['B'])
    t = brentq(lambda z: pB * norm.sf(z - MU1) + (1 - pB) * norm.sf(z - MU0) - selA / N, -6, 6)
    tp2 = int(round(cB['pos'] * norm.sf(t - MU1)))
    fp2 = selA - tp2
    cB2 = dict(TP=tp2, FN=cB['pos'] - tp2, FP=fp2, TN=cB['neg'] - fp2, pos=cB['pos'], neg=cB['neg'])
    fig, axes = panels(1.0, 3)
    for ax, c, title in [(axes[0], cA, 'Group A'), (axes[1], cB, 'Group B'),
                         (axes[2], cB2, 'Group B, lower threshold')]:
        grid(ax, c, order, hi)
        sel = c['TP'] + c['FP']
        ax.set_title(f'{title}\n{c["pos"]} of 100 qualified', fontsize=TEXT_PT, linespacing=1.3)
        ax.set_xlabel(f'selected {sel} of 100 = {pct(sel, N)}\n{c["TP"]} qualified, {c["FP"]} not',
                      linespacing=1.35, color=INK)
    fig.canvas.draw()
    con = ConnectionPatch(xyA=(COLS * STEP + 0.2, -4.5), coordsA=axes[1].transData,
                          xyB=(-0.6, -4.5), coordsB=axes[2].transData, arrowstyle='-|>', color=REF,
                          lw=LW_THIN + 0.2, mutation_scale=7)
    fig.add_artist(con)
    legend(fig, ncol=3)
    save_figure(fig, 'Demographic_Parity')
    plt.close(fig)
    print(f'DP: selected A {selA} B {selB}; B lowered TP {tp2} FP {fp2} (threshold z = {t:.2f})')
    return cB2


def equality_of_opportunity():
    order, hi = ['TP', 'FN', 'FP', 'TN'], {'TP', 'FN'}
    fig, axes = panels(0.82, 2)
    for ax, c, name in [(axes[0], cA, 'A'), (axes[1], cB, 'B')]:
        grid(ax, c, order, hi)
        ax.set_title(f'Group {name}\n{c["pos"]} qualified, outlined', fontsize=TEXT_PT, linespacing=1.3)
        ax.set_xlabel(f'selected {c["TP"]} of {c["pos"]} qualified\nTPR = {pct(c["TP"], c["pos"])}',
                      linespacing=1.35, color=INK)
    legend(fig)
    save_figure(fig, 'Equality_of_Opportunity')
    plt.close(fig)
    print(f'EOpp: TPR A {cA["TP"] / cA["pos"]:.3f} B {cB["TP"] / cB["pos"]:.3f}')


def equality_of_odds():
    order = ['TP', 'FN', 'FP', 'TN']
    fig, axes = panels(0.82, 2)
    for ax, c, name in [(axes[0], cA, 'A'), (axes[1], cB, 'B')]:
        grid(ax, c, order, {'TP', 'FN', 'FP', 'TN'}, outlines=[(0, c['pos'] - 1), (c['pos'], N - 1)])
        ax.set_title(f'Group {name}\ntop: {c["pos"]} qualified; bottom: {c["neg"]} not', fontsize=TEXT_PT,
                     linespacing=1.3)
        ax.set_xlabel(f'TPR = {c["TP"]} of {c["pos"]} = {pct(c["TP"], c["pos"])}\n'
                      f'FPR = {c["FP"]} of {c["neg"]} = {pct(c["FP"], c["neg"])}', linespacing=1.35, color=INK)
    legend(fig, faded=False)
    save_figure(fig, 'Equality_of_Odds')
    plt.close(fig)


def predictive_parity():
    order, hi = ['TP', 'FP', 'FN', 'TN'], {'TP', 'FP'}
    fig, axes = panels(0.82, 2)
    ppv = {}
    for ax, c, name in [(axes[0], cA, 'A'), (axes[1], cB, 'B')]:
        sel = c['TP'] + c['FP']
        ppv[name] = c['TP'] / sel
        grid(ax, c, order, hi)
        ax.set_title(f'Group {name}\n{sel} selected, outlined', fontsize=TEXT_PT, linespacing=1.3)
        ax.set_xlabel(f'{c["TP"]} of {sel} qualified\nPPV = {num(100 * ppv[name], 1)}%', linespacing=1.35,
                      color=INK)
    legend(fig)
    save_figure(fig, 'Predictive_Parity')
    plt.close(fig)
    print('PPV', {k: round(v, 4) for k, v in ppv.items()})


# ===========================================================================
def calibration_within_groups():
    rng = np.random.default_rng(11)
    n = 200_000
    bins = np.linspace(0, 1, 11)
    # probability calibrated for both groups combined (equal group sizes: 45% qualified)
    pooled = np.log(0.45 / 0.55) - (MU1 ** 2 - MU0 ** 2) / 2
    frac = {}
    for g in ('A', 'B'):
        y = (rng.random(n) < float(BASE[g])).astype(int)
        z = np.where(y == 1, rng.normal(MU1, 1, n), rng.normal(MU0, 1, n))
        s = 1 / (1 + np.exp(-(pooled + (MU1 - MU0) * z)))
        idx = np.clip(np.digitize(s, bins) - 1, 0, 9)
        frac[g] = np.array([y[idx == b].mean() for b in range(10)])
    fig = book_canvas(1.0, 2.2)
    ax = fig.add_axes([0.0, 0.2, 1.0, 0.68])
    size, gap, pair_gap, bin_w = 0.6, 0.12, 0.28, 2.4
    step = size + gap
    for b in range(10):
        xb = b * bin_w
        for j, (g, col) in enumerate([('A', NML_CYAN), ('B', NML_PURPLE)]):
            x = xb + j * (step + pair_gap)
            k = int(round(10 * frac[g][b]))
            for i in range(10):
                ax.add_patch(Rectangle((x, i * step), size, size, facecolor=col if i < k else 'white',
                                       edgecolor=col if i < k else REF, lw=0.6, zorder=2))
        mid = (bins[b] + bins[b + 1]) / 2
        yl = mid * 10 * step - gap / 2
        ax.plot([xb - 0.2, xb + 2 * step + pair_gap + 0.08], [yl, yl], color=INK, lw=LW_THIN + 0.3, zorder=3)
        pair_w = 2 * step + pair_gap - gap
        edge = xb - (bin_w - pair_w) / 2
        for e, lab in ([(edge, f'{bins[b]:.1f}')] + ([(edge + bin_w, '1.0')] if b == 9 else [])):
            ax.plot([e, e], [-0.25, -0.05], color=REF, lw=0.6)
            ax.text(e, -0.35, lab, ha='center', va='top', color=MUTED, fontsize=SMALL_PT)
    b5 = 5
    xb = b5 * bin_w
    ax.annotate(f'A: {frac["A"][b5]:.0%}\nB: {frac["B"][b5]:.0%}', xy=(xb + step + pair_gap / 2, 10 * step),
                xytext=(xb + step, 10 * step + 1.35), ha='center', va='bottom', color=INK, linespacing=1.2,
                fontsize=SMALL_PT, arrowprops=dict(arrowstyle='-', color=REF, lw=LW_THIN))
    ax.set_xlim(-1.0, 10 * bin_w - 0.6)
    ax.set_ylim(-1.0, 10 * step + 2.6)
    ax.set_aspect('equal')
    bare_axes(ax)
    fig.text(0.0, 0.99, 'share of people in each score range who are qualified (one filled square = 10%)',
             va='top', color=INK)
    fig.text(0.5, 0.155, 'model score range', ha='center', va='top', color=INK)
    handles = [Rectangle((0, 0), 1, 1, facecolor=NML_CYAN, label='Group A'),
               Rectangle((0, 0), 1, 1, facecolor=NML_PURPLE, label='Group B'),
               Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=REF, lw=0.6, label='not qualified'),
               Line2D([], [], color=INK, lw=LW_THIN + 0.3, label='calibrated level (range midpoint)')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, handlelength=1.0,
               handleheight=0.9, columnspacing=0.9)
    save_figure(fig, 'Calibration_within_Groups', crop=False)
    plt.close(fig)
    print('Calibration A', np.round(100 * frac['A'], 1), 'B', np.round(100 * frac['B'], 1))


FIGURES = {'Demographic_Parity': demographic_parity, 'Equality_of_Opportunity': equality_of_opportunity,
           'Equality_of_Odds': equality_of_odds, 'Predictive_Parity': predictive_parity,
           'Calibration_within_Groups': calibration_within_groups}

if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
