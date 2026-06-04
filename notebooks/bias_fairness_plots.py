"""Generate figures for the Bias & Fairness chapter.

Every figure shows the metric's *mechanism* and computes every displayed rate
from a small synthetic labeled dataset (fixed seed) — no hardcoded numbers.

Two groups (A, B) throughout. Each person has a true label Y (1 = "qualified"
/ positive) and a model score in [0, 1]; binary predictions come from a
threshold. The scenarios are tuned only via base rates and score separation;
all selection rates, TPR, FPR, PPV and calibration curves are measured.

Run from notebooks/:  uv run python bias_fairness_plots.py
"""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve

SEED = 7
RNG = np.random.default_rng(SEED)

GREY = '#c9c9c9'        # "absent" / not-caught
GREY_LINE = '#9a9a9a'   # subtle reference lines
N = 6000               # samples per group — large enough for stable rates


def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


def gen(n, base_rate, sep, rng):
    """A synthetic group. Y ~ Bernoulli(base_rate); a latent value separates
    the classes by `sep` standard deviations; score = sigmoid(latent) in [0,1]."""
    y = (rng.random(n) < base_rate).astype(int)
    latent = np.where(y == 1, rng.normal(sep, 1.0, n), rng.normal(-sep, 1.0, n))
    score = 1.0 / (1.0 + np.exp(-latent))
    return y, score


def rates(y, score, thr):
    """Confusion-derived rates at a decision threshold."""
    pred = score >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    sel = (tp + fp) / len(y)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, tpr=tpr, fpr=fpr, ppv=ppv, sel=sel)


def pct(x):
    return f'{round(100 * x)}\\%'


# ===========================================================================
# 1. Demographic Parity — equal selection RATE, unequal MERIT
#    Left:  both groups selected at the same rate (parity satisfied).
#    Right: the selected sets, split qualified / unqualified. Group B's
#           qualified pool is smaller, so meeting the same rate forces the
#           model to select unqualified people. PPV of the selected set is
#           measured, never assumed.
# ===========================================================================
SEL = 0.40
yA, sA = gen(N, 0.55, 1.0, RNG)   # group A: 55% truly qualified
yB, sB = gen(N, 0.25, 1.0, RNG)   # group B: 25% truly qualified

# per-group threshold at the (1 - SEL) score quantile  ->  exactly SEL selected
tA = np.quantile(sA, 1 - SEL)
tB = np.quantile(sB, 1 - SEL)
selA = sA >= tA
selB = sB >= tB
rateA, rateB = selA.mean(), selB.mean()
qualA = (selA & (yA == 1)).mean()      # qualified, selected   (cyan)
unqA = (selA & (yA == 0)).mean()       # unqualified, selected (red)
qualB = (selB & (yB == 1)).mean()
unqB = (selB & (yB == 0)).mean()
ppvA = qualA / (qualA + unqA)
ppvB = qualB / (qualB + unqB)
baseA, baseB = yA.mean(), yB.mean()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))
xpos = [0, 1]
labels = ['Group A', 'Group B']

# left: selection rate
axL.bar(xpos, [rateA, rateB], width=0.55, color=start_color,
        edgecolor='white', linewidth=2, zorder=3)
axL.axhline(SEL, color=GREY_LINE, lw=1.4, ls=(0, (5, 3)), zorder=2)
axL.text(1.46, SEL, f'equal rate\n(parity ✓)', va='center', ha='left',
         fontsize=12, color='#6f6f6f')
for x, r in zip(xpos, [rateA, rateB]):
    axL.text(x, r + 0.012, pct(r).replace('\\', ''), ha='center', va='bottom',
             fontsize=14, fontweight='bold', color=start_color)
axL.set_title('What parity checks: selection rate', fontsize=15, pad=10)
axL.set_ylabel('selected  ($\\hat{Y}=1$)  /  group')
axL.set_ylim(0, 0.62)
axL.set_xlim(-0.6, 2.0)
axL.set_xticks(xpos)
axL.set_xticklabels(labels, fontsize=13)
axL.set_yticks([0, 0.2, 0.4, 0.6])
axL.set_yticklabels(['0%', '20%', '40%', '60%'])
despine(axL, keep=('left',))

# right: composition of the selected set
axR.bar(xpos, [qualA, qualB], width=0.55, color=start_color,
        edgecolor='white', linewidth=2, zorder=3, label='qualified  (Y = 1)')
axR.bar(xpos, [unqA, unqB], width=0.55, bottom=[qualA, qualB], color=end_color,
        edgecolor='white', linewidth=2, zorder=3, label='unqualified  (Y = 0)')
for x, base in zip(xpos, [baseA, baseB]):
    axR.plot([x - 0.33, x + 0.33], [base, base], color='#2a2a2a', lw=1.6, zorder=5)
    axR.text(x + 0.36, base, 'truly\nqualified', va='center', ha='left',
             fontsize=9.5, color='#2a2a2a')
for x, q, u in zip(xpos, [qualA, qualB], [unqA, unqB]):
    axR.text(x, q + u + 0.012, f'{q/(q+u):.0%} qualified', ha='center', va='bottom',
             fontsize=12.5, fontweight='bold', color='#2a2a2a')
axR.set_title('What parity ignores: who was selected', fontsize=15, pad=10)
axR.set_ylabel('share of the group')
axR.set_ylim(0, 0.62)
axR.set_xlim(-0.6, 2.0)
axR.set_xticks(xpos)
axR.set_xticklabels(labels, fontsize=13)
axR.set_yticks([0, 0.2, 0.4, 0.6])
axR.set_yticklabels(['0%', '20%', '40%', '60%'])
axR.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2,
           frameon=False, fontsize=11.5)
despine(axR, keep=('left',))

fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.16, wspace=0.3)
save_figure(fig, 'Demographic_Parity')
plt.close()
print(f"1. DP: rate A={rateA:.3f} B={rateB:.3f} | PPV(selected) A={ppvA:.3f} B={ppvB:.3f} "
      f"| base A={baseA:.3f} B={baseB:.3f}")


# ===========================================================================
# 2. Equality of Opportunity — restrict to Y = 1, compare TPR
#    Each bar is the truly-qualified subpopulation of a group; the filled part
#    is the fraction the model correctly identifies (TPR). Fair = equal fill.
# ===========================================================================
def eo_panel(ax, groups, thr, title, link):
    bars_tpr = []
    for i, (y, s) in enumerate(groups):
        r = rates(y, s, thr)
        bars_tpr.append(r['tpr'])
        ax.bar(i, r['tpr'], width=0.55, color=start_color, edgecolor='white',
               linewidth=2, zorder=3)
        ax.bar(i, 1 - r['tpr'], width=0.55, bottom=r['tpr'], color=GREY,
               edgecolor='white', linewidth=2, zorder=3)
        ax.text(i, r['tpr'] / 2, f'caught\n{r["tpr"]:.0%}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        ax.text(i, r['tpr'] + (1 - r['tpr']) / 2, 'missed', ha='center', va='center',
                fontsize=11, color='#6f6f6f')
    if link:
        ax.plot([0, 1], [bars_tpr[0], bars_tpr[1]], color=GREY_LINE, lw=1.4,
                ls=(0, (5, 3)), zorder=4)
        ax.text(0.5, 1.13, 'equal TPR ✓', ha='center', va='center',
                fontsize=12, color='#6f6f6f')
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_ylim(0, 1.22)
    ax.set_xlim(-0.7, 1.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Group A', 'Group B'], fontsize=13)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0%', '50%', '100%'])
    despine(ax, keep=('left',))
    return bars_tpr


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))
# fair: same model quality for both groups
fairA = gen(N, 0.5, 1.1, RNG)
fairB = gen(N, 0.5, 1.1, RNG)
tpr_fair = eo_panel(axL, [fairA, fairB], 0.5, 'Fair: qualified people caught equally', True)
# unfair: model is much weaker for group B
unfA = gen(N, 0.5, 1.4, RNG)
unfB = gen(N, 0.5, 0.35, RNG)
tpr_unf = eo_panel(axR, [unfA, unfB], 0.5, 'Unfair: group B routinely overlooked', False)
gap = abs(tpr_unf[0] - tpr_unf[1])
axR.annotate('', xy=(1.34, tpr_unf[1]), xytext=(1.34, tpr_unf[0]),
             arrowprops=dict(arrowstyle='<->', color=end_color, lw=1.8))
axR.text(1.40, (tpr_unf[0] + tpr_unf[1]) / 2, f'{gap:.0%}\ngap', va='center',
         ha='left', fontsize=12, color=end_color, fontweight='bold')
axR.set_xlim(-0.7, 1.95)
axL.set_ylabel('of the truly qualified (Y = 1)')
fig.suptitle('Restricted to the truly qualified (Y = 1): what share does the model identify?',
             fontsize=13, y=0.99, color='#555555')
fig.subplots_adjust(left=0.075, right=0.97, top=0.86, bottom=0.1, wspace=0.25)
save_figure(fig, 'Equality_of_Opportunity')
plt.close()
print(f"2. EOpp: fair TPR A={tpr_fair[0]:.3f} B={tpr_fair[1]:.3f} | "
      f"unfair TPR A={tpr_unf[0]:.3f} B={tpr_unf[1]:.3f}")


# ===========================================================================
# 3. Equality of Odds — same operating point in ROC space
#    Each group's ROC curve plus its (FPR, TPR) operating point. Fair = both
#    land on the same point; unfair = different points.
# ===========================================================================
def roc_panel(ax, groups, thr, title, same_curve_note):
    pts = []
    for (y, s), color, name in zip(groups, [start_color, middle_color], ['Group A', 'Group B']):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=color, lw=3.5, solid_capstyle='round', zorder=3, label=name)
        r = rates(y, s, thr)
        pts.append((r['fpr'], r['tpr'], color, name))
    ax.plot([0, 1], [0, 1], color=GREY_LINE, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for fpr, tpr, color, name in pts:
        ax.scatter(fpr, tpr, s=150, color=color, edgecolor='white', linewidth=1.8, zorder=5)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel('false positive rate')
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    despine(ax)
    return pts


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))
# fair: identical score distributions, one shared threshold -> same point
fA = gen(N, 0.4, 1.0, RNG)
fB = gen(N, 0.4, 1.0, RNG)
pf = roc_panel(axL, [fA, fB], 0.5, 'Fair: groups meet at one operating point', True)
axL.annotate('both groups land\nat the same point',
             xy=(pf[0][0], pf[0][1]), xytext=(pf[0][0] + 0.16, pf[0][1] - 0.20),
             fontsize=11, color='#444444',
             arrowprops=dict(arrowstyle='->', color='#888888', lw=1.3))
axL.legend(loc='lower right', fontsize=12, frameon=False)
axL.set_ylabel('true positive rate')
# unfair: B's score is less informative; the two points diverge
uA = gen(N, 0.4, 1.3, RNG)
uB = gen(N, 0.4, 0.45, RNG)
pu = roc_panel(axR, [uA, uB], 0.5, 'Unfair: groups sit at different points', False)
for fpr, tpr, color, name in pu:
    axR.annotate(f'{name}\n({fpr:.2f}, {tpr:.2f})', xy=(fpr, tpr),
                 xytext=(fpr + 0.10, tpr - 0.13 if name == 'Group A' else tpr + 0.02),
                 fontsize=11, color=color, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
axR.legend(loc='lower right', fontsize=12, frameon=False)
fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.12, wspace=0.2)
save_figure(fig, 'Equality_of_Odds')
plt.close()
print(f"3. EOdds: fair A=({pf[0][0]:.3f},{pf[0][1]:.3f}) B=({pf[1][0]:.3f},{pf[1][1]:.3f}) | "
      f"unfair A=({pu[0][0]:.3f},{pu[0][1]:.3f}) B=({pu[1][0]:.3f},{pu[1][1]:.3f})")


# ===========================================================================
# 4. Predictive Parity — restrict to Ŷ = 1, compare PPV (horizontal bars)
#    Each bar is the flagged subpopulation; the filled part is the fraction
#    that is truly positive (PPV). Horizontal, to mirror-but-distinguish from
#    Equality of Opportunity's vertical Y=1 view.
# ===========================================================================
def pp_panel(ax, groups, thr, title, link):
    ppvs = []
    for i, (y, s) in enumerate(groups):
        r = rates(y, s, thr)
        ppvs.append(r['ppv'])
        ax.barh(i, r['ppv'], height=0.55, color=start_color, edgecolor='white',
                linewidth=2, zorder=3)
        ax.barh(i, 1 - r['ppv'], height=0.55, left=r['ppv'], color=end_color,
                edgecolor='white', linewidth=2, zorder=3)
        ax.text(r['ppv'] / 2, i, f'correct\n{r["ppv"]:.0%}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        ax.text(r['ppv'] + (1 - r['ppv']) / 2, i, 'wrong', ha='center', va='center',
                fontsize=11, color='white')
    if link:
        ax.plot([ppvs[0], ppvs[1]], [0, 1], color=GREY_LINE, lw=1.4,
                ls=(0, (5, 3)), zorder=4)
        ax.text(min(ppvs) - 0.02, 0.5, 'equal PPV ✓', ha='right', va='center',
                fontsize=12, color='#6f6f6f')
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Group A', 'Group B'], fontsize=13)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['0%', '50%', '100%'])
    despine(ax, keep=('bottom',))
    return ppvs


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0))
# fair: same base rate + model -> same PPV
pfA = gen(N, 0.45, 1.0, RNG)
pfB = gen(N, 0.45, 1.0, RNG)
ppv_fair = pp_panel(axL, [pfA, pfB], 0.5, 'Fair: a "yes" is equally trustworthy', True)
# unfair: B has a low base rate -> a positive prediction is less often correct
puA = gen(N, 0.55, 1.0, RNG)
puB = gen(N, 0.18, 1.0, RNG)
ppv_unf = pp_panel(axR, [puA, puB], 0.5, 'Unfair: a "yes" means less for group B', False)
axL.set_xlabel('of everyone flagged ($\\hat{Y}=1$)')
axR.set_xlabel('of everyone flagged ($\\hat{Y}=1$)')
fig.suptitle('Restricted to the flagged ($\\hat{Y}=1$): what share is truly positive (PPV)?',
             fontsize=13, y=1.0, color='#555555')
fig.subplots_adjust(left=0.085, right=0.97, top=0.82, bottom=0.14, wspace=0.25)
save_figure(fig, 'Predictive_Parity')
plt.close()
print(f"4. PP: fair PPV A={ppv_fair[0]:.3f} B={ppv_fair[1]:.3f} | "
      f"unfair PPV A={ppv_unf[0]:.3f} B={ppv_unf[1]:.3f}")


# ===========================================================================
# 5. Calibration within Groups — reliability diagram per group
#    Group A: y ~ Bernoulli(score)        -> calibrated, on the diagonal.
#    Group B: y ~ Bernoulli(0.6*score+0.2) -> the score over/under-states risk,
#    so the curve bows off the diagonal. Observed frequencies are binned from
#    data (sklearn.calibration_curve).
# ===========================================================================
nC = 16000
scoreA = RNG.random(nC)
yA_c = (RNG.random(nC) < scoreA).astype(int)
scoreB = RNG.random(nC)
trueB = 0.6 * scoreB + 0.2                     # what the score should have said
yB_c = (RNG.random(nC) < trueB).astype(int)

fracA, meanA = calibration_curve(yA_c, scoreA, n_bins=10, strategy='uniform')
fracB, meanB = calibration_curve(yB_c, scoreB, n_bins=10, strategy='uniform')

fig, ax = plt.subplots(figsize=(8.2, 7.0))
ax.plot([0, 1], [0, 1], color=GREY_LINE, lw=1.3, ls=(0, (4, 3)), zorder=1,
        label='perfectly calibrated')
ax.plot(meanA, fracA, 'o-', color=start_color, lw=3, markersize=8,
        solid_capstyle='round', zorder=4, label='Group A — calibrated')
ax.plot(meanB, fracB, 's-', color=end_color, lw=3, markersize=8,
        solid_capstyle='round', zorder=4, label='Group B — miscalibrated')
ax.legend(loc='lower right', fontsize=12.5, frameon=False)

# annotate the meaning gap at the bin nearest score 0.3
bi = int(np.argmin(np.abs(meanB - 0.3)))
sx, sy = meanB[bi], fracB[bi]
ax.plot([sx, sx], [sx, sy], color='#444444', lw=1.4, zorder=5)
ax.scatter([sx], [sy], s=70, color=end_color, edgecolor='white', linewidth=1.4, zorder=6)
ax.annotate(f'a score of {sx:.2f} means a\n{sy:.0%} true rate for Group B',
            xy=(sx, sy), xytext=(0.05, 0.66), fontsize=11.5, color='#444444', ha='left',
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.3))

ax.set_xlabel('predicted score')
ax.set_ylabel('observed frequency of $Y=1$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
despine(ax)
fig.tight_layout()
save_figure(fig, 'Calibration_within_Groups')
plt.close()
print(f"5. Cal: group B score~0.3 -> observed {sy:.3f}")

print("\nAll bias & fairness figures regenerated.")
