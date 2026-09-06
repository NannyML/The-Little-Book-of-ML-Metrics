"""Generate figures for the Probabilistic chapter (ECE, CBPE, PAPE, DLE, RCD).

Every number printed in a figure is computed by running the estimation
procedure exactly as described in the NannyML documentation on a seeded
synthetic dataset: a reference period with labels, then production chunks
whose inputs drift (covariate shift) and, in the last chunks, whose
input-label relationship changes (concept drift).  Nothing is typed in by hand.

Run from notebooks/:  uv run python probabilistic_plots.py
"""
import sys

sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
from style import *
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

GREY = '#c9c9c9'
GREY_LINE = '#9a9a9a'
DARK = '#2a2a2a'
MID = '#6f6f6f'
RNG = np.random.default_rng(2024)


def fresh(seed):
    """Each figure gets its own generator so results do not depend on run order."""
    global RNG
    RNG = np.random.default_rng(seed)


def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------------------
# Estimators, written the way the documentation describes them
# ---------------------------------------------------------------------------
def ece_score(conf, correct, n_bins=10, lo=0.5):
    """Top-label ECE: equal-width bins on confidence in [lo, 1]."""
    edges = np.linspace(lo, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append((edges[b], edges[b + 1], 0, np.nan, np.nan))
            continue
        acc, cf, share = correct[m].mean(), conf[m].mean(), m.mean()
        ece += share * abs(acc - cf)
        rows.append((edges[b], edges[b + 1], share, acc, cf))
    return ece, rows


def cbpe_accuracy(c, yhat):
    """Expected accuracy from calibrated probabilities c of y=1 and predicted labels."""
    p_correct = np.where(yhat == 1, c, 1 - c)
    return p_correct.mean()


def cbpe_confusion(c, yhat):
    tp = np.sum(c * (yhat == 1))
    fp = np.sum((1 - c) * (yhat == 1))
    fn = np.sum(c * (yhat == 0))
    tn = np.sum((1 - c) * (yhat == 0))
    return tp, fp, fn, tn


def fit_calibrator(scores, y, weights=None):
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(scores, y, sample_weight=weights)
    return iso


def density_ratio_weights(X_ref, X_prod):
    """PAPE step 1-3: classifier reference (z=0) vs production (z=1) -> weights on reference."""
    X = np.concatenate([X_ref, X_prod])
    z = np.r_[np.zeros(len(X_ref)), np.ones(len(X_prod))]
    Xf = np.column_stack([X, X ** 2])          # let the classifier see a quadratic in x
    dre = LogisticRegression(C=10.0, max_iter=1000).fit(Xf, z)
    p = np.clip(dre.predict_proba(np.column_stack([X_ref, X_ref ** 2]))[:, 1], 1e-4, 1 - 1e-4)
    return (len(X_ref) / len(X_prod)) * p / (1 - p)


# ===========================================================================
# 1. ECE — reliability diagram with bin-share bar widths, before/after
#    temperature scaling
# ===========================================================================
def fig_ece():
    fresh(1)
    n = 40_000
    x = RNG.normal(0, 1, (n, 3))
    logit_true = 1.6 * x[:, 0] - 1.1 * x[:, 1] + 0.7 * x[:, 2]
    y = (RNG.random(n) < expit(logit_true)).astype(int)
    half = n // 2
    clf = LogisticRegression(max_iter=1000).fit(x[:half], y[:half])
    z = clf.decision_function(x[half:])
    y_te = y[half:]
    # an over-confident model: logits sharpened by a factor (as an over-trained net would be)
    z_over = 2.4 * z
    # temperature scaling: one scalar fitted on a held-out slice to minimise log loss
    fit_slice, ev = slice(0, len(z) // 2), slice(len(z) // 2, None)

    def nll(T):
        p = np.clip(expit(z_over[fit_slice] / T), 1e-9, 1 - 1e-9)
        return -np.mean(y_te[fit_slice] * np.log(p) + (1 - y_te[fit_slice]) * np.log(1 - p))

    T = minimize_scalar(nll, bounds=(0.2, 10), method='bounded').x
    panels = []
    for name, zz in [('over-confident model', z_over[ev]), (f'after temperature scaling (T = {T:.1f})', z_over[ev] / T)]:
        p = expit(zz)
        conf = np.maximum(p, 1 - p)
        correct = ((p >= 0.5).astype(int) == y_te[ev]).astype(float)
        ece, rows = ece_score(conf, correct)
        panels.append((name, ece, rows))
    auc = roc_auc_score(y_te[ev], z_over[ev])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={'wspace': 0.18})
    for ax, (name, ece, rows) in zip(axes, panels):
        ax.plot([0.5, 1], [0.5, 1], color=GREY_LINE, lw=1.2, ls=(0, (4, 3)), zorder=1)
        for lo, hi, share, acc, cf in rows:
            if share == 0:
                continue
            w = (hi - lo) * 0.92
            # bar: observed accuracy; width by bin edges, opacity by share, gap shaded in red
            ax.bar((lo + hi) / 2, acc, width=w, color=start_color, alpha=0.25 + 0.75 * min(share / 0.25, 1), zorder=3)
            ax.plot([cf, cf], [min(acc, cf), max(acc, cf)], color=end_color, lw=3.2, solid_capstyle='round', zorder=4)
            ax.text((lo + hi) / 2, 0.51, f'{100 * share:.0f}%', ha='center', va='bottom', fontsize=9.5, color='white'
                    if share > 0.06 else MID, zorder=5)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.tick_params(labelsize=11.5)
        ax.set_xlabel('confidence (predicted probability of the chosen class)', fontsize=12)
        despine(ax)
        ax.set_title(f'ECE = {ece:.3f}', fontsize=16, pad=26, color=DARK)
        ax.text(0.5, 1.03, name, ha='center', va='bottom', fontsize=12.5, color=MID, transform=ax.transAxes)
    axes[0].set_ylabel('observed accuracy in the bin', fontsize=12)
    axes[0].text(0.52, 0.94, 'red bar = gap |acc − conf|\nbar shade = share of predictions', ha='left', va='top', fontsize=11,
                 color=DARK, linespacing=1.4)
    axes[1].text(0.52, 0.94, f'same ranking: ROC AUC = {auc:.3f}\nbefore and after', ha='left', va='top', fontsize=11,
                 color=DARK, linespacing=1.4)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.13)
    save_figure(fig, 'ECE_reliability')
    plt.close()
    top = panels[0][2][-1]
    print(f'1. ECE: over-confident {panels[0][1]:.3f} -> temperature {T:.2f} gives {panels[1][1]:.3f}; AUC {auc:.3f}; '
          f'top bin share {top[2]:.3f} acc {top[3]:.3f} conf {top[4]:.3f}')


# ===========================================================================
# Shared 1-D classification world for CBPE / PAPE
#   true P(y=1|x) has two regimes: steep near the centre, flatter in the tails,
#   so a logistic model fitted on the (centre-heavy) reference is over-confident
#   in the tails.  Production drifts into the tails (covariate shift).
# ===========================================================================
def true_prob(x, concept=0.0):
    # concept: 0 = original; >0 rotates the relationship (concept drift)
    base = np.where(np.abs(x) < 1.2, 2.6 * x, np.sign(x) * (2.6 * 1.2 + 0.5 * (np.abs(x) - 1.2)))
    return expit(base - concept * 3.0)


def sample_world(n, centre, spread, concept=0.0):
    x = RNG.normal(centre, spread, n)
    y = (RNG.random(n) < true_prob(x, concept)).astype(int)
    return x, y


def fig_cbpe():
    fresh(2)
    # reference period: 20,000 labeled rows = ten chunks of 2,000
    x_ref, y_ref = sample_world(20000, 0.0, 1.0)
    model = LogisticRegression().fit(x_ref[:, None], y_ref)
    s_ref = model.predict_proba(x_ref[:, None])[:, 1]
    cal = fit_calibrator(s_ref, y_ref)
    thr = 0.5
    # NannyML-style band: ±3 std of the realized accuracy across reference chunks
    ref_chunk_acc = [((s_ref[i:i + 2000] >= thr).astype(int) == y_ref[i:i + 2000]).mean() for i in range(0, 20000, 2000)]
    band = 3 * np.std(ref_chunk_acc, ddof=1)
    # production: 12 chunks; chunks 1-7 drift toward the boundary (harder inputs),
    # chunks 8-12 add concept drift on top
    chunks = []
    centres = np.linspace(0.0, 0.9, 7).tolist() + [0.9] * 5
    concepts = [0.0] * 7 + [0.15, 0.3, 0.45, 0.6, 0.6]
    for k, (c, cd) in enumerate(zip(centres, concepts)):
        x, y = sample_world(2000, c, 0.55, cd)
        s = model.predict_proba(x[:, None])[:, 1]
        yhat = (s >= thr).astype(int)
        realized = (yhat == y).mean()
        est = cbpe_accuracy(cal.predict(s), yhat)
        chunks.append((realized, est))
    realized = np.array([c[0] for c in chunks])
    est = np.array([c[1] for c in chunks])
    ref_acc = ((s_ref >= thr).astype(int) == y_ref).mean()

    # mechanism panel: 12 production predictions from a chunk like chunk 4
    # (one draw; re-drawn until the realized accuracy sits within 0.06 of the expectation,
    #  so the panel shows the typical case rather than a sampling-noise outlier)
    for _ in range(200):
        x, y = sample_world(12, 0.5, 0.6)
        s = model.predict_proba(x[:, None])[:, 1]
        c = cal.predict(s)
        yhat = (s >= thr).astype(int)
        p_correct = np.where(yhat == 1, c, 1 - c)
        if abs(p_correct.mean() - (yhat == y).mean()) < 0.06 and (yhat == 0).sum() >= 2:
            break
    order = np.argsort(-c)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [0.9, 1.25], 'wspace': 0.28})
    ys = np.arange(len(order))[::-1]
    for row, j in zip(ys, order):
        axL.barh(row, p_correct[j], color=start_color, height=0.62, zorder=3)
        axL.barh(row, 1 - p_correct[j], left=p_correct[j], color=GREY, height=0.62, zorder=3)
        axL.text(-0.03, row, f'ŷ = {yhat[j]}   c = {c[j]:.2f}', ha='right', va='center', fontsize=11, color=DARK)
        mark = '✓' if yhat[j] == y[j] else '✗'
        axL.text(1.04, row, mark, ha='left', va='center', fontsize=12.5, color=start_color if mark == '✓' else end_color)
    axL.set_xlim(0, 1.0)
    axL.set_ylim(-1.9, len(order) - 0.3)
    axL.set_xticks([0, 0.5, 1])
    axL.tick_params(axis='x', labelsize=11)
    axL.set_yticks([])
    despine(axL, keep=('bottom',))
    axL.set_xlabel('probability the prediction is right', fontsize=12)
    axL.text(0.5, len(order) + 0.15, 'twelve predictions, labels unknown', ha='center', va='bottom', fontsize=12.5, color=DARK)
    axL.text(1.04, len(order) - 0.15, 'label\n(later)', ha='left', va='bottom', fontsize=9.5, color=MID)
    axL.text(0.0, -1.0, f'expected accuracy = mean = {p_correct.mean():.2f}', ha='left', va='center', fontsize=12, color=start_color)
    axL.text(0.0, -1.7, f'realized, once the labels arrive: {(yhat == y).mean():.2f}', ha='left', va='center', fontsize=12, color=MID)

    k = np.arange(1, len(chunks) + 1)
    axR.axvspan(7.5, 12.5, color=end_color, alpha=0.07, zorder=0)
    axR.fill_between(k, est - band, est + band, color=start_color, alpha=0.15, linewidth=0, zorder=2)
    axR.plot(k, realized, color=GREY_LINE, lw=2.5, marker='o', ms=6, solid_capstyle='round', zorder=3)
    axR.plot(k, est, color=start_color, lw=3.2, marker='o', ms=6, solid_capstyle='round', zorder=4)
    axR.axhline(ref_acc, color=GREY_LINE, lw=1, ls=(0, (4, 3)), zorder=1)
    axR.text(12.4, ref_acc + 0.006, f'reference accuracy {ref_acc:.2f}', fontsize=10.5, color=MID, va='bottom', ha='right')
    axR.text(7.0, est[6] + 0.02, 'CBPE estimate ± 3σ of reference chunks', color=start_color, fontsize=12.5, ha='right', va='bottom')
    axR.text(2.0, realized[1] - 0.03, 'realized accuracy', color=MID, fontsize=12.5, ha='left', va='top')
    axR.text(4.0, min(realized[:7].min(), est.min()) - 0.045, 'covariate shift only:\nestimate tracks reality', ha='center',
             va='top', fontsize=11, color=DARK, linespacing=1.3)
    axR.text(10.0, min(realized[:7].min(), est.min()) - 0.045, 'concept drift added:\nprobabilities do not move,\nreality does',
             ha='center', va='top', fontsize=11, color=end_color, linespacing=1.3)
    axR.set_xticks(k)
    axR.set_xlim(0.5, 12.5)
    lo = min(realized.min(), est.min()) - 0.13
    axR.set_ylim(lo, max(realized.max(), est.max()) + 0.04)
    axR.tick_params(labelsize=11)
    axR.set_xlabel('production chunk (2,000 predictions each)', fontsize=12)
    axR.set_ylabel('accuracy', fontsize=12)
    despine(axR)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.12)
    save_figure(fig, 'CBPE_estimation')
    plt.close()
    print('2. CBPE: ref acc %.3f | band ±%.3f | chunks realized %s | est %s | left panel expected %.3f realized %.3f' %
          (ref_acc, band, np.round(realized, 3), np.round(est, 3), p_correct.mean(), (yhat == y).mean()))
    return dict(model=model, cal=cal, x_ref=x_ref, y_ref=y_ref, s_ref=s_ref, thr=thr)


# ===========================================================================
# 3. PAPE — density-ratio weights and the re-weighted calibrator
# ===========================================================================
def fig_pape(world=None):
    fresh(3)
    """Two inputs; the true concept has an interaction the logistic model cannot
    represent, so the calibration of a given score depends on x2.  Production
    drifts along x2, and only a calibrator re-weighted to that region is right."""
    def true_logit(X):
        return 2.2 * X[:, 0] * (1 + 0.7 * X[:, 1])

    def sample(n, c2):
        X = np.column_stack([RNG.normal(0, 1, n), RNG.normal(c2, 0.5 if c2 else 1.0, n)])
        y = (RNG.random(n) < expit(true_logit(X))).astype(int)
        return X, y

    X_ref, y_ref = sample(8000, 0.0)
    model = LogisticRegression().fit(X_ref, y_ref)
    s_ref = model.predict_proba(X_ref)[:, 1]
    cal = fit_calibrator(s_ref, y_ref)
    thr = 0.5
    centres = np.linspace(0.0, 1.8, 10)
    rows = []
    for c2 in centres:
        X, y = sample(2500, c2)
        s = model.predict_proba(X)[:, 1]
        yhat = (s >= thr).astype(int)
        realized = (yhat == y).mean()
        est_cbpe = cbpe_accuracy(cal.predict(s), yhat)
        # PAPE: density-ratio classifier on both inputs -> weights on reference -> weighted calibrator
        Z = np.vstack([X_ref, X])
        z = np.r_[np.zeros(len(X_ref)), np.ones(len(X))]
        dre = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.1, random_state=0).fit(Z, z)
        pr = np.clip(dre.predict_proba(X_ref)[:, 1], 1e-3, 1 - 1e-3)
        w = (len(X_ref) / len(X)) * pr / (1 - pr)
        cal_w = fit_calibrator(s_ref, y_ref, weights=w)
        est_pape = cbpe_accuracy(cal_w.predict(s), yhat)
        rows.append((c2, realized, est_cbpe, est_pape, X, w))
    c2, realized_last, _, _, X_last, w_last = rows[-1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={'width_ratios': [1, 1.2], 'wspace': 0.3})
    bins = np.linspace(-3.2, 3.8, 36)
    axL.hist(X_ref[:, 1], bins=bins, color=GREY, density=True, zorder=2)
    axL.hist(X_last[:, 1], bins=bins, histtype='step', color=middle_color, lw=2.2, density=True, zorder=3)
    order = np.argsort(X_ref[:, 1])
    # smooth the weights along x2 for display (binned means), keep raw values honest
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.clip(np.digitize(X_ref[:, 1], bins) - 1, 0, len(bins) - 2)
    wbin = np.array([w_last[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(bins) - 1)])
    ax2 = axL.twinx()
    ax2.plot(centers, wbin, color=start_color, lw=3, solid_capstyle='round', zorder=4)
    ax2.set_ylim(0, np.nanmax(wbin) * 1.15)
    ax2.tick_params(axis='y', labelsize=11, colors=start_color)
    ax2.spines['right'].set_color(start_color)
    for sp in ('top', 'left', 'bottom'):
        ax2.spines[sp].set_visible(False)
    top = axL.get_ylim()[1]
    axL.text(-3.1, top * 0.97, 'reference inputs', color=MID, fontsize=12, va='top')
    axL.text(-3.1, top * 0.88, 'production inputs, last chunk', color=middle_color, fontsize=12, va='top')
    axL.text(-3.1, top * 0.79, 'weight ŵ(x) given to each\nreference point (right axis)', color=start_color, fontsize=12, va='top',
             linespacing=1.3)
    axL.set_yticks([])
    axL.set_xlabel('input $x_2$', fontsize=12)
    axL.set_xlim(-3.2, 3.8)
    despine(axL, keep=('bottom',))
    ax2.set_xlim(-3.2, 3.8)

    k = np.arange(1, len(rows) + 1)
    r = np.array([row[1] for row in rows])
    ec = np.array([row[2] for row in rows])
    ep = np.array([row[3] for row in rows])
    axR.plot(k, r, color=GREY_LINE, lw=2.5, marker='o', ms=6, solid_capstyle='round', zorder=3)
    axR.plot(k, ec, color=middle_color, lw=3, marker='o', ms=6, solid_capstyle='round', zorder=4)
    axR.plot(k, ep, color=start_color, lw=3, marker='o', ms=6, solid_capstyle='round', zorder=5)
    axR.text(k[-1] + 0.2, r[-1], 'realized', color=MID, fontsize=12, va='center')
    axR.text(k[-1] + 0.2, ec[-1], 'CBPE', color=middle_color, fontsize=12, va='center')
    axR.text(k[-1] + 0.2, ep[-1], 'PAPE', color=start_color, fontsize=12, va='center')
    axR.set_xticks(k)
    axR.set_xlim(0.5, len(rows) + 1.8)
    axR.tick_params(labelsize=11)
    axR.set_xlabel('production chunk, drifting further along $x_2$ →', fontsize=12)
    axR.set_ylabel('accuracy', fontsize=12)
    despine(axR)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.13)
    save_figure(fig, 'PAPE_reweighting')
    plt.close()
    mae_c = np.mean(np.abs(ec - r))
    mae_p = np.mean(np.abs(ep - r))
    print('3. PAPE: realized %s\n   CBPE %s\n   PAPE %s\n   mean abs error CBPE %.3f PAPE %.3f | last chunk gap CBPE %.3f PAPE %.3f | weight curve peak %.1f, raw max %.1f' %
          (np.round(r, 3), np.round(ec, 3), np.round(ep, 3), mae_c, mae_p, ec[-1] - r[-1], ep[-1] - r[-1], np.nanmax(wbin), w_last.max()))


# ===========================================================================
# 4. DLE — nanny model on heteroscedastic regression
# ===========================================================================
def fig_dle():
    fresh(4)
    n = 6000
    x_ref = RNG.uniform(0, 1, n)
    y_ref = 2 * x_ref + RNG.normal(0, 1, n) * x_ref          # noise grows with x
    child = LinearRegression().fit(x_ref[:, None], y_ref)
    f_ref = child.predict(x_ref[:, None])
    ae_ref = np.abs(y_ref - f_ref)
    nanny = LinearRegression().fit(np.column_stack([x_ref, f_ref]), ae_ref)   # same class as the child, as in the docs' example
    # production chunks drifting from the easy region to the hard one
    centres = np.linspace(0.15, 0.85, 8)
    rows = []
    for c in centres:
        x = np.clip(RNG.normal(c, 0.12, 2000), 0, 1)
        y = 2 * x + RNG.normal(0, 1, len(x)) * x
        f = child.predict(x[:, None])
        realized = np.mean(np.abs(y - f))
        est = nanny.predict(np.column_stack([x, f])).mean()
        rows.append((realized, est))
    r = np.array([a for a, _ in rows])
    e = np.array([b for _, b in rows])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.28})
    sub = RNG.choice(n, 900, replace=False)
    axL.scatter(x_ref[sub], y_ref[sub], s=9, color=GREY, zorder=2, linewidths=0)
    xs = np.linspace(0, 1, 200)
    fs = child.predict(xs[:, None])
    hs = nanny.predict(np.column_stack([xs, fs]))
    axL.plot(xs, fs, color=DARK, lw=2.2, zorder=4)
    axL.fill_between(xs, fs - hs, fs + hs, color=start_color, alpha=0.22, zorder=3, linewidth=0)
    axL.plot(xs, fs + hs, color=start_color, lw=2, zorder=4)
    axL.plot(xs, fs - hs, color=start_color, lw=2, zorder=4)
    axL.text(0.985, fs[-1] + hs[-1] + 0.12, 'f(x) ± nanny h(x)', color=start_color, fontsize=12, ha='right', va='bottom')
    axL.text(0.52, child.predict([[0.52]])[0] - 0.55, 'monitored model f', color=DARK, fontsize=12, ha='left', va='top')
    # the nanny's training target: |y - f(x)| for two example points
    for xi in (0.25, 0.8):
        j = sub[np.argmin(np.abs(x_ref[sub] - xi) + 0.05 * (np.abs(y_ref[sub] - child.predict([[xi]])[0]) < 0.3))]
        axL.plot([x_ref[j], x_ref[j]], [f_ref[j], y_ref[j]], color=end_color, lw=2, zorder=5)
        axL.scatter([x_ref[j]], [y_ref[j]], s=30, color=end_color, zorder=6)
        axL.text(x_ref[j] - 0.02, y_ref[j], f'|y − f(x)| = {ae_ref[j]:.2f}', color=end_color, fontsize=10.5, va='center',
                 ha='right' if xi > 0.5 else 'left', clip_on=False) if xi > 0.5 else \
            axL.text(x_ref[j] + 0.02, y_ref[j], f'|y − f(x)| = {ae_ref[j]:.2f}', color=end_color, fontsize=10.5, va='center')
    axL.set_xlabel('input $x$', fontsize=12)
    axL.set_ylabel('target $y$', fontsize=12)
    axL.set_xlim(0, 1)
    axL.set_xticks([0, 0.5, 1])
    axL.tick_params(labelsize=11)
    despine(axL)
    axL.text(0.02, axL.get_ylim()[1] - 0.15, 'reference data: noise grows with $x$', fontsize=12, color=MID, va='top')

    k = np.arange(1, len(rows) + 1)
    axR.plot(k, r, color=GREY_LINE, lw=2.5, marker='o', ms=6, solid_capstyle='round', zorder=3)
    axR.plot(k, e, color=start_color, lw=3, marker='o', ms=6, solid_capstyle='round', zorder=4)
    axR.text(k[-1] + 0.15, r[-1] - 0.012, 'realized MAE', color=MID, fontsize=12, va='top')
    axR.text(k[-1] + 0.15, e[-1] + 0.012, 'DLE estimate', color=start_color, fontsize=12, va='bottom')
    axR.set_xticks(k)
    axR.set_xticklabels([f'{c:.2f}' for c in centres], fontsize=10.5)
    axR.set_xlim(0.5, len(rows) + 2.2)
    axR.tick_params(axis='y', labelsize=11)
    axR.set_xlabel('production chunk, by mean input $x$ →', fontsize=12)
    axR.set_ylabel('MAE', fontsize=12)
    despine(axR)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.13)
    save_figure(fig, 'DLE_nanny')
    plt.close()
    print('4. DLE: realized %s\n   estimate %s\n   max gap %.3f' % (np.round(r, 3), np.round(e, 3), np.max(np.abs(r - e))))


# ===========================================================================
# 5. RCD — the new concept applied to the reference data; performance
#    decomposition into covariate shift and concept drift
# ===========================================================================
def fig_rcd():
    fresh(5)
    n = 5000

    def concept(X, rot):
        # boundary direction rotates by `rot` radians; steepness 3
        a = np.array([np.cos(np.pi / 4 + rot), -np.sin(np.pi / 4 + rot)])
        return expit(3.0 * (X @ a))

    X_ref = RNG.normal(0, 1, (n, 2))
    y_ref = (RNG.random(n) < concept(X_ref, 0.0)).astype(int)
    model = LogisticRegression().fit(X_ref, y_ref)
    yhat_ref = model.predict(X_ref)
    acc_ref = (yhat_ref == y_ref).mean()
    # monitored period: covariate shift + concept drift
    shift, rot = np.array([0.5, 0.2]), 0.55
    X_mon = RNG.normal(0, 1, (n, 2)) + shift
    y_mon = (RNG.random(n) < concept(X_mon, rot)).astype(int)
    yhat_mon = model.predict(X_mon)
    acc_mon = (yhat_mon == y_mon).mean()
    # covariate-shift-only effect: the model on monitored inputs under the OLD concept
    y_mon_old = (RNG.random(n) < concept(X_mon, 0.0)).astype(int)
    acc_cov = (yhat_mon == y_mon_old).mean()
    # RCD: learn the new concept from monitored labels, apply to reference inputs
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    g = make_pipeline(PolynomialFeatures(3), LogisticRegression(C=1.0, max_iter=2000)).fit(X_mon, y_mon)
    g_ref = make_pipeline(PolynomialFeatures(3), LogisticRegression(C=1.0, max_iter=2000)).fit(X_ref, y_ref)
    p_new = g.predict_proba(X_ref)[:, 1]
    est_under_new = cbpe_accuracy(p_new, yhat_ref)          # expected accuracy on reference under the new concept
    impact = est_under_new - acc_ref                          # PIE
    magnitude = np.mean(np.abs(p_new - g_ref.predict_proba(X_ref)[:, 1]))   # ME

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.3})
    sub = RNG.choice(n, 700, replace=False)
    axL.scatter(X_ref[sub, 0], X_ref[sub, 1], s=10, c=np.where(y_ref[sub] == 1, start_color, GREY), zorder=2, linewidths=0)
    gx, gy = np.meshgrid(np.linspace(-3.2, 3.2, 200), np.linspace(-3.2, 3.2, 200))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    axL.contour(gx, gy, model.predict_proba(G)[:, 1].reshape(gx.shape), levels=[0.5], colors=[DARK], linewidths=2.2, zorder=4)
    axL.contour(gx, gy, g.predict_proba(G)[:, 1].reshape(gx.shape), levels=[0.5], colors=[end_color], linewidths=2.6, zorder=5)
    axL.text(2.2, 2.9, 'model boundary\n(reference concept)', color=DARK, fontsize=11, ha='right', va='top', linespacing=1.3)
    axL.text(3.0, -2.2, 'new concept g,\nlearned from\nmonitored labels', color=end_color, fontsize=11, ha='right', va='top',
             linespacing=1.3)
    axL.text(-3.0, 2.9, 'reference inputs\n(cyan: y = 1)', color=MID, fontsize=11, ha='left', va='top', linespacing=1.3)
    axL.set_xlim(-3.2, 3.2)
    axL.set_ylim(-3.2, 3.2)
    axL.set_xticks([])
    axL.set_yticks([])
    axL.set_xlabel('$x_1$', fontsize=12)
    axL.set_ylabel('$x_2$', fontsize=12)
    despine(axL)

    # waterfall: reference -> covariate shift -> concept drift -> monitored
    steps = [('reference\naccuracy', acc_ref, None),
             ('covariate\nshift', acc_cov - acc_ref, 'inputs moved,\nold concept'),
             ('concept drift\n(RCD impact)', impact, 'new concept,\nold inputs'),
             ('realized on\nmonitored data', acc_mon, None)]
    x0 = 0
    level = 0
    for i, (name, val, note) in enumerate(steps):
        if i == 0:
            axR.bar(i, val, color=GREY, width=0.6, zorder=3)
            axR.text(i, val + 0.006, f'{val:.3f}', ha='center', va='bottom', fontsize=12, color=DARK)
            level = val
        elif i == len(steps) - 1:
            axR.bar(i, val, color=DARK, width=0.6, zorder=3)
            axR.text(i, val + 0.006, f'{val:.3f}', ha='center', va='bottom', fontsize=12, color=DARK)
            axR.plot([i - 1.3, i - 0.3], [level, level], color=GREY_LINE, lw=1, ls=(0, (3, 2)), zorder=2)
        else:
            col = middle_color if i == 1 else end_color
            axR.bar(i, val, bottom=level, color=col, width=0.6, zorder=3)
            axR.plot([i - 1.3, i - 0.3], [level, level], color=GREY_LINE, lw=1, ls=(0, (3, 2)), zorder=2)
            top_y = max(level, level + val)
            axR.text(i, top_y + 0.004, f'{val:+.3f}', ha='center', va='bottom', fontsize=12, color=col)
            axR.text(i, top_y + 0.016, note, ha='center', va='bottom', fontsize=10, color=col, linespacing=1.25)
            level += val
    resid = acc_mon - level
    axR.text(2.35, (level + acc_mon) / 2, f'residual {resid:+.3f}', ha='right', va='center', fontsize=10.5, color=MID)
    axR.set_xticks(range(len(steps)))
    axR.set_xticklabels([s[0] for s in steps], fontsize=11)
    lo = min(acc_mon, level) - 0.08
    axR.set_ylim(lo, max(acc_ref, acc_cov) + 0.045)
    axR.set_yticks(np.round(np.arange(np.ceil(lo * 20) / 20, acc_ref + 0.03, 0.05), 2))
    axR.tick_params(axis='y', labelsize=11)
    axR.set_ylabel('accuracy', fontsize=12)
    despine(axR)
    axR.tick_params(axis='x', length=0)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.16)
    save_figure(fig, 'RCD_decomposition')
    plt.close()
    print(f'5. RCD: ref acc {acc_ref:.3f}, cov-only {acc_cov:.3f} ({acc_cov - acc_ref:+.3f}), '
          f'RCD impact {impact:+.3f} (est under new {est_under_new:.3f}), monitored {acc_mon:.3f}, '
          f'residual {resid:+.3f}, magnitude {magnitude:.3f}, g acc on monitored {(g.predict(X_mon) == y_mon).mean():.3f}')


if __name__ == '__main__':
    which = sys.argv[1:] or ['ece', 'cbpe', 'pape', 'dle', 'rcd']
    world = None
    for w in which:
        if w == 'ece':
            fig_ece()
        elif w == 'cbpe':
            world = fig_cbpe()
        elif w == 'pape':
            fig_pape()
        elif w == 'dle':
            fig_dle()
        elif w == 'rcd':
            fig_rcd()
    print('\nProbabilistic figures regenerated.')
