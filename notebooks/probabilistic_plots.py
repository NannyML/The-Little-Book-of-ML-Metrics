"""Generate figures for the Probabilistic chapter (ECE, CBPE, PAPE, DLE, RCD).

Seeded educational implementations of the documented estimation principles,
not executions or parity tests of the NannyML package. The monitored model is
trained on data independent of reference data; unseen production labels are
used only to evaluate estimates. ECE also separates calibration from evaluation.

Intentional simplifications: isotonic calibration is always fitted; PAPE uses
a scikit-learn domain classifier; DLE uses a linear loss predictor clipped at
zero; RCD uses polynomial logistic regression. CBPE's shaded band illustrates
reference variation, not a validated interval or an alert threshold. All
plotted values and captions are derived from the seeded experiments. See
verify_probabilistic_plots.py and design/probabilistic-verification/ for checks.

Run from notebooks/:  uv run python probabilistic_plots.py
"""
import sys

sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
from style import *
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

GREY = '#c9c9c9'
GREY_LINE = '#9a9a9a'
DARK = '#2a2a2a'
MID = '#6f6f6f'
RNG = np.random.default_rng(2024)


MIN_REF = 50       # PAPE: weight bins with fewer reference points are too noisy to draw


def rounded_percentages(shares):
    """Whole percentages that add up to 100 (largest-remainder rounding)."""
    raw = 100 * np.asarray(shares, float) / np.sum(shares)
    out = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - out))[:100 - out.sum()]:
        out[k] += 1
    return out.tolist()


def fresh(seed):
    """Each figure gets its own generator so results do not depend on run order."""
    global RNG
    RNG = np.random.default_rng(seed)


def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------------------
# Educational estimators: documented core equations, explicit simplifications
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


def nonnegative_loss(model, x, predictions):
    """Use the same nonnegative loss prediction for chart envelopes and MAE."""
    return np.maximum(0.0, model.predict(np.column_stack([x, predictions])))


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
    for name, zz in [('before temperature scaling', z_over[ev]), (f'after scaling with T = {T:.1f}', z_over[ev] / T)]:
        p = expit(zz)
        conf = np.maximum(p, 1 - p)
        correct = ((p >= 0.5).astype(int) == y_te[ev]).astype(float)
        ece, rows = ece_score(conf, correct)
        panels.append((name, ece, rows))
    auc = roc_auc_score(y_te[ev], z_over[ev])

    fig, axes = book_figure(1.0, 2.25, 1, 2, sharey=True)
    for ax, (name, ece, rows) in zip(axes, panels):
        ax.plot([0.5, 1], [0.5, 1], color=REF, lw=LW_THIN, ls=(0, (3, 2)), zorder=1)
        pct = rounded_percentages([share for _, _, share, _, _ in rows])
        first = min(k for k, r in enumerate(rows) if r[2] > 0)
        for k, (lo, hi, share, acc, cf) in enumerate(rows):
            if share == 0:
                continue
            ax.bar((lo + hi) / 2, acc, width=(hi - lo) * 0.9, color=NML_CYAN, alpha=0.5, linewidth=0, zorder=3)
            ax.plot([cf, cf], [min(acc, cf), max(acc, cf)], color=NML_RED, lw=LW + 0.4, solid_capstyle='butt', zorder=4)
            ax.text((lo + hi) / 2, 0.025, f'{pct[k]}%' if k == first else f'{pct[k]}', ha='center', va='bottom',
                    fontsize=SMALL_PT,
                    color=INK, zorder=5)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.02)
        unit_ticks(ax, 'x', 0.1, lo=0.5)
        unit_ticks(ax, 'y', 0.5)
        ax.set_xlabel('confidence bin')
        ax.set_title(f'{name}\nECE = {num(ece, 3)}', loc='left', fontsize=TEXT_PT, linespacing=1.3)
        tidy_axes(ax)
    axes[0].set_ylabel('observed accuracy')
    save_figure(fig, 'ECE_reliability')
    plt.close()
    top = panels[0][2][-1]
    print(f'1. ECE: over-confident {panels[0][1]:.3f} -> temperature {T:.2f} gives {panels[1][1]:.3f}; AUC {auc:.3f}; '
          f'top bin share {top[2]:.3f} acc {top[3]:.3f} conf {top[4]:.3f}')
    return locals()


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


def sample_world(n, centre, spread, concept=0.0, rng=None):
    rng = RNG if rng is None else rng
    x = rng.normal(centre, spread, n)
    y = (rng.random(n) < true_prob(x, concept)).astype(int)
    return x, y


def fig_cbpe():
    fresh(2)
    # Independent child training; reference labels fit the calibrator only.
    x_train, y_train = sample_world(20000, 0.0, 1.0, rng=np.random.default_rng(2002))
    x_ref, y_ref = sample_world(20000, 0.0, 1.0)
    model = LogisticRegression().fit(x_train[:, None], y_train)
    s_ref = model.predict_proba(x_ref[:, None])[:, 1]
    cal = fit_calibrator(s_ref, y_ref)
    thr = 0.5
    # Illustrative reference variation only: neither uncertainty nor alert limits
    ref_chunk_acc = [((s_ref[i:i + 2000] >= thr).astype(int) == y_ref[i:i + 2000]).mean() for i in range(0, 20000, 2000)]
    band = 3 * np.std(ref_chunk_acc, ddof=1)
    # production: 12 chunks; chunks 1-7 shift away from the boundary (easier inputs),
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

    # First draw from a fixed independent stream. Never condition on its outcomes.
    x, y = sample_world(12, 0.5, 0.6, rng=np.random.default_rng(2004))
    s = model.predict_proba(x[:, None])[:, 1]
    c = cal.predict(s)
    yhat = (s >= thr).astype(int)
    p_correct = np.where(yhat == 1, c, 1 - c)
    x_val, y_val = sample_world(5000, 0.0, 1.0, rng=np.random.default_rng(2003))
    s_val = model.predict_proba(x_val[:, None])[:, 1]
    validation = dict(realized=float(np.mean((s_val >= thr) == y_val)),
                      estimated=float(cbpe_accuracy(cal.predict(s_val), s_val >= thr)))
    order = np.argsort(-c)

    fig, (axL, axR) = book_figure(1.0, 2.5, 1, 2, gridspec_kw={'width_ratios': [0.9, 1.25]})
    ys = np.arange(len(order))[::-1]
    for row, j in zip(ys, order):
        axL.barh(row, p_correct[j], color=NML_CYAN, height=0.62, linewidth=0, zorder=3)
        axL.barh(row, 1 - p_correct[j], left=p_correct[j], color=LIGHT, height=0.62, linewidth=0, zorder=3)
        axL.text(-0.3, row, f'{yhat[j]}', ha='center', va='center', color=INK, fontsize=SMALL_PT)
        axL.text(-0.2, row, num(c[j]), ha='left', va='center', color=INK, fontsize=SMALL_PT)   # '0.' aligns the decimals
        ok = yhat[j] == y[j]
        axL.text(1.06, row, '\u2713' if ok else '\u2717', ha='left', va='center', color=NML_CYAN if ok else NML_RED,
                 fontsize=SMALL_PT)
    top = len(order) - 0.2
    for xx, t, ha in [(-0.3, '$\\hat{y}$', 'center'), (-0.2, '$c$', 'left'), (1.06, 'later', 'left')]:
        axL.text(xx, top, t, ha=ha, va='bottom', color=MUTED, fontsize=SMALL_PT)
    axL.set_xlim(0, 1.0)
    axL.set_ylim(-0.6, len(order) + 0.5)
    axL.set_xticks([0, 0.5, 1.0], labels=['0.0', '0.5', '1.0'])
    axL.set_yticks([])
    axL.set_xlabel('chance of being right')
    axL.set_title(f'twelve predictions\nexpected {num(p_correct.mean())}, realized {num((yhat == y).mean())}',
                  loc='left', fontsize=TEXT_PT, linespacing=1.3)
    tidy_axes(axL, left=False)
    k = np.arange(1, len(chunks) + 1)
    axR.axvspan(7.5, 12.5, color=RED_TINT, zorder=0, linewidth=0)
    axR.fill_between(k, est - band, est + band, color=NML_CYAN, alpha=0.15, linewidth=0, zorder=2)
    axR.plot(k, realized, color=REF, lw=LW - 0.2, marker='o', ms=2.6, zorder=3)
    axR.plot(k, est, color=NML_CYAN, marker='o', ms=2.6, zorder=4)
    axR.axhline(ref_acc, color=REF, lw=LW_THIN, ls=(0, (3, 2)), zorder=1)
    axR.text(12.6, ref_acc, f'reference\n{num(ref_acc)}', color=MUTED, va='center', linespacing=1.15)
    label_end(axR, 12, est[-1], 'CBPE\nestimate', NML_CYAN, dx=4, linespacing=1.15)
    label_end(axR, 12, realized[-1], 'realized', MUTED, dx=4)
    axR.text(4.0, 0.57, 'input mix\nchanges', ha='center', va='bottom', color=INK, linespacing=1.2)
    axR.text(10.0, max(realized.max(), est.max()) + 0.035, 'concept drift', ha='center', va='top', color=NML_RED)
    axR.set_xticks([1, 4, 8, 12])
    axR.set_xlim(0.5, 12.5)
    axR.set_ylim(0.55, max(realized.max(), est.max()) + 0.04)
    axR.set_xlabel('production chunk')
    axR.set_ylabel('accuracy')
    tidy_axes(axR)
    save_figure(fig, 'CBPE_estimation')
    plt.close()
    print('2. CBPE: ref acc %.3f | band ±%.3f | chunks realized %s | est %s | left panel expected %.3f realized %.3f' %
          (ref_acc, band, np.round(realized, 3), np.round(est, 3), p_correct.mean(), (yhat == y).mean()))
    return locals()


# ===========================================================================
# 3. PAPE — density-ratio weights and the re-weighted calibrator
# ===========================================================================
def fig_pape():
    fresh(3)
    """Two inputs; the true concept has an interaction the logistic model cannot
    represent, so the calibration of a given score depends on x2.  Production
    drifts along x2, and only a calibrator re-weighted to that region is right."""
    def true_logit(X):
        return 2.2 * X[:, 0] * (1 + 0.7 * X[:, 1])

    def sample(n, c2, rng=None):
        rng = RNG if rng is None else rng
        X = np.column_stack([rng.normal(0, 1, n), rng.normal(c2, 0.5 if c2 else 1.0, n)])
        y = (rng.random(n) < expit(true_logit(X))).astype(int)
        return X, y

    X_train, y_train = sample(8000, 0.0, rng=np.random.default_rng(3003))
    X_ref, y_ref = sample(8000, 0.0)
    model = LogisticRegression().fit(X_train, y_train)
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
    # Evaluate the final fitted calibrators on an additional unseen batch.
    X_val, y_val = sample(5000, c2, rng=np.random.default_rng(3004))
    s_val = model.predict_proba(X_val)[:, 1]
    validation = dict(realized=float(np.mean((s_val >= thr) == y_val)),
                      cbpe=float(cbpe_accuracy(cal.predict(s_val), s_val >= thr)),
                      pape=float(cbpe_accuracy(cal_w.predict(s_val), s_val >= thr)))

    fig = book_canvas(1.0, 2.5, layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    axL = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0], sharex=axL)
    axR = fig.add_subplot(grid[:, 1])
    bins = np.linspace(-3.2, 3.8, 36)
    axL.hist(X_ref[:, 1], bins=bins, color=LIGHT, density=True, zorder=2)
    axL.hist(X_last[:, 1], bins=bins, histtype='step', color=INK, lw=LW_THIN + 0.2, density=True, zorder=3)
    centers = (bins[:-1] + bins[1:]) / 2
    idx = np.clip(np.digitize(X_ref[:, 1], bins) - 1, 0, len(bins) - 2)
    counts = np.bincount(idx, minlength=len(bins) - 1)
    wbin = np.array([w_last[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(bins) - 1)])
    shown = np.where(counts >= MIN_REF, wbin, np.nan)   # bins with too few reference points are not drawn
    print('PAPE drawn bins', [(round(c, 2), int(n), round(w, 1)) for c, n, w in zip(centers, counts, shown) if n >= MIN_REF][-4:])
    ax2.plot(centers, shown, color=NML_CYAN, zorder=4)
    ax2.set_ylim(0, np.nanmax(shown) * 1.15)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axL.legend(handles=[Patch(facecolor=LIGHT, label='reference'),
                        Line2D([], [], color=INK, lw=LW_THIN + 0.2, label='production')],
               loc='upper left', handlelength=1.0)
    axL.set_ylabel('density')
    axL.set_yticks([0, 0.4, 0.8])
    ax2.set_ylabel('mean weight')
    ax2.set_xlabel('input x\u2082')
    axL.tick_params(axis='x', labelbottom=False)
    ax2.set_xticks([-3, 0, 3], labels=['\u22123', '0', '3'])
    axL.set_xlim(-3.2, 3.8)
    tidy_axes(axL)
    tidy_axes(ax2)
    fig.align_ylabels([axL, ax2])

    k = np.arange(1, len(rows) + 1)
    r = np.array([row[1] for row in rows])
    ec = np.array([row[2] for row in rows])
    ep = np.array([row[3] for row in rows])
    axR.plot(k, r, color=REF, lw=LW - 0.2, marker='o', ms=2.6, zorder=3)
    axR.plot(k, ec, color=NML_PURPLE, marker='o', ms=2.6, zorder=4)
    axR.plot(k, ep, color=NML_CYAN, marker='o', ms=2.6, zorder=5)
    label_end(axR, k[-1], r[-1] - 0.006, 'realized', MUTED, dx=4)
    label_end(axR, k[-1], ec[-1], 'CBPE', NML_PURPLE, dx=4)
    label_end(axR, k[-1], ep[-1] + 0.007, 'PAPE', NML_CYAN, dx=4)
    axR.set_xticks([1, 4, 7, 10])
    axR.set_xlim(0.5, len(rows) + 0.5)
    axR.set_yticks([.75, .80, .85, .90], labels=['0.75', '0.80', '0.85', '0.90'])
    axR.set_xlabel('production chunk')
    axR.set_ylabel('accuracy')
    tidy_axes(axR)
    save_figure(fig, 'PAPE_reweighting')
    plt.close()
    mae_c = np.mean(np.abs(ec - r))
    mae_p = np.mean(np.abs(ep - r))
    print('3. PAPE: realized %s\n   CBPE %s\n   PAPE %s\n   mean abs error CBPE %.3f PAPE %.3f | last chunk gap CBPE %.3f PAPE %.3f | weight curve peak %.1f, raw max %.1f' %
          (np.round(r, 3), np.round(ec, 3), np.round(ep, 3), mae_c, mae_p, ec[-1] - r[-1], ep[-1] - r[-1], np.nanmax(wbin), w_last.max()))
    return locals()


# ===========================================================================
# 4. DLE — nanny model on heteroscedastic regression
# ===========================================================================
def fig_dle():
    fresh(4)
    n = 6000
    train_rng = np.random.default_rng(4004)
    x_train = train_rng.uniform(0, 1, n)
    y_train = 2 * x_train + train_rng.normal(0, 1, n) * x_train
    x_ref = RNG.uniform(0, 1, n)
    y_ref = 2 * x_ref + RNG.normal(0, 1, n) * x_ref          # noise grows with x
    child = LinearRegression().fit(x_train[:, None], y_train)
    f_ref = child.predict(x_ref[:, None])
    ae_ref = np.abs(y_ref - f_ref)
    nanny = LinearRegression().fit(np.column_stack([x_ref, f_ref]), ae_ref)
    # production chunks drifting from the easy region to the hard one
    centres = np.linspace(0.15, 0.85, 8)
    rows = []
    for c in centres:
        x = np.clip(RNG.normal(c, 0.12, 2000), 0, 1)
        y = 2 * x + RNG.normal(0, 1, len(x)) * x
        f = child.predict(x[:, None])
        realized = np.mean(np.abs(y - f))
        est = nonnegative_loss(nanny, x, f).mean()
        rows.append((realized, est))
    r = np.array([a for a, _ in rows])
    e = np.array([b for _, b in rows])
    val_rng = np.random.default_rng(4005)
    x_val = val_rng.uniform(0, 1, 5000)
    y_val = 2 * x_val + val_rng.normal(0, 1, len(x_val)) * x_val
    f_val = child.predict(x_val[:, None])
    validation = dict(realized=float(np.mean(np.abs(y_val-f_val))),
                      estimated=float(nonnegative_loss(nanny, x_val, f_val).mean()))

    fig, (axL, axR) = book_figure(1.0, 2.2, 1, 2, gridspec_kw={'width_ratios': [1.15, 1]})
    sub = RNG.choice(n, 900, replace=False)
    axL.scatter(x_ref[sub], y_ref[sub], s=2.5, color=LIGHT, zorder=2, linewidths=0)
    xs = np.linspace(0, 1, 200)
    fs = child.predict(xs[:, None])
    hs = nonnegative_loss(nanny, xs, fs)
    axL.fill_between(xs, fs - hs, fs + hs, color=NML_CYAN, alpha=0.2, zorder=3, linewidth=0)
    axL.plot(xs, fs + hs, color=NML_CYAN, lw=LW_THIN + 0.3, zorder=4)
    axL.plot(xs, fs - hs, color=NML_CYAN, lw=LW_THIN + 0.3, zorder=4)
    axL.plot(xs, fs, color=INK, lw=LW - 0.2, zorder=4)
    axL.text(1.02, fs[-1] + hs[-1], '$f(x) + h(x)$', color=NML_CYAN, va='center', clip_on=False)
    axL.text(1.02, fs[-1], 'model $f$', color=INK, va='center', clip_on=False)
    axL.text(1.02, fs[-1] - hs[-1], '$f(x) - h(x)$', color=NML_CYAN, va='center', clip_on=False)
    for xi in (0.25, 0.8):
        j = sub[np.argmin(np.abs(x_ref[sub] - xi) + 0.05 * (np.abs(y_ref[sub] - child.predict([[xi]])[0]) < 0.3))]
        axL.plot([x_ref[j], x_ref[j]], [f_ref[j], y_ref[j]], color=NML_RED, lw=LW - 0.2, zorder=5)
        axL.scatter([x_ref[j]], [y_ref[j]], s=10, color=NML_RED, zorder=6, linewidths=0)
        axL.annotate(f'|error| = {num(ae_ref[j])}', xy=(x_ref[j], y_ref[j]),
                     xytext=(.03 if xi < .5 else .58, -.95), color=NML_RED, ha='left',
                     va='top', arrowprops=dict(arrowstyle='-', color=NML_RED, lw=LW_THIN, shrinkA=1, shrinkB=2),
                     bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
    axL.set_xlabel('input $x$')
    axL.set_ylabel('target $y$')
    axL.set_xlim(0, 1)
    axL.set_xticks([0, 0.5, 1.0], labels=['0.0', '0.5', '1.0'])
    minus_axis(axL, 'y', 0)
    axL.set_title('reference data', loc='left', fontsize=TEXT_PT)
    tidy_axes(axL)
    axR.plot(centres, e, color=NML_CYAN, zorder=3)
    axR.scatter(centres, r, s=16, facecolors='white', edgecolors=MUTED, linewidths=0.8, zorder=4)
    from matplotlib.lines import Line2D
    axR.legend(handles=[Line2D([], [], color=NML_CYAN, label='DLE estimate'),
                        Line2D([], [], ls='none', marker='o', ms=MS, mfc='white', mec=MUTED, mew=0.8,
                               label='measured MAE')], loc='upper left', handlelength=1.2, borderaxespad=0.2)
    axR.set_xticks([0.2, 0.4, 0.6, 0.8], labels=['0.2', '0.4', '0.6', '0.8'])
    axR.set_xlim(0.1, 0.9)
    axR.set_xlabel('chunk center $x$')
    axR.set_ylabel('MAE')
    tidy_axes(axR)
    save_figure(fig, 'DLE_nanny')
    plt.close()
    print('4. DLE: realized %s\n   estimate %s\n   max gap %.3f' % (np.round(r, 3), np.round(e, 3), np.max(np.abs(r - e))))
    return locals()


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

    train_rng = np.random.default_rng(5005)
    X_train = train_rng.normal(0, 1, (n, 2))
    y_train = (train_rng.random(n) < concept(X_train, 0.0)).astype(int)
    X_ref = RNG.normal(0, 1, (n, 2))
    y_ref = (RNG.random(n) < concept(X_ref, 0.0)).astype(int)
    model = LogisticRegression().fit(X_train, y_train)
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
    val_rng = np.random.default_rng(5006)
    X_val = val_rng.normal(0, 1, (5000, 2)) + shift
    y_val = (val_rng.random(5000) < concept(X_val, rot)).astype(int)
    p_val = g.predict_proba(X_val)[:, 1]
    validation = dict(concept_accuracy=float(np.mean((p_val >= .5) == y_val)),
                      probability_mae_to_oracle=float(np.mean(np.abs(p_val-concept(X_val, rot)))))

    fig, (axL, axR) = book_figure(1.0, 2.25, 1, 2, gridspec_kw={'width_ratios': [1, 1.9]})
    sub = RNG.choice(n, 700, replace=False)
    axL.scatter(X_ref[sub, 0], X_ref[sub, 1], s=2.5, c=np.where(y_ref[sub] == 1, NML_CYAN, LIGHT), zorder=2,
                linewidths=0)
    gx, gy = np.meshgrid(np.linspace(-3.2, 3.2, 200), np.linspace(-3.2, 3.2, 200))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    axL.contour(gx, gy, model.predict_proba(G)[:, 1].reshape(gx.shape), levels=[0.5], colors=[INK],
                linewidths=LW - 0.2, zorder=4)
    axL.contour(gx, gy, g.predict_proba(G)[:, 1].reshape(gx.shape), levels=[0.5], colors=[NML_RED],
                linewidths=LW, zorder=5)
    axL.set_xlim(-3.2, 3.2)
    axL.set_ylim(-3.2, 3.2)
    axL.set_aspect('equal')
    axL.set_xticks([])
    axL.set_yticks([])
    axL.set_xlabel('x\u2081')
    axL.set_ylabel('x\u2082')
    axL.set_title('reference inputs (cyan: y = 1)', loc='left', fontsize=TEXT_PT)
    tidy_axes(axL)
    from matplotlib.lines import Line2D
    axL.legend(handles=[Line2D([], [], color=INK, lw=LW - 0.2, label='model f'),
                        Line2D([], [], color=NML_RED, lw=LW, label='new concept g')],
               loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, handlelength=1.2)

    resid = acc_mon - (acc_cov + impact)
    steps = [('Reference', acc_ref, None),
             ('Input mix', acc_cov-acc_ref, NML_PURPLE),
             ('RCD impact', impact, NML_RED),
             ('Residual', resid, REF),
             ('Observed', acc_mon, None)]
    # Printed steps are differences of rounded levels, so the printed numbers add up.
    X = [0, 1.35, 2.5, 3.65, 5.0]     # the two levels sit a little apart from the three steps
    level, shown = acc_ref, round(acc_ref, 3)
    for i, (name, val, col) in enumerate(steps):
        x = X[i]
        if col is None:
            axR.plot([x - .28, x + .28], [val, val], color=INK, lw=LW + 0.4, zorder=4)
            axR.text(x, val + .008, num(val, 3), ha='center', va='bottom', color=INK)
        else:
            axR.plot([X[i - 1] + .28, x - .28], [level, level], color=REF, lw=LW_THIN, ls=(0, (3, 2)), zorder=2)
            axR.bar(x, val, bottom=level, color=col, width=.56, linewidth=0, zorder=3)
            new_shown = round(level + val, 3)
            up = val >= 0 or i == 1          # gains are labelled above the bar, losses below it
            axR.text(x, (max(level, level + val) + .008) if up else (min(level, level + val) - .008),
                     num(new_shown - shown, 3, sign=True), ha='center', va='bottom' if up else 'top',
                     color=col if col != REF else MUTED)
            level += val
            shown = new_shown
    axR.plot([X[3] + .28, X[4] - .28], [acc_mon, acc_mon], color=REF, lw=LW_THIN, ls=(0, (3, 2)))
    axR.set_xticks(X, labels=['reference', 'input\nmix', 'RCD\nimpact', 'residual', 'new\ndata'],
                   fontsize=SMALL_PT)
    axR.set_xlim(-.5, X[-1] + .5)
    lo = min(acc_mon, acc_cov + impact) - .035
    axR.set_ylim(lo, max(acc_ref, acc_cov) + .055)
    ticks = np.round(np.arange(np.ceil(lo * 20) / 20, max(acc_ref, acc_cov) + .05, .05), 2)
    axR.set_yticks(ticks, labels=[f'{t:.2f}' for t in ticks])
    axR.set_ylabel('accuracy')
    tidy_axes(axR)
    axR.tick_params(axis='x', length=0)
    save_figure(fig, 'RCD_decomposition')
    plt.close()
    print(f'5. RCD: ref acc {acc_ref:.3f}, cov-only {acc_cov:.3f} ({acc_cov - acc_ref:+.3f}), '
          f'RCD impact {impact:+.3f} (est under new {est_under_new:.3f}), monitored {acc_mon:.3f}, '
          f'residual {resid:+.3f}, magnitude {magnitude:.3f}, g acc on monitored {(g.predict(X_mon) == y_mon).mean():.3f}')
    return locals()


if __name__ == '__main__':
    which = sys.argv[1:] or ['ece', 'cbpe', 'pape', 'dle', 'rcd']
    for w in which:
        if w == 'ece':
            fig_ece()
        elif w == 'cbpe':
            fig_cbpe()
        elif w == 'pape':
            fig_pape()
        elif w == 'dle':
            fig_dle()
        elif w == 'rcd':
            fig_rcd()
    print('\nProbabilistic figures regenerated.')
