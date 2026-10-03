"""Dedicated figures for pages that used to share one image, plus MASE and wMAPE.

Clustering (Ch 4): one synthetic 3-class dataset, several clusterings of it, and
for each metric a figure that shows how the metric is *computed* from them:
  - Mutual Info Score: per-cell contributions of the contingency table
  - Homogeneity: class composition of each cluster and the conditional entropy
  - V-Measure: the (homogeneity, completeness) plane with iso-V curves
  - FMI: the (pairwise precision, pairwise recall) plane with iso-FMI curves
Ranking (Ch 5): CG is order-blind — two orderings of the same items.
Regression (Ch 2): MASE from the naive lag-1 forecast; wMAPE from the channel table.

Every number printed is computed here.  Run from notebooks/:
    uv run python dedicated_figures.py
"""
import sys

sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
from style import *
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (mutual_info_score, homogeneity_score, completeness_score, v_measure_score,
                             fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score)
from sklearn.metrics.cluster import contingency_matrix
from scipy.stats import entropy

GREY = '#c9c9c9'
GREY_LINE = '#9a9a9a'
DARK = '#2a2a2a'
MID = '#6f6f6f'
CYAN_MAP = mcolors.LinearSegmentedColormap.from_list('cyan_map', ['#ffffff', start_color])
DIV_MAP = mcolors.LinearSegmentedColormap.from_list('div_map', [end_color, '#ffffff', start_color])
RNG = np.random.default_rng(3)


def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------------------
# Shared clustering world
# ---------------------------------------------------------------------------
N, K = 300, 3
X_sep, y = make_blobs(n_samples=N, centers=K, cluster_std=0.6, random_state=4)
X_ovl, _ = make_blobs(n_samples=N, centers=K, cluster_std=2.2, random_state=4)   # same labels, noisier points


def km(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


CLUST = {
    'well separated, $k=3$': km(X_sep, 3),
    'overlapping, $k=3$': km(X_ovl, 3),
    'over-split, $k=6$': km(X_sep, 6),
    'one cluster, $k=1$': np.zeros(N, dtype=int),
}


def pair_pr(y_true, y_pred):
    """Pairwise precision and recall from the contingency table."""
    c = contingency_matrix(y_true, y_pred)
    same_both = (c * (c - 1) / 2).sum()
    same_pred = (c.sum(0) * (c.sum(0) - 1) / 2).sum()
    same_true = (c.sum(1) * (c.sum(1) - 1) / 2).sum()
    return same_both / same_pred if same_pred else 0.0, same_both / same_true if same_true else 0.0


# ===========================================================================
# 1. Mutual Info Score — per-cell contributions
# ===========================================================================
def fig_mi():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={'wspace': 0.35})
    out = {}
    for ax, name in zip(axes, ['well separated, $k=3$', 'overlapping, $k=3$']):
        pred = CLUST[name]
        c = contingency_matrix(y, pred).astype(float)
        p = c / c.sum()
        pc, pk = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            contrib = np.where(p > 0, p * np.log(p / (pc @ pk)), 0.0)
        mi = contrib.sum()
        out[name] = (mi, mutual_info_score(y, pred))
        vmax = np.abs(contrib).max()
        ax.imshow(contrib, cmap=DIV_MAP, vmin=-vmax, vmax=vmax, aspect='auto')
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ax.text(j, i - 0.16, f'{int(c[i, j])}', ha='center', va='center', fontsize=13, color=DARK)
                ax.text(j, i + 0.22, f'{contrib[i, j]:+.3f}', ha='center', va='center', fontsize=10.5,
                        color=end_color if contrib[i, j] < 0 else (start_color if contrib[i, j] > 0.02 else MID))
        ax.set_xticks(range(c.shape[1]))
        ax.set_xticklabels([f'cluster {j}' for j in range(c.shape[1])], fontsize=11)
        ax.set_yticks(range(c.shape[0]))
        ax.set_yticklabels([f'class {i}' for i in range(c.shape[0])], fontsize=11)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'MI = sum of cells = {mi:.3f} nats', fontsize=15, pad=26, color=DARK)
        ax.text(0.5, 1.04, name, ha='center', va='bottom', fontsize=12.5, color=MID, transform=ax.transAxes)
        ax.text(0.5, -0.14, f'H(class) = {entropy(pc.ravel()):.3f} nats, the ceiling', ha='center', va='top',
                fontsize=11, color=MID, transform=ax.transAxes)
    axes[0].text(-0.75, K - 0.45, 'count\n$p\\log\\frac{p}{p_c p_k}$', ha='right', va='center', fontsize=10.5, color=MID)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.14)
    save_figure(fig, 'MI_contingency_contributions')
    plt.close()
    print('1. MI', {k: (round(a, 3), round(b, 3)) for k, (a, b) in out.items()})


# ===========================================================================
# 2. Homogeneity — class composition of each cluster
# ===========================================================================
def fig_homogeneity():
    names = ['well separated, $k=3$', 'overlapping, $k=3$', 'one cluster, $k=1$']
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), gridspec_kw={'wspace': 0.3})
    Hc = entropy(np.bincount(y) / N)
    out = {}
    class_colors = [start_color, middle_color, end_color]
    for ax, name in zip(axes, names):
        pred = CLUST[name]
        c = contingency_matrix(y, pred).astype(float)     # classes x clusters
        ncl = c.shape[1]
        h_cond = 0.0
        for j in range(ncl):
            col = c[:, j]
            share = col.sum() / N
            hk = entropy(col / col.sum())
            h_cond += share * hk
            bottom = 0
            for i in range(K):
                ax.bar(j, col[i], bottom=bottom, color=class_colors[i], width=0.62, zorder=3, edgecolor='white', linewidth=1)
                bottom += col[i]
            ax.text(j, col.sum() + 4, f'H = {hk:.2f}', ha='center', va='bottom', fontsize=11, color=DARK)
        h = 1 - h_cond / Hc
        out[name] = (h, homogeneity_score(y, pred))
        ax.set_xticks(range(ncl))
        ax.set_xticklabels([f'cluster {j}' for j in range(ncl)], fontsize=11)
        ax.set_ylim(0, N * 1.12 if ncl == 1 else 150)
        ax.set_yticks([])
        ax.tick_params(axis='x', length=0)
        despine(ax, keep=('bottom',))
        ax.set_title(f'Homogeneity = {h:.2f}', fontsize=15, pad=26, color=DARK)
        ax.text(0.5, 1.04, name, ha='center', va='bottom', fontsize=12.5, color=MID, transform=ax.transAxes)
        ax.text(0.5, -0.12, f'H(class | cluster) = {h_cond:.2f} of H(class) = {Hc:.2f}', ha='center', va='top', fontsize=11,
                color=MID, transform=ax.transAxes)
    handles = [plt.Rectangle((0, 0), 1, 1, color=class_colors[i]) for i in range(K)]
    fig.legend(handles, [f'class {i}' for i in range(K)], loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.82, bottom=0.2)
    save_figure(fig, 'Homogeneity_cluster_composition')
    plt.close()
    print('2. Homogeneity', {k: (round(a, 3), round(b, 3)) for k, (a, b) in out.items()})


# ===========================================================================
# 3. V-Measure — the (h, c) plane with iso-V curves
# ===========================================================================
def fig_vmeasure():
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    g = np.linspace(0.001, 1, 400)
    H, C = np.meshgrid(g, g)
    V = 2 * H * C / (H + C)
    levels = [0.2, 0.4, 0.6, 0.8]
    cs = ax.contour(H, C, V, levels=levels, colors=[GREY_LINE], linewidths=1.0)
    # label each iso-curve where it crosses h = 0.8: c = v h / (2h - v)
    ax.clabel(cs, fmt=lambda v: f'V = {v:g}', fontsize=10.5, colors=MID, inline=True,
              manual=[(0.8, v * 0.8 / (1.6 - v)) for v in levels])
    pts = {}
    for name, color in zip(CLUST, [start_color, middle_color, end_color, DARK]):
        pred = CLUST[name]
        h, c, v = homogeneity_score(y, pred), completeness_score(y, pred), v_measure_score(y, pred)
        pts[name] = (h, c, v)
        ax.scatter(h, c, s=160, color=color, zorder=5, edgecolor='white', linewidth=1.5, clip_on=False)
    # points on the top edge are labeled above the frame; the two inside get one-line labels in open space
    kw = dict(fontsize=11.5, color=DARK, clip_on=False)
    h, c, v = pts['one cluster, $k=1$']
    ax.text(h, 1.035, f'one cluster, $k=1$:  V = {v:.2f}', ha='left', va='bottom', **kw)
    h, c, v = pts['well separated, $k=3$']
    ax.text(h, 1.035, f'well separated, $k=3$:  V = {v:.2f}', ha='right', va='bottom', **kw)
    h, c, v = pts['overlapping, $k=3$']
    ax.text(h - 0.11, c + 0.045, f'overlapping, $k=3$:  V = {v:.2f}', ha='left', va='bottom', **kw)
    h, c, v = pts['over-split, $k=6$']
    ax.text(h - 0.03, c, f'over-split, $k=6$:  V = {v:.2f}', ha='right', va='center', **kw)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(labelsize=12)
    ax.set_xlabel('homogeneity $h$  (each cluster holds one class)', fontsize=13)
    ax.set_ylabel('completeness $c$  (each class sits in one cluster)', fontsize=13)
    despine(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.11)
    save_figure(fig, 'V_Measure_plane')
    plt.close()
    print('3. V-Measure', {k: tuple(round(x, 3) for x in v) for k, v in pts.items()})


# ===========================================================================
def fig_fmi():
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    g = np.linspace(0.001, 1, 400)
    P, R = np.meshgrid(g, g)
    F = np.sqrt(P * R)
    levels = [0.3, 0.5, 0.8, 0.9]
    cs = ax.contour(P, R, F, levels=levels, colors=[GREY_LINE], linewidths=1.0)
    # label each iso-curve where it crosses p = 0.8 (r = f^2 / p), the 0.9 curve at p = 0.95
    # label each iso-curve where it crosses p = 0.8 (r = f^2 / p); the two upper ones at p = 0.95, clear of the points
    ax.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=10.5, colors=MID, inline=True,
              manual=[(0.8, v * v / 0.8) if v < 0.8 else (0.95, v * v / 0.95) for v in levels])
    F1 = 2 * P * R / (P + R)
    cs2 = ax.contour(P, R, F1, levels=[0.5], colors=[end_color], linewidths=0.9, linestyles='dashed', alpha=0.8)
    # Label the dashed pairwise-F1 contour away from the example points.
    ax.clabel(cs2, fmt=lambda v: f'F1 = {v:g}', fontsize=10, colors=end_color, inline=True,
              manual=[(0.55, 0.5 * 0.55 / (2 * 0.55 - 0.5))])
    pts = {}
    for name, color in zip(CLUST, [start_color, middle_color, end_color, DARK]):
        pred = CLUST[name]
        p, r = pair_pr(y, pred)
        fmi = fowlkes_mallows_score(y, pred)
        pts[name] = (p, r, fmi, adjusted_rand_score(y, pred))
        ax.scatter(p, r, s=160, color=color, zorder=5, edgecolor='white', linewidth=1.5, clip_on=False)
    kw = dict(fontsize=11.5, color=DARK, clip_on=False)
    p, r, fmi, _ = pts['one cluster, $k=1$']
    ax.text(p, 1.035, f'one cluster, $k=1$:  FMI = {fmi:.2f}', ha='center', va='bottom', **kw)
    p, r, fmi, _ = pts['well separated, $k=3$']
    ax.text(p, 1.035, f'well separated, $k=3$:  FMI = {fmi:.2f}', ha='right', va='bottom', **kw)
    p, r, fmi, _ = pts['overlapping, $k=3$']
    ax.text(p - 0.03, r - 0.035, f'overlapping, $k=3$:  FMI = {fmi:.2f}', ha='right', va='top', **kw)
    p, r, fmi, _ = pts['over-split, $k=6$']
    ax.text(p - 0.03, r, f'over-split, $k=6$:  FMI = {fmi:.2f}', ha='right', va='center', **kw)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(labelsize=12)
    ax.set_xlabel('pairwise precision  (pairs put together that belong together)', fontsize=12.5)
    ax.set_ylabel('pairwise recall  (pairs that belong together and were put together)', fontsize=12.5)
    despine(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.11)
    save_figure(fig, 'FMI_plane')
    plt.close()
    print('4. FMI', {k: tuple(round(x, 3) for x in v) for k, v in pts.items()})


# ===========================================================================
# 5. CG — order-blind: two rankings of the same items
# ===========================================================================
def fig_cg():
    rels_a = np.array([3, 3, 2, 1, 0, 1, 0, 0, 2, 0])
    rels_b = rels_a[::-1].copy()          # same items, reversed
    K = len(rels_a)
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.8), sharex=True, gridspec_kw={'hspace': 0.55})
    out = {}
    for ax, rels, name in zip(axes, [rels_a, rels_b], ['best items first', 'best items last']):
        cg = rels.cumsum()
        dcg = (rels / np.log2(np.arange(2, K + 2))).cumsum()
        out[name] = (int(cg[-1]), round(float(dcg[-1]), 2))
        pos = np.arange(1, K + 1)
        colors = [nml_cmap(0.15 + 0.28 * r) if r > 0 else GREY for r in rels]
        ax.bar(pos, rels, color=colors, width=0.62, zorder=3)
        for x, r in zip(pos, rels):
            ax.text(x, r + 0.12, f'{r}', ha='center', va='bottom', fontsize=11, color=DARK if r else MID)
        ax.set_yticks([])
        ax.set_ylim(0, 4.4)
        ax.set_xticks(pos)
        ax.tick_params(axis='x', labelsize=11, length=0)
        despine(ax, keep=('bottom',))
        ax.text(K + 0.7, 2.0, f'CG@{K} = {cg[-1]}', fontsize=14, color=start_color, va='center', ha='left')
        ax.text(K + 0.7, 0.6, f'DCG@{K} = {dcg[-1]:.2f}', fontsize=12, color=MID, va='center', ha='left')
        ax.set_title(name, fontsize=13, loc='left', color=DARK, pad=6)
        ax.set_xlim(0.3, K + 4.5)
    axes[1].set_xlabel('rank position', fontsize=12)
    axes[0].text(0.5, 4.2, 'relevance grade of the item at each position', fontsize=11, color=MID, va='top')
    fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.14)
    save_figure(fig, 'CG_order_blind')
    plt.close()
    print('5. CG', out)


# ===========================================================================
# 6. MASE — model errors against the naive lag-1 forecast
# ===========================================================================
def spread(vals, gap):
    """Nudge values apart so labels stacked on them don't overlap (keeps order)."""
    order = np.argsort(vals)
    out = np.array(vals, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def fig_mase():
    T = 40
    t = np.arange(T)
    level = 50 + 12 * np.sin(2 * np.pi * t / 14) + 0.4 * t
    yv = level + RNG.normal(0, 3.0, T)
    naive = np.r_[np.nan, yv[:-1]]
    model = level + RNG.normal(0, 2.0, T) + 1.5 * np.sin(2 * np.pi * t / 14 + 1.0)
    e_model = np.abs(yv - model)
    e_naive = np.abs(yv - naive)
    mae_model = e_model[1:].mean()
    mae_naive = np.nanmean(e_naive)
    ratio = mae_model / mae_naive
    fig, (axT, axB) = book_figure(1.0, 2.75, 2, 1, sharex=True,
                                  gridspec_kw={'height_ratios': [1.2, 1]})
    axT.plot(t, naive, color=REF, lw=LW_THIN + 0.2, ls=(0, (3, 2)), zorder=2)
    axT.plot(t, model, color=NML_CYAN, zorder=3)
    axT.plot(t, yv, color=INK, lw=LW_THIN + 0.3, marker='o', ms=2.2, zorder=4)
    ends = spread([yv[-1], model[-1], naive[-1]], 7.5)
    for yy_, txt, col in zip(ends, ['actual', 'model forecast', "naive: yesterday's value"],
                             [INK, NML_CYAN, MUTED]):
        axT.text(T + 0.3, yy_, txt, color=col, va='center')
    axT.set_ylabel('demand')
    axT.set_yticks([40, 60, 80])
    axT.set_ylim(36, 84)
    tidy_axes(axT, bottom=False)
    w = 0.42
    axB.bar(t[1:] - w / 2, e_naive[1:], width=w, color=LIGHT, zorder=3, linewidth=0)
    axB.bar(t[1:] + w / 2, e_model[1:], width=w, color=NML_CYAN, zorder=3, linewidth=0)
    axB.plot([0, T - 1], [mae_naive] * 2, color=REF, lw=LW_THIN + 0.2, ls=(0, (3, 2)), zorder=4)
    axB.plot([0, T - 1], [mae_model] * 2, color=NML_CYAN, lw=LW_THIN + 0.2, ls=(0, (3, 2)), zorder=4)
    lab = spread([mae_model, mae_naive], 2.6)
    axB.text(T + 0.3, lab[1], f'naive MAE = {num(mae_naive)}', color=MUTED, va='center')
    axB.text(T + 0.3, lab[0], f'model MAE = {num(mae_model)}', color=NML_CYAN, va='center')
    axB.set_title(f'model MAE / naive MAE = {num(mae_model)} / {num(mae_naive)} = {num(ratio)}'
                  '  (same 39 days)', loc='left', fontsize=TEXT_PT)
    axB.set_ylabel('absolute error')
    axB.set_xlabel('day')
    axB.set_yticks([0, 5, 10, 15])
    axB.set_xlim(-1, T)
    axB.set_xticks(range(0, T + 1, 10))
    tidy_axes(axB)
    save_figure(fig, 'MASE_naive_baseline')
    plt.close(fig)
    print(f'MASE model MAE {mae_model:.3f} naive MAE {mae_naive:.3f} ratio {ratio:.3f}')


# ===========================================================================
# wMAPE: MAPE per channel vs each channel's contribution to wMAPE
# ===========================================================================
def fig_wmape():
    channels = ['B2B', 'Online', 'Marketplace', 'Retail']
    actual = np.array([110_000, 90_000, 120_000, 1_000_000], float)
    forecast = np.array([99_010, 79_990, 110_040, 900_000], float)
    err = np.abs(actual - forecast)
    mape_i = 100 * err / actual
    weight = actual / actual.sum()
    contrib = 100 * err / actual.sum()
    wmape = contrib.sum()
    mape = mape_i.mean()
    fig, (axL, axR) = book_figure(1.0, 1.95, 1, 2, sharey=True, sharex=True)
    yy = np.arange(len(channels))[::-1]
    axL.barh(yy, mape_i, color=LIGHT, height=0.62, zorder=3)
    for yi, v in zip(yy, mape_i):
        axL.text(v + 0.3, yi, f'{num(v, 1)}%', va='center', color=INK)
    axR.barh(yy, contrib, color=NML_CYAN, height=0.62, zorder=3)
    for yi, v, wgt in zip(yy, contrib, weight):
        axR.text(v + 0.3, yi, f'{num(v)}  ({100 * wgt:.0f}% of revenue)', va='center', color=INK)
    axL.set_title(f'MAPE per channel\nmean {num(mape, 1)}%', loc='left', fontsize=TEXT_PT, linespacing=1.3)
    axR.set_title(f'contribution to wMAPE\nsum {num(wmape, 1)}%', loc='left', fontsize=TEXT_PT, linespacing=1.3)
    axL.set_yticks(yy, labels=channels)
    axL.set_xlabel('error / actual (%)')
    axR.set_xlabel('error / total actual (points)')
    for ax in (axL, axR):
        ax.set_xlim(0, 12.5)
        ax.set_xticks([0, 5, 10])
        tidy_axes(ax)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axR.tick_params(axis='y', labelleft=False)
    save_figure(fig, 'wMAPE_compare_MAPE')
    plt.close(fig)
    print('wMAPE', dict(mape_i=np.round(mape_i, 1), weight=np.round(100 * weight, 1),
                        contrib=np.round(contrib, 2), wmape=round(wmape, 2), mape=round(mape, 2)))


FIGURES = {'MI_contingency_contributions': lambda: fig_mi(),
           'Homogeneity_cluster_composition': lambda: fig_homogeneity(),
           'V_Measure_plane': lambda: fig_vmeasure(), 'FMI_plane': lambda: fig_fmi(),
           'CG_order_blind': lambda: fig_cg(), 'MASE_naive_baseline': lambda: fig_mase(),
           'wMAPE_compare_MAPE': lambda: fig_wmape()}


if __name__ == '__main__':
    # Only MASE draws from RNG, so each figure can be rebuilt on its own.
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
