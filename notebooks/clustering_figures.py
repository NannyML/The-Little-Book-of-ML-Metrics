"""Figures for the Clustering chapter, drawn at printed size.

Scenario names, class colors and table styling are shared by all eleven
figures. Every printed number is computed here.
Run: uv run python notebooks/clustering_figures.py [NAME ...]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, completeness_score,
                             davies_bouldin_score, fowlkes_mallows_score, homogeneity_score,
                             mutual_info_score, rand_score, silhouette_samples, silhouette_score,
                             v_measure_score)
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix
from style import *  # noqa: F401,F403

CLASS_COLORS = [NML_CYAN, NML_PURPLE, NML_RED]
COUNT_MAP = mcolors.LinearSegmentedColormap.from_list('counts', ['#ffffff', NML_CYAN])
DIV_MAP = mcolors.LinearSegmentedColormap.from_list('contrib', [NML_RED, '#ffffff', NML_CYAN])
K_ = '$k$'


def ink_for(rgb):
    """Black or white text, chosen from the rendered cell color."""
    lin = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 'white' if sum(c * w for c, w in zip(lin, [.2126, .7152, .0722])) < .3 else INK


def table_axes(ax):
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)


def scatter_axes(ax):
    bare_axes(ax)
    ax.set_aspect('equal', adjustable='datalim')


# ---------------------------------------------------------------------------
# One 3-class dataset and its clusterings, shared by MI, Homogeneity, V and FMI
# ---------------------------------------------------------------------------
N, K = 300, 3
X_SEP, Y = make_blobs(n_samples=N, centers=K, cluster_std=0.6, random_state=4)
X_OVL, _ = make_blobs(n_samples=N, centers=K, cluster_std=2.2, random_state=4)


def km(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)


CLUST = {
    'well separated': (km(X_SEP, 3), 3),
    'overlapping': (km(X_OVL, 3), 3),
    'over-split': (km(X_SEP, 6), 6),
    'one cluster': (np.zeros(N, dtype=int), 1),
}


def scenario(name):
    return f'{name}, {K_} = {CLUST[name][1]}'


def pair_pr(y_true, y_pred):
    c = contingency_matrix(y_true, y_pred)
    same_both = (c * (c - 1) / 2).sum()
    same_pred = (c.sum(0) * (c.sum(0) - 1) / 2).sum()
    same_true = (c.sum(1) * (c.sum(1) - 1) / 2).sum()
    return same_both / same_pred, same_both / same_true


# ===========================================================================
def fig_mi():
    fig, axes = book_figure(1.0, 2.15, 1, 2)
    hc = None
    for ax, name in zip(axes, ['well separated', 'overlapping']):
        pred = CLUST[name][0]
        c = contingency_matrix(Y, pred).astype(float)
        p = c / c.sum()
        pc, pk = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            contrib = np.where(p > 0, p * np.log(p / (pc @ pk)), 0.0)
        mi = contrib.sum()
        assert abs(mi - mutual_info_score(Y, pred)) < 1e-9
        hc = entropy(pc.ravel())
        vmax = 0.37
        im = ax.imshow(contrib, cmap=DIV_MAP, vmin=-vmax, vmax=vmax, aspect='auto')
        for i in range(K):
            for j in range(K):
                ink = ink_for(im.cmap(im.norm(contrib[i, j])))
                ax.text(j, i - 0.17, f'{int(c[i, j])}', ha='center', va='center', color=ink,
                        fontweight='semibold')
                txt = '0' if c[i, j] == 0 else num(contrib[i, j], 3, sign=True)
                ax.text(j, i + 0.2, txt, ha='center', va='center', color=ink, fontsize=SMALL_PT)
        ax.set_xticks(range(K), labels=[f'cluster {j}' for j in range(K)])
        ax.set_yticks(range(K), labels=[f'class {i}' for i in range(K)])
        ax.xaxis.tick_top()
        table_axes(ax)
        ax.set_title(f'{scenario(name)}\nMI = {num(mi, 3)} nats', loc='left', fontsize=TEXT_PT,
                     linespacing=1.3, pad=6)
        print(f'MI {name}: {mi:.4f}')
    fig.supxlabel(f'each cell: the count, and below it the contribution to MI (nats)\n'
                  f'largest possible MI here: H(class) = {num(hc, 3)} nats', fontsize=TEXT_PT,
                  color=MUTED, linespacing=1.4)
    save_figure(fig, 'MI_contingency_contributions')
    plt.close(fig)


# ===========================================================================
def fig_rand():
    X_s, y_s = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
    p_s = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_s).labels_
    X_o, y_o = make_blobs(n_samples=300, centers=3, cluster_std=3.8, random_state=42)
    p_o = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_o).labels_
    p_r = np.random.default_rng(7).integers(0, 3, size=len(y_s))
    fig, axes = book_figure(1.0, 1.95, 1, 3)
    for ax, X, y, pred, name in [(axes[0], X_s, y_s, p_s, 'well separated'),
                                 (axes[1], X_o, y_o, p_o, 'overlapping'),
                                 (axes[2], X_s, y_s, p_r, 'random labels')]:
        # align cluster ids with classes so disagreeing points can be marked
        cm = contingency_matrix(y, pred)
        r, cidx = linear_sum_assignment(-cm)
        mapping = {cl: cls for cls, cl in zip(r, cidx)}
        aligned = np.array([mapping[c] for c in pred])
        for c in range(3):
            m = aligned == c
            ax.scatter(X[m, 0], X[m, 1], color=CLASS_COLORS[c], s=4, linewidths=0, alpha=0.9)
        if name == 'overlapping':
            wrong = aligned != y
            ax.scatter(X[wrong, 0], X[wrong, 1], s=13, facecolors='none', edgecolors=INK,
                       linewidths=0.6)
            print('Rand overlapping: points outside their class cluster', int(wrong.sum()))
        scatter_axes(ax)
        ri, ari = rand_score(y, pred), adjusted_rand_score(y, pred)
        ax.set_title(name, fontsize=TEXT_PT)
        ax.set_xlabel(f'RI = {num(ri)}   ARI = {num(ari)}', labelpad=4)
        print(f'Rand {name}: RI {ri:.4f} ARI {ari:.4f}')
    save_figure(fig, 'Rand_Index_separation')
    plt.close(fig)


# ===========================================================================
def _elbow(name, score_fn, best_fn, ylabel, note_text, yticks, fmt=None):
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=42)
    ks = np.arange(2, 10)
    scores = np.array([score_fn(X, KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).labels_)
                       for k in ks])
    bi = int(best_fn(scores))
    fig, ax = book_figure(0.82, 2.2)
    ax.plot(ks, scores, color=NML_CYAN, marker='o', ms=3.2, mec='white', mew=0.6, clip_on=False)
    ax.scatter([ks[bi]], [scores[bi]], s=34, facecolors='white', edgecolors=NML_PURPLE,
               linewidths=1.1, zorder=5, clip_on=False)
    ax.annotate(note_text.format(k=ks[bi]), (ks[bi], scores[bi]), xytext=(9, 0),
                textcoords='offset points', color=NML_PURPLE, va='center',
                arrowprops=dict(arrowstyle='-', color=NML_PURPLE, lw=LW_THIN, shrinkA=0, shrinkB=4))
    ax.set_xticks(ks)
    ax.set_yticks(yticks)
    if fmt:
        fmt(ax)
    ax.set_xlabel('number of clusters $k$')
    ax.set_ylabel(ylabel)
    tidy_axes(ax)
    save_figure(fig, name)
    plt.close(fig)
    print(name, dict(zip(ks.tolist(), np.round(scores, 3).tolist())))


def fig_ch():
    _elbow('Calinski_Harabasz_elbow', calinski_harabasz_score, np.argmax, 'CH index',
           'peak at $k$ = {k}', [0, 2000, 4000, 6000, 8000, 10000, 12000],
           fmt=lambda ax: thousands_axis(ax, 'y'))


def fig_db():
    _elbow('Davies_Bouldin_elbow', davies_bouldin_score, np.argmin, 'Davies-Bouldin index',
           'minimum at $k$ = {k}', [0, 0.4, 0.8, 1.2],
           fmt=lambda ax: minus_axis(ax, 'y', 1))


# ===========================================================================
def _blobs_bad():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=42)
    return y, KMeans(n_clusters=3, random_state=42, n_init=10).fit(X).labels_


def fig_contingency():
    y, pred = _blobs_bad()
    cm = contingency_matrix(y, pred)
    _, col = linear_sum_assignment(-cm)
    cm = cm[:, col]
    fig, ax = book_figure(0.82, 2.2)
    im = ax.imshow(cm, cmap=COUNT_MAP, vmin=0, vmax=100, aspect='auto')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color=ink_for(im.cmap(im.norm(cm[i, j]))), fontweight='semibold')
    ax.set_xticks(range(3), labels=[f'cluster {j}' for j in range(3)])
    ax.set_yticks(range(3), labels=[f'class {i}' for i in range(3)])
    ax.xaxis.tick_top()
    ax.set_xlabel('cluster (reordered to match the classes)')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('reference class')
    for k in range(4):
        ax.axhline(k - 0.5, color='white', lw=2)
        ax.axvline(k - 0.5, color='white', lw=2)
    table_axes(ax)
    save_figure(fig, 'Contingency_Matrix_heatmap')
    plt.close(fig)
    print('Contingency', cm.tolist())


def fig_pair_confusion():
    y, pred = _blobs_bad()
    pcm = pair_confusion_matrix(y, pred)
    fig, ax = book_figure(0.82, 2.35)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    names = [['$C_{00}$: apart in both', '$C_{01}$: together only\nin the prediction'],
             ['$C_{10}$: together only\nin the reference', '$C_{11}$: together in both']]
    for i in range(2):
        for j in range(2):
            agree = i == j
            ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                   facecolor=CYAN_TINT if agree else PURPLE_TINT, edgecolor='none'))
            ax.text(j, i - 0.12, num(pcm[i, j], 0, thousands=True), ha='center', va='center',
                    fontsize=10, fontweight='semibold', color=NML_CYAN if agree else NML_PURPLE)
            ax.text(j, i + 0.2, names[i][j], ha='center', va='center', color=INK, linespacing=1.2)
    ax.set_xticks([0, 1], labels=['apart', 'together'])
    ax.set_yticks([0, 1], labels=['apart', 'together'])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('pair in the prediction')
    ax.set_ylabel('pair in the reference')
    table_axes(ax)
    save_figure(fig, 'Pair_Confusion_Matrix')
    plt.close(fig)
    print('Pair confusion', pcm.tolist(), 'sum', int(pcm.sum()))


# ===========================================================================
def fig_completeness():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
    runs = [(KMeans(n_clusters=3, random_state=42, n_init=10).fit(X).labels_, f'{K_} = 3, the true count'),
            (np.zeros(len(y), dtype=int), f'one cluster, {K_} = 1'),
            (KMeans(n_clusters=9, random_state=42, n_init=10).fit(X).labels_, f'over-split, {K_} = 9')]
    shades = {NML_CYAN: [NML_CYAN, '#05617c', '#86d4ec'],
              NML_PURPLE: [NML_PURPLE, '#8a63bd', '#cbbbe3'],
              NML_RED: [NML_RED, '#8f1414', '#f2a9a9']}
    fig, axes = book_figure(1.0, 2.1, 1, 3)
    for ax, (pred, title) in zip(axes, runs):
        cm = contingency_matrix(y, pred)
        for c in np.unique(pred):
            cls = int(np.argmax(cm[:, c]))                     # class this cluster belongs to
            siblings = [k for k in np.unique(pred) if int(np.argmax(cm[:, k])) == cls]
            if len(np.unique(pred)) == 1:
                color = MUTED
            else:
                color = shades[CLASS_COLORS[cls]][siblings.index(c)]
            m = pred == c
            ax.scatter(X[m, 0], X[m, 1], color=color, s=6, linewidths=0)
        scatter_axes(ax)
        comp, hom = completeness_score(y, pred), homogeneity_score(y, pred)
        ax.set_title(title, fontsize=TEXT_PT)
        ax.set_xlabel(f'Completeness = {num(comp)}\nHomogeneity = {num(hom)}', labelpad=4, linespacing=1.3)
        print(f'Completeness {title}: C {comp:.4f} H {hom:.4f}')
    save_figure(fig, 'Completeness_Score_tradeoff')
    plt.close(fig)


# ===========================================================================
def fig_homogeneity():
    names = ['well separated', 'overlapping', 'one cluster']
    hc = entropy(np.bincount(Y) / N)
    fig, axes = book_figure(1.0, 2.6, 1, 3, sharey=True,
                            gridspec_kw={'width_ratios': [3, 3, 2.1]})
    for ax, name in zip(axes, names):
        pred = CLUST[name][0]
        c = contingency_matrix(Y, pred).astype(float)
        h_cond = 0.0
        for j in range(c.shape[1]):
            col = c[:, j]
            hk = entropy(col / col.sum())
            h_cond += col.sum() / N * hk
            bottom = 0
            for i in range(K):
                ax.bar(j, col[i], bottom=bottom, color=CLASS_COLORS[i], width=0.66,
                       edgecolor='white', linewidth=0.5)
                bottom += col[i]
            ax.text(j, col.sum() + 6, num(hk), ha='center', va='bottom', color=INK)
        h = 1 - h_cond / hc
        assert abs(h - homogeneity_score(Y, pred)) < 1e-9
        ax.set_title(f'{scenario(name)}\nHomogeneity = {num(h)}\nH(class | cluster) = {num(h_cond)}',
                     loc='left', fontsize=TEXT_PT, linespacing=1.3)
        ax.set_xticks(range(c.shape[1]), labels=[str(j) for j in range(c.shape[1])])
        ax.set_xlim(-0.6, c.shape[1] - 0.4)
        ax.set_xlabel('cluster')
        tidy_axes(ax)
        ax.tick_params(axis='x', length=0)
        print(f'Homogeneity {name}: {h:.4f}, H(class|cluster) {h_cond:.4f}')
    axes[0].set_ylim(0, 330)
    axes[0].set_yticks([0, 100, 200, 300])
    axes[0].set_ylabel('points in the cluster')
    handles = [matplotlib.patches.Patch(color=CLASS_COLORS[i], label=f'class {i}') for i in range(K)]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, handlelength=1.0)
    save_figure(fig, 'Homogeneity_cluster_composition')
    plt.close(fig)


# ===========================================================================
def _plane(name, metric_fn, coords_fn, label, xlabel, ylabel, labels, extra=None):
    fig, ax = book_figure(0.82, 2.85)
    g = np.linspace(0.002, 1, 500)
    A, B = np.meshgrid(g, g)
    levels = [0.2, 0.4, 0.6, 0.8]
    cs = ax.contour(A, B, metric_fn(A, B), levels=levels, colors=[REF], linewidths=LW_THIN)
    ax.clabel(cs, fmt=lambda v: f'{label} = {v:g}', fontsize=SMALL_PT, colors=MUTED, inline=True,
              inline_spacing=2, manual=[(0.87, float(B[np.argmin(np.abs(metric_fn(0.87, g) - v)), 0]))
                                        for v in levels])
    if extra:
        extra(ax, A, B)
    pts = {}
    for key, color in zip(CLUST, [NML_CYAN, NML_PURPLE, NML_RED, INK]):
        pred = CLUST[key][0]
        x, yv, score = coords_fn(pred)
        pts[key] = (x, yv, score)
        ax.scatter(x, yv, s=30, color=color, zorder=6, edgecolors='white', linewidths=0.8,
                   clip_on=False)
    from matplotlib.lines import Line2D
    handles = []
    for key, color in zip(CLUST, [NML_CYAN, NML_PURPLE, NML_RED, INK]):
        x, yv, score = pts[key]
        handles.append(Line2D([], [], marker='o', ls='none', ms=4.5, mfc=color, mec='white', mew=0.6,
                              label=f'{key if key != "over-split" else "over-split, " + K_ + " = 6"}: {label} = {num(score)}'))
    order = [3, 0, 1, 2]
    fig.legend(handles=[handles[i] for i in order], loc='outside lower center', ncol=2,
               handletextpad=0.2, columnspacing=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    unit_ticks(ax, 'x', 0.5)
    unit_ticks(ax, 'y', 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    tidy_axes(ax)
    save_figure(fig, name)
    plt.close(fig)
    return pts


def fig_vmeasure():
    pts = _plane('V_Measure_plane', lambda h, c: 2 * h * c / (h + c),
                 lambda p: (homogeneity_score(Y, p), completeness_score(Y, p), v_measure_score(Y, p)),
                 'V', 'homogeneity $h$ (each cluster holds one class)',
                 'completeness $c$\n(each class in one cluster)',
                 {'one cluster': (0.04, 0.95, 'left', 'top'),
                  'well separated': (0.96, 0.95, 'right', 'top'),
                  'overlapping': (0.61, 0.66, 'right', 'center'),
                  'over-split': (0.96, 0.55, 'right', 'top')})
    print('V-measure', {k: tuple(round(v, 4) for v in t) for k, t in pts.items()})


def fig_fmi():
    def f1_curve(ax, A, B):
        F1 = 2 * A * B / (A + B)
        cs = ax.contour(A, B, F1, levels=[0.5], colors=[NML_PURPLE], linewidths=LW_THIN,
                        linestyles=[(0, (3, 2))])
        ax.clabel(cs, fmt=lambda v: 'pairwise F1 = 0.5', fontsize=SMALL_PT, colors=NML_PURPLE,
                  inline=True, inline_spacing=2, manual=[(0.55, 0.5 * 0.55 / (2 * 0.55 - 0.5))])
    pts = _plane('FMI_plane', lambda p, r: np.sqrt(p * r),
                 lambda p: (*pair_pr(Y, p), fowlkes_mallows_score(Y, p)),
                 'FMI', 'pairwise precision', 'pairwise recall',
                 {'one cluster': (0.37, 0.95, 'left', 'top'),
                  'well separated': (0.96, 0.95, 'right', 'top'),
                  'overlapping': (0.72, 0.77, 'right', 'center'),
                  'over-split': (0.96, 0.44, 'right', 'top')}, extra=f1_curve)
    print('FMI', {k: tuple(round(v, 4) for v in t) for k, t in pts.items()})


# ===========================================================================
def fig_silhouette():
    X_g, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
    X_b, _ = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=42)
    fig, axes = book_figure(1.0, 2.1, 1, 2)
    for ax, X, title in [(axes[0], X_g, 'well separated'), (axes[1], X_b, 'overlapping')]:
        labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X).labels_
        sil = silhouette_samples(X, labels)
        avg = silhouette_score(X, labels)
        lo, centers = 5, []
        for i in range(3):
            s = np.sort(sil[labels == i])
            ax.fill_betweenx(np.arange(lo, lo + len(s)), 0, s, facecolor=CLASS_COLORS[i], linewidth=0)
            centers.append(lo + len(s) / 2)
            lo += len(s) + 12
        ax.axvline(avg, color=INK, ls=(0, (3, 2)), lw=LW_THIN + 0.1)
        ax.text(avg + 0.03, lo + 2, f'mean $S$ = {num(avg)}', color=INK, ha='left', va='bottom')
        ax.set_yticks(centers, labels=[f'cluster {i}' for i in range(3)])
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(-0.2, 1)
        ax.set_ylim(0, lo + 22)
        ax.set_xticks([0, 0.5, 1.0], labels=['0.0', '0.5', '1.0'])
        ax.set_xlabel('silhouette coefficient $s(i)$')
        ax.set_title(title, loc='left', fontsize=TEXT_PT)
        tidy_axes(ax)
        ax.spines['left'].set_visible(False)
        print(f'Silhouette {title}: {avg:.4f}, min {sil.min():.3f}')
    save_figure(fig, 'Silhouette_Score_comparison')
    plt.close(fig)


FIGURES = {'MI_contingency_contributions': fig_mi, 'Rand_Index_separation': fig_rand,
           'Calinski_Harabasz_elbow': fig_ch, 'Contingency_Matrix_heatmap': fig_contingency,
           'Pair_Confusion_Matrix': fig_pair_confusion, 'Completeness_Score_tradeoff': fig_completeness,
           'Homogeneity_cluster_composition': fig_homogeneity, 'V_Measure_plane': fig_vmeasure,
           'Davies_Bouldin_elbow': fig_db, 'FMI_plane': fig_fmi,
           'Silhouette_Score_comparison': fig_silhouette}

if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
