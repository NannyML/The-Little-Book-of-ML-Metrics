"""Figures for the Ranking chapter, drawn at printed size.

CG and DCG use one list and one color mapping (CG purple, DCG cyan); relevant
items are cyan throughout. Every printed number is computed here.
Run: uv run python notebooks/ranking_figures.py [NAME ...]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from style import *  # noqa: F401,F403

CG_COLOR, DCG_COLOR = NML_PURPLE, NML_CYAN
BEST_FIRST = np.array([3, 3, 2, 2, 1, 1, 0, 0, 0, 0])


def discount(n):
    return 1.0 / np.log2(np.arange(2, n + 2))


# ===========================================================================
def at_k():
    rel = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
                    0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    total = 20
    assert rel.sum() == total
    K = np.arange(1, 51)
    prec, rec = rel.cumsum() / K, rel.cumsum() / total
    fig, ax = book_figure(0.82, 1.9)
    ax.plot(K, prec, color=NML_CYAN, lw=LW - 0.2)
    ax.plot(K, rec, color=NML_PURPLE, lw=LW - 0.2)
    label_end(ax, 50, prec[-1], 'Precision@K', NML_CYAN)
    label_end(ax, 50, rec[-1], 'Recall@K', NML_PURPLE)
    ax.scatter([20], [prec[19]], s=16, facecolors='white', edgecolors=INK, linewidths=0.8, zorder=5)
    note(ax, 24, 0.22, f'equal at $K$ = 20: {num(prec[19])}', xy=(20.3, prec[19] - 0.02), ha='left',
         va='center')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    unit_ticks(ax, 'y', 0.5)
    ax.set_xlabel('$K$ (length of the top-$K$ list)')
    ax.set_ylabel('score')
    tidy_axes(ax)
    save_figure(fig, 'at_K_precision_recall')
    plt.close(fig)


# ===========================================================================
def _strip_header(ax, K):
    for r in range(1, K + 1):
        ax.text(r, 0, str(r), ha='center', va='bottom', color=MUTED, transform=ax.get_xaxis_transform())


def mrr():
    K = 10
    queries = [(1, []), (2, [5]), (3, [7]), (5, [10]), (10, []), (None, [])]
    fracs = {1: '1', 2: '1/2', 3: '1/3', 5: '1/5', 10: '1/10'}
    fig, ax = book_figure(1.0, 2.35)
    rrs = []
    n = len(queries)
    for row, (first, later) in enumerate(queries):
        y = n - 1 - row
        relevant = ([first] if first else []) + later
        for r in range(1, K + 1):
            if r not in relevant:
                ax.scatter(r, y, s=3, color=LIGHT, zorder=1, linewidths=0)
        for r in later:
            ax.scatter(r, y, s=30, facecolors='white', edgecolors=NML_CYAN, linewidths=1.0, zorder=2)
        if first:
            ax.scatter(first, y, s=36, color=NML_CYAN, zorder=3, linewidths=0)
        rr = 1 / first if first else 0.0
        rrs.append(rr)
        ax.text(-0.2, y, f'Q{row + 1}', ha='right', va='center', color=INK)
        ax.text(K + 1.6, y, fracs[first] if first else '0 (no hit)', ha='left', va='center',
                color=INK if first else NML_RED)
    mrr_v = float(np.mean(rrs))
    ax.text(K + 1.6, n - 0.35, 'RR', ha='left', va='bottom', color=MUTED)
    ax.plot([K + 1.5, K + 3.9], [-0.55, -0.55], color=INK, lw=0.6)
    ax.text(K + 1.6, -0.75, f'MRR = {num(mrr_v)}', ha='left', va='top', color=INK)
    for r in range(1, K + 1):
        ax.text(r, n - 0.35, str(r), ha='center', va='bottom', color=MUTED)
    ax.text(5.5, n + 0.25, 'rank', ha='center', va='bottom', color=MUTED)
    ax.set_xlim(-0.6, K + 4)
    ax.set_ylim(-1.3, n + 0.9)
    bare_axes(ax)
    handles = [Line2D([], [], marker='o', ls='none', ms=5, mfc=NML_CYAN, mec=NML_CYAN, label='first relevant item'),
               Line2D([], [], marker='o', ls='none', ms=4.5, mfc='white', mec=NML_CYAN, label='later relevant item'),
               Line2D([], [], marker='o', ls='none', ms=2, mfc=LIGHT, mec=LIGHT, label='irrelevant item')]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, handletextpad=0.2)
    save_figure(fig, 'MRR_ranked_lists')
    plt.close(fig)
    print(f'MRR {mrr_v:.4f}')


def map_strips():
    K = 10
    rows = [('good ranking', [1, 2, 3, 6]), ('poor ranking', [4, 7, 9, 10])]
    fig, ax = book_figure(1.0, 1.75)
    for row, (name, ranks) in enumerate(rows):
        y = len(rows) - 1 - row
        precs = [(i + 1) / r for i, r in enumerate(ranks)]
        ap = float(np.mean(precs))
        for r in range(1, K + 1):
            if r not in ranks:
                ax.scatter(r, y, s=3, color=LIGHT, linewidths=0)
        for r, p in zip(ranks, precs):
            ax.scatter(r, y, s=36, color=NML_CYAN, linewidths=0, zorder=3)
            ax.text(r, y - 0.3, num(p), ha='center', va='top', color=INK)
        ax.text(-0.2, y, name, ha='right', va='center', color=INK)
        ax.text(K + 1.0, y, f'AP = {num(ap)}', ha='left', va='center', color=INK)
        print(f'MAP {name}: AP {ap:.4f}')
    for r in range(1, K + 1):
        ax.text(r, 1.45, str(r), ha='center', va='bottom', color=MUTED)
    ax.text(5.5, 1.8, 'rank', ha='center', va='bottom', color=MUTED)
    ax.set_xlim(-0.4, K + 3.0)
    ax.set_ylim(-0.75, 2.1)
    bare_axes(ax)
    handles = [Line2D([], [], marker='o', ls='none', ms=5, mfc=NML_CYAN, mec=NML_CYAN,
                      label='relevant item, with the precision at its rank'),
               Line2D([], [], marker='o', ls='none', ms=2, mfc=LIGHT, mec=LIGHT, label='irrelevant item')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, handletextpad=0.2)
    save_figure(fig, 'MAP_precision_strips')
    plt.close(fig)


# ===========================================================================
def hit_rate():
    Ks = np.array([1, 3, 5, 10, 20, 50])
    hr = np.array([0.15, 0.35, 0.52, 0.72, 0.88, 0.96])
    fig, ax = book_figure(0.82, 1.95)
    ax.plot(Ks, 100 * hr, color=NML_CYAN, marker='o', ms=3, mec='white', mew=0.5, clip_on=False)
    for k, v, off, ha in [(1, hr[0], (5, -2), 'left'), (10, hr[3], (5, -7), 'left'),
                          (50, hr[5], (0, 6), 'right')]:
        ax.annotate(f'$K$ = {k}: {int(round(100 * v))}%', (k, 100 * v), xytext=off,
                    textcoords='offset points', ha=ha, va='center', color=INK)
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 25, 50, 75, 100], labels=['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('$K$ (length of the top-$K$ list)')
    ax.set_ylabel('users with a hit')
    tidy_axes(ax)
    save_figure(fig, 'Hit_Rate_vs_K')
    plt.close(fig)


# ===========================================================================
def cg_order_blind():
    lists = [('best items first', BEST_FIRST), ('best items last', BEST_FIRST[::-1])]
    fig, axes = book_figure(1.0, 2.15, 2, 1, sharex=True)
    d = discount(10)
    for ax, (name, rels) in zip(axes, lists):
        pos = np.arange(1, 11)
        ax.bar(pos, rels, color=NML_CYAN, width=0.62, linewidth=0)
        for x, r in zip(pos, rels):
            ax.text(x, r + 0.15, str(r), ha='center', va='bottom', color=INK if r else MUTED)
        cg, dcg = int(rels.sum()), float((rels * d).sum())
        ax.text(11.0, 1.9, f'CG@10 = {cg}', color=CG_COLOR, va='center')
        ax.text(11.0, 0.7, f'DCG@10 = {num(dcg)}', color=DCG_COLOR, va='center')
        ax.set_title(name, loc='left', fontsize=TEXT_PT, pad=4)
        ax.set_ylim(0, 3.9)
        ax.set_xlim(0.4, 13.8)
        tidy_axes(ax, left=False)
        ax.spines['bottom'].set_bounds(0.5, 10.5)
        print(f'CG {name}: CG {cg} DCG {dcg:.4f}')
    fig.get_layout_engine().set(hspace=0.12)
    axes[1].set_xticks(range(1, 11))
    axes[1].set_xlabel('rank position')
    axes[0].tick_params(axis='x', length=0)
    save_figure(fig, 'CG_order_blind')
    plt.close(fig)


def cg_vs_dcg():
    rels = BEST_FIRST
    pos = np.arange(1, 11)
    cg, dcg = rels.cumsum(), (rels * discount(10)).cumsum()
    fig, ax = book_figure(0.82, 2.1)
    ax.plot(pos, cg, color=CG_COLOR, marker='o', ms=2.6, mec='white', mew=0.4)
    ax.plot(pos, dcg, color=DCG_COLOR, marker='o', ms=2.6, mec='white', mew=0.4)
    label_end(ax, 10, cg[-1], f'CG@10 = {cg[-1]}', CG_COLOR, dx=12)
    label_end(ax, 10, dcg[-1], f'DCG@10 = {num(dcg[-1])}', DCG_COLOR, dx=12)
    gap = cg[-1] - dcg[-1]
    ax.annotate('', xy=(10.35, cg[-1]), xytext=(10.35, dcg[-1]),
                arrowprops=dict(arrowstyle='<->', color=REF, lw=LW_THIN, shrinkA=0, shrinkB=0,
                                mutation_scale=6))
    note(ax, 10.6, (cg[-1] + dcg[-1]) / 2, f'{num(gap)} lower\n({100 * gap / cg[-1]:.0f}%)',
         ha='left', va='center')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 13)
    ax.set_xticks(pos)
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_xlabel('rank $K$')
    ax.set_ylabel('cumulative score')
    tidy_axes(ax)
    save_figure(fig, 'CG_vs_DCG')
    plt.close(fig)
    print(f'CG vs DCG: CG {cg[-1]} DCG {dcg[-1]:.4f} gap {gap:.3f} ({100 * gap / cg[-1]:.1f}%)')


def ndcg_degradation():
    def ndcg(r):
        d = discount(len(r))
        return float((r * d).sum() / (np.sort(r)[::-1] * d).sum())
    np.random.seed(7)
    rels = np.array([5, 4, 3, 3, 2, 2, 1, 1, 0, 0], dtype=float)
    scores, cur = [ndcg(rels)], rels.copy()
    for _ in range(20):
        i, j = np.random.choice(len(cur), size=2, replace=False)
        cur[i], cur[j] = cur[j], cur[i]
        scores.append(ndcg(cur))
    scores = np.array(scores)
    worst = int(np.argmin(np.diff(scores))) + 1
    fig, ax = book_figure(0.82, 1.95)
    ax.plot([0, 20], [1, 1], color=REF, lw=LW_THIN, ls='--', zorder=1)
    ax.text(20, 1.02, 'perfect ranking', color=MUTED, ha='right', va='bottom')
    ax.plot(np.arange(21), scores, color=NML_CYAN, marker='o', ms=2.6, mec='white', mew=0.4, zorder=3)
    note(ax, worst + 1.5, 0.35, 'largest drop: a rank-1 item\nis demoted to rank 7',
         xy=(worst, scores[worst] - 0.03), ha='left', va='center')
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0, 5, 10, 15, 20])
    unit_ticks(ax, 'y', 0.5)
    ax.set_xlabel('number of random pair swaps')
    ax.set_ylabel('nDCG')
    tidy_axes(ax)
    save_figure(fig, 'nDCG_degradation')
    plt.close(fig)
    print('nDCG', np.round(scores, 3).tolist(), 'worst step', worst)


# ===========================================================================
def fcp():
    true = np.array([1, 2, 3, 4, 5])
    panels = [('good ranking', np.array([1, 3, 2, 4, 5]), NML_CYAN),
              ('poor ranking', np.array([4, 2, 5, 3, 1]), NML_RED)]
    fig, axes = book_figure(1.0, 2.15, 1, 2)
    for ax, (name, pred, col) in zip(axes, panels):
        conc = disc = 0
        for i in range(5):
            for j in range(i + 1, 5):
                if (true[i] - true[j]) * (pred[i] - pred[j]) > 0:
                    conc += 1
                else:
                    disc += 1
                    ax.plot([true[i], true[j]], [pred[i], pred[j]], color=NML_RED, lw=LW_THIN,
                            alpha=0.8, zorder=1)
        f = conc / (conc + disc)
        ax.scatter(true, pred, s=30, color=col, zorder=3, linewidths=0)
        ax.set_title(f'{name}: FCP = {conc}/{conc + disc} = {num(f)}', loc='left', fontsize=TEXT_PT)
        ax.set_xlim(0.6, 5.4)
        ax.set_ylim(0.6, 5.4)
        ax.set_xticks(range(1, 6))
        ax.set_yticks(range(1, 6))
        ax.set_aspect('equal')
        ax.set_xlabel('true preference rank')
        tidy_axes(ax)
        print(f'FCP {name}: {conc}/{conc + disc}')
    axes[0].set_ylabel('predicted rank')
    fig.legend(handles=[Line2D([], [], color=NML_RED, lw=LW_THIN, label='pair in the wrong order')],
               loc='outside lower center')
    save_figure(fig, 'FCP_comparison')
    plt.close(fig)


# ===========================================================================
def diversity():
    names = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Romance']
    colors = [NML_CYAN, '#1b5f9c', NML_PURPLE, '#8a2d8c', NML_RED, '#8f1414']
    cmap = dict(zip(names, colors))
    rows = [('six genres', names), ('one genre', ['Action'] * 6)]

    def div(g):
        pairs = [(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
        return sum(a != b for a, b in pairs) / len(pairs)
    fig, ax = book_figure(1.0, 1.25)
    for row, (label, genres) in enumerate(rows):
        y = 1 - row
        for i, g in enumerate(genres):
            ax.add_patch(Rectangle((i, y), 0.97, 0.8, facecolor=cmap[g], edgecolor='none'))
            ax.text(i + 0.48, y + 0.4, g, ha='center', va='center', color='white', fontsize=SMALL_PT)
        ax.text(-0.1, y + 0.4, label, ha='right', va='center', color=INK)
        ax.text(6.15, y + 0.4, f'diversity\n= {num(div(genres))}', ha='left', va='center', color=INK,
                linespacing=1.25)
    ax.set_xlim(-0.05, 7.25)
    ax.set_ylim(-0.1, 1.9)
    bare_axes(ax)
    save_figure(fig, 'Diversity_comparison')
    plt.close(fig)


def novelty():
    P = np.linspace(0.004, 1.0, 400)
    fig, ax = book_figure(0.82, 1.9)
    ax.plot(P, -np.log2(P), color=NML_CYAN)
    for p, txt, off, ha in [(0.01, 'rare item: $P(i)$ = 0.01,\n6.6 bits', (8, -3), 'left'),
                            (0.5, 'mainstream item: $P(i)$ = 0.5,\n1 bit', (6, 9), 'left')]:
        v = -np.log2(p)
        ax.scatter([p], [v], s=16, color=NML_CYAN, edgecolors='white', linewidths=0.5, zorder=4)
        ax.annotate(txt, (p, v), xytext=off, textcoords='offset points', ha=ha, va='center', color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8)
    unit_ticks(ax, 'x', 0.2)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xlabel('item popularity $P(i)$')
    ax.set_ylabel('novelty (bits)')
    tidy_axes(ax)
    save_figure(fig, 'Novelty_curve')
    plt.close(fig)


def serendipity():
    np.random.seed(5)
    n = 60
    rel = np.random.beta(2.0, 2.5, n)
    unexp = np.random.beta(2.0, 2.5, n)
    fig, ax = book_figure(0.82, 2.9)
    g = np.linspace(0.005, 1.0, 300)
    R, U = np.meshgrid(g, g)
    cs = ax.contour(R, U, R * U, levels=[0.25, 0.5], colors=[NML_PURPLE], linewidths=LW_THIN,
                    alpha=0.7)
    ax.clabel(cs, fmt=lambda v: f'product = {v:g}', fontsize=SMALL_PT, colors=NML_PURPLE, inline=True,
              inline_spacing=2, manual=[(0.6, 0.25 / 0.6), (0.82, 0.5 / 0.82)])
    ax.scatter(rel, unexp, s=12, color=NML_PURPLE, alpha=0.9, edgecolors='white', linewidths=0.4,
               zorder=3)
    for x, y, t, ha in [(0.02, 0.98, 'surprising but\nirrelevant', 'left'),
                        (0.98, 0.98, 'relevant and\nsurprising', 'right'),
                        (0.02, 0.02, 'irrelevant and\nobvious', 'left'),
                        (0.98, 0.02, 'relevant but\nobvious', 'right')]:
        ax.text(x, y, t, ha=ha, va='top' if y > 0.5 else 'bottom', color=MUTED, linespacing=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    unit_ticks(ax, 'x', 0.5)
    unit_ticks(ax, 'y', 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('relevance')
    ax.set_ylabel('unexpectedness')
    tidy_axes(ax)
    save_figure(fig, 'Serendipity_scatter')
    plt.close(fig)


def coverage():
    np.random.seed(0)
    size, cols, rows = 100, 20, 5
    systems = [('System A: popular items only', 0.12), ('System B: moderate spread', 0.48),
               ('System C: broad use of the catalog', 0.89)]
    fig, axes = book_figure(1.0, 2.2, 3, 1)
    for ax, (label, cov) in zip(axes, systems):
        idx = set(np.random.choice(size, int(round(size * cov)), replace=False))
        for k in range(size):
            r, c = divmod(k, cols)
            ax.add_patch(Rectangle((c, rows - 1 - r), 0.86, 0.86,
                                   facecolor=NML_CYAN if k in idx else LIGHT, edgecolor='none'))
        ax.set_title(f'{label}: coverage = {num(len(idx) / size)}', loc='left', fontsize=TEXT_PT, pad=3)
        ax.set_xlim(-0.1, cols)
        ax.set_ylim(-0.1, rows)
        ax.set_aspect('equal')
        bare_axes(ax)
    save_figure(fig, 'Coverage_comparison')
    plt.close(fig)


FIGURES = {'at_K_precision_recall': at_k, 'MRR_ranked_lists': mrr, 'MAP_precision_strips': map_strips,
           'Hit_Rate_vs_K': hit_rate, 'CG_order_blind': cg_order_blind, 'CG_vs_DCG': cg_vs_dcg,
           'nDCG_degradation': ndcg_degradation, 'FCP_comparison': fcp,
           'Diversity_comparison': diversity, 'Novelty_curve': novelty,
           'Serendipity_scatter': serendipity, 'Coverage_comparison': coverage}

if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
