"""Generate figures for the GenAI chapter.

Every number printed in a figure is computed here, either exactly (MAUVE, IS,
FID on constructed distributions; MOS on constructed ratings; DSG on a worked
example) or from real model outputs stored in `data/genai/*.json` by
`data/genai/gen_model_data.py` (GPT-2 surprisals, BERTScore matrices, CLIP
cosines, BLIP P(yes), LPIPS, PESQ, STOI).  Nothing is typed in by hand.

Run from notebooks/:  uv run python genai_plots.py
"""
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
from style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import linalg, signal

DATA = Path(__file__).resolve().parent / 'data' / 'genai'
GREY = '#c9c9c9'
GREY_LINE = '#9a9a9a'
DARK = '#2a2a2a'
MID = '#6f6f6f'
CYAN_MAP = mcolors.LinearSegmentedColormap.from_list('cyan_map', ['#ffffff', start_color])
RNG = np.random.default_rng(7)


def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


def load(name):
    with open(DATA / f'{name}.json') as f:
        return json.load(f)


def clean_tok(t):
    t = t.replace('Ġ', ' ').replace('##', '·')
    return t.strip() if t.strip() else t


# ===========================================================================
# 1. Perplexity — per-token surprisal under GPT-2, two sentences
# ===========================================================================
def fig_perplexity():
    p = load('perplexity')
    nat, swp = p['natural'], p['one_swap']
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=False)
    ymax = max(max(nat['surprisal']), max(swp['surprisal'])) + 2.2
    for ax, d, key in zip(axes, [nat, swp], ['natural', 'one_swap']):
        s = np.array(d['surprisal'])
        toks = [clean_tok(t) for t in d['tokens']]
        x = np.arange(len(s))
        colors = [start_color] * len(s)
        if key == 'one_swap':
            base = np.array(nat['surprisal'])
            for i, t in enumerate(toks):
                if t == 'Ohio':
                    colors[i] = end_color
                elif s[i] - base[i] > 1.5:
                    colors[i] = middle_color
        ax.bar(x, s, color=colors, width=0.62, zorder=3)
        m = s.mean()
        ax.axhline(m, color=GREY_LINE, lw=1.2, ls=(0, (5, 3)), zorder=2)
        ax.text(len(s) - 0.5, ymax - 0.3, f'mean {m:.2f} nats  →  PPL = e$^{{{m:.2f}}}$ = {d["ppl"]:.1f}',
                ha='right', va='top', fontsize=13, color=DARK)
        ax.text(-0.45, m + 0.15, 'mean', ha='left', va='bottom', fontsize=10.5, color=MID)
        ax.text(-0.5, ymax - 0.3, '“' + d['text'] + '”', ha='left', va='top', fontsize=12.5,
                color=MID, style='italic')
        ax.set_xticks(x)
        ax.set_xticklabels(toks, fontsize=12)
        ax.set_yticks([0, 5, 10])
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.7, len(s) - 0.3)
        ax.tick_params(axis='y', labelsize=12)
        ax.spines['left'].set_bounds(0, 10)
        despine(ax, keep=('left',))
        ax.tick_params(axis='x', length=0)
        if key == 'one_swap':
            i = toks.index('Ohio')
            ax.annotate(f'p = e$^{{-{s[i]:.1f}}}$ = {np.exp(-s[i]):.5f}', xy=(i, s[i]), xytext=(i - 3.2, s[i] + 0.4),
                        fontsize=12, color=end_color, ha='right', va='center',
                        arrowprops=dict(arrowstyle='-', color=end_color, lw=1))
            j = [k for k, t in enumerate(toks) if t == 'France'][0]
            ax.annotate('“France” after “Ohio”\nis now a surprise too', xy=(j, s[j]), xytext=(j - 2.6, s[j] + 2.6),
                        fontsize=11.5, color=middle_color, ha='center', va='bottom',
                        arrowprops=dict(arrowstyle='-', color=middle_color, lw=1))
    axes[0].set_ylabel('surprisal  −log p  (nats)', fontsize=13)
    axes[1].set_ylabel('surprisal  −log p  (nats)', fontsize=13)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.1, hspace=0.36)
    save_figure(fig, 'Perplexity_surprisal')
    plt.close()
    print(f"1. PPL natural={nat['ppl']:.1f} swap={swp['ppl']:.1f} shuffled={p['shuffled']['ppl']:.0f}")


# ===========================================================================
# 2. BERTScore — greedy token matching matrix + BERTScore vs unigram precision
# ===========================================================================
def fig_bertscore():
    b = load('bertscore')
    pair = b['pairs']['paraphrase']
    sim = np.array(pair['sim'])
    rt = [clean_tok(t) for t in pair['ref_tokens']]
    ct = [clean_tok(t) for t in pair['cand_tokens']]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.0, 1.15], 'wspace': 0.22})

    # left: similarity matrix
    axL.imshow(sim, cmap=CYAN_MAP, vmin=0.2, vmax=1.0, aspect='auto')
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            v = sim[i, j]
            axL.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=12,
                     color='white' if v > 0.62 else DARK)
    rmax = sim.argmax(axis=1)
    cmax = sim.argmax(axis=0)
    for i, j in enumerate(rmax):   # recall: each reference token's best candidate token
        axL.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=DARK, lw=2.2))
    for j, i in enumerate(cmax):   # precision: each candidate token's best reference token
        axL.plot(j + 0.36, i - 0.36, marker='o', ms=6, color=end_color, zorder=5)
    axL.set_xticks(range(len(ct)))
    axL.set_xticklabels(ct, fontsize=12.5)
    axL.set_yticks(range(len(rt)))
    axL.set_yticklabels(rt, fontsize=12.5)
    axL.xaxis.tick_top()
    axL.set_xlabel('candidate:  “' + pair['cand'] + '”', fontsize=12, labelpad=8)
    axL.xaxis.set_label_position('top')
    axL.set_ylabel('reference:  “' + pair['ref'] + '”', fontsize=12)
    axL.tick_params(length=0)
    for s in axL.spines.values():
        s.set_visible(False)
    axL.text(-0.5, len(rt) - 0.25, f'□ row max  →  R = mean = {pair["R"]:.2f}\n'
                                    f'• column max  →  P = mean = {pair["P"]:.2f}\n'
                                    f'F$_{{BERT}}$ = {pair["F"]:.2f}      unigram precision = {pair["bleu1"]:.2f}',
             fontsize=11.5, color=DARK, va='top', ha='left', linespacing=1.5, transform=axL.transData,
             clip_on=False)
    axL.set_ylim(len(rt) + 1.1, -0.5)

    # right: BERTScore F vs unigram precision for four candidate types
    order = ['paraphrase', 'unrelated', 'role_swap', 'negation']
    labels = {'paraphrase': 'paraphrase', 'unrelated': 'unrelated', 'role_swap': 'roles swapped', 'negation': 'negated'}
    step = 1.35
    ys = np.arange(len(order))[::-1] * step
    for y, k in zip(ys, order):
        d = b['pairs'][k]
        axR.plot([d['bleu1'], d['F']], [y, y], color=GREY, lw=2.5, zorder=2)
        axR.scatter(d['bleu1'], y, s=110, color=GREY_LINE, zorder=4)
        axR.scatter(d['F'], y, s=130, color=start_color, zorder=5)
        axR.text(0, y + 0.55, labels[k], ha='left', va='center', fontsize=13, color=DARK)
        axR.text(0, y + 0.30, '“' + d['ref'] + '”  →  “' + d['cand'] + '”', ha='left', va='center',
                 fontsize=10, color=MID, style='italic')
        if abs(d['F'] - d['bleu1']) < 0.12:
            axR.text(d['F'], y - 0.3, f'{d["F"]:.2f}', ha='center', va='center', fontsize=11.5, color=start_color)
            axR.text(d['bleu1'], y - 0.3, f'{d["bleu1"]:.2f}', ha='center', va='center', fontsize=11.5, color=MID)
        else:
            axR.text(d['F'] + 0.03, y, f'{d["F"]:.2f}', ha='left', va='center', fontsize=11.5, color=start_color)
            axR.text(d['bleu1'] - 0.03, y, f'{d["bleu1"]:.2f}', ha='right', va='center', fontsize=11.5, color=MID)
    top = ys[0] + 0.55
    hF = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=start_color, markersize=11)
    hB = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=GREY_LINE, markersize=10)
    axR.legend([hF, hB], ['F$_{BERT}$', 'unigram precision (BLEU-1)'], loc='upper right', frameon=True, framealpha=1,
               edgecolor=GREY, fontsize=12, handletextpad=0.2, borderpad=0.6)
    axR.set_xlim(-0.02, 1.08)
    axR.set_ylim(-0.75, top + 1.3)
    axR.set_xticks([0, 0.5, 1.0])
    axR.tick_params(axis='x', labelsize=12)
    axR.set_yticks([])
    axR.spines['bottom'].set_bounds(0, 1)
    despine(axR, keep=('bottom',))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.09)
    save_figure(fig, 'BERTScore_matching')
    plt.close()
    print("2. BERTScore:", {k: round(v['F'], 3) for k, v in b['pairs'].items()})


# ===========================================================================
# 3. MAUVE — divergence frontier computed exactly on quantized histograms
# ===========================================================================
def kl(a, b):
    m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / b[m])))


def mauve_curve(P, Q, c=5, n=600):
    lam = np.linspace(1e-6, 1 - 1e-6, n)
    pts = [(0.0, 1.0)]  # extreme point, as in the reference implementation
    for w in lam:
        R = w * P + (1 - w) * Q
        pts.append((np.exp(-c * kl(Q, R)), np.exp(-c * kl(P, R))))
    pts.append((1.0, 0.0))
    pts = np.array(pts)
    o = np.argsort(pts[:, 0])
    auc = float(np.trapezoid(pts[o, 1], pts[o, 0]))
    return pts, auc


def fig_mauve():
    P = np.array([.28, .22, .18, .14, .10, .08, 0, 0])
    Qs = {
        'close to human': (np.array([.25, .24, .17, .15, .11, .08, 0, 0]), start_color),
        'junk: text people never write': (0.75 * P + 0.25 * np.array([0, 0, 0, 0, 0, 0, .5, .5]), end_color),
        'missing modes: only the common text': (np.array([.28, .22, .18, 0, 0, 0, 0, 0]) / .68, middle_color),
    }
    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[0.9, 1.25], wspace=0.28, hspace=0.55)
    axR = fig.add_subplot(gs[:, 1])
    K = len(P)
    xb = np.arange(K)
    results = {}
    for row, (name, (Q, color)) in enumerate(Qs.items()):
        ax = fig.add_subplot(gs[row, 0])
        ax.bar(xb - 0.19, P, width=0.36, color=GREY, zorder=2)
        ax.bar(xb + 0.19, Q, width=0.36, color=color, zorder=3)
        ax.set_xticks(xb)
        ax.set_xticklabels([f'{i + 1}' for i in xb], fontsize=10.5)
        ax.set_yticks([0, 0.2, 0.4])
        ax.set_yticklabels(['0', '.2', '.4'], fontsize=9.5)
        ax.set_ylim(0, 0.46)
        ax.spines['left'].set_bounds(0, 0.4)
        despine(ax, keep=('bottom', 'left'))
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', labelsize=9.5, length=3)
        pts, auc = mauve_curve(P, Q)
        results[name] = (pts, auc, color)
        ax.set_title(name, fontsize=12.5, loc='left', pad=4, color=color)
        if row == 0:
            ax.text(K - 0.6, 0.44, 'share of texts: human P (grey)  vs  model Q', ha='right', va='top', fontsize=10.5, color=MID)
        if row == 2:
            ax.set_xlabel('embedding cluster', fontsize=12)
    for name, (pts, auc, color) in results.items():
        o = np.argsort(pts[:, 0])
        axR.plot(pts[o, 0], pts[o, 1], color=color, lw=3.2, solid_capstyle='round', zorder=4)
        axR.fill_between(pts[o, 0], 0, pts[o, 1], color=color, alpha=0.10, zorder=1)
    # direct labels
    axR.text(0.55, 0.93, f'MAUVE = {results["close to human"][1]:.2f}', color=start_color, fontsize=13, ha='center')
    axR.text(0.27, 0.62, f'MAUVE = {results["junk: text people never write"][1]:.2f}', color=end_color, fontsize=13, ha='center')
    axR.text(0.72, 0.30, f'MAUVE = {results["missing modes: only the common text"][1]:.2f}', color=middle_color, fontsize=13, ha='center')
    axR.set_xlim(0, 1.0)
    axR.set_ylim(0, 1.0)
    axR.set_xticks([0, 0.5, 1])
    axR.set_yticks([0, 0.5, 1])
    axR.tick_params(labelsize=12)
    axR.set_xlabel('exp(−c · KL(Q ‖ R$_λ$))      ← falls when the model writes junk', fontsize=12.5)
    axR.set_ylabel('exp(−c · KL(P ‖ R$_λ$))      ← falls when modes are missing', fontsize=12.5)
    despine(axR)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.12)
    save_figure(fig, 'MAUVE_frontier')
    plt.close()
    print("3. MAUVE:", {k: round(v[1], 3) for k, v in results.items()})


# ===========================================================================
# 4. Inception Score — p(y|x) per image, marginal p(y), IS = exp(mean KL)
# ===========================================================================
def inception_score(pyx):
    py = pyx.mean(axis=0)
    kls = [kl(r, py) for r in pyx]
    return float(np.exp(np.mean(kls))), py, float(np.mean(kls))


def fig_is():
    K = 4
    hi, lo = 0.94, 0.02
    sharp_div = np.array([[hi if j == i else lo for j in range(K)] for i in range(K)])
    sharp_one = np.array([[hi if j == 0 else lo for j in range(K)] for i in range(K)])
    blurry = np.array([[0.40 if j == i else 0.20 for j in range(K)] for i in range(K)])
    panels = [('sharp and diverse', sharp_div), ('sharp, but one class', sharp_one), ('blurry', blurry)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw={'wspace': 0.35})
    out = {}
    for ax, (name, pyx) in zip(axes, panels):
        IS, py, mkl = inception_score(pyx)
        out[name] = IS
        grid = np.vstack([pyx, np.full((1, K), np.nan), py[None, :]])
        ax.imshow(grid, cmap=CYAN_MAP, vmin=0, vmax=1, aspect='auto')
        for i in range(K):
            for j in range(K):
                v = pyx[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=11.5, color='white' if v > 0.6 else DARK)
        for j in range(K):
            ax.text(j, K + 1, f'{py[j]:.2f}', ha='center', va='center', fontsize=11.5, color='white' if py[j] > 0.6 else DARK)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f'class {j + 1}' for j in range(K)], fontsize=11)
        ax.set_yticks(list(range(K)) + [K + 1])
        ax.set_yticklabels([f'image {i + 1}' for i in range(K)] + ['marginal p(y)'], fontsize=11)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'IS = {IS:.2f}', fontsize=16, pad=34, color=DARK)
        ax.text(0.5, 1.035, name, ha='center', va='bottom', fontsize=12.5, color=MID, transform=ax.transAxes)
        ax.text(0.5, -0.16, f'mean KL(p(y|x) ‖ p(y)) = {mkl:.2f} nats', ha='center', va='top', fontsize=11,
                color=MID, transform=ax.transAxes)
        ax.text(K - 0.5, K, 'p(y|x) rows', ha='right', va='center', fontsize=9.5, color=MID)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.14)
    save_figure(fig, 'IS_mechanism')
    plt.close()
    print("4. IS:", {k: round(v, 2) for k, v in out.items()}, f"(max {K})")


# ===========================================================================
# 5. FID — 2-D feature-space analogue, exact Fréchet distance between fits
# ===========================================================================
def fid(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a.T), np.cov(b.T)
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = covmean.real
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    cov_term = float(np.trace(s1 + s2 - 2 * covmean))
    return mean_term + cov_term, mean_term, cov_term


def fig_fid():
    n = 500
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    real = RNG.multivariate_normal([0, 0], cov, n)
    shifted = RNG.multivariate_normal([1.0, 0.5], cov, n)
    collapsed = RNG.multivariate_normal([0, 0], 0.05 * cov, n)
    d = np.array([0.85, 0.85])          # two clusters along the main axis, same total mean/covariance
    S = cov - np.outer(d, d)
    half = n // 2
    bimodal = np.vstack([RNG.multivariate_normal(d, S, half), RNG.multivariate_normal(-d, S, n - half)])
    panels = [('mean shifted', shifted, start_color), ('variety collapsed', collapsed, middle_color),
              ('two clouds, same mean and covariance', bimodal, end_color)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), gridspec_kw={'wspace': 0.12})
    out = {}
    for ax, (name, gen, color) in zip(axes, panels):
        F, mt, ct = fid(real, gen)
        out[name] = (F, mt, ct)
        ax.scatter(real[:, 0], real[:, 1], s=9, color=GREY, zorder=2, linewidths=0)
        ax.scatter(gen[:, 0], gen[:, 1], s=9, color=color, zorder=3, linewidths=0, alpha=0.85)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'FID = {F:.2f}', fontsize=16, pad=32, color=DARK)
        ax.text(0.5, 1.03, f'mean term {mt:.2f}  +  covariance term {ct:.2f}', ha='center', va='bottom',
                fontsize=11, color=MID, transform=ax.transAxes)
        ax.text(0.5, -0.02, name, ha='center', va='top', fontsize=12.5, color=color, transform=ax.transAxes)
    axes[0].text(-3.4, 3.3, 'real (grey)  vs  generated', ha='left', va='top', fontsize=11, color=MID)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.08)
    save_figure(fig, 'FID_mechanism')
    plt.close()
    print("5. FID:", {k: tuple(round(x, 2) for x in v) for k, v in out.items()})


# ===========================================================================
# 6. LPIPS — four distortions with the same pixel MSE, very different LPIPS
# ===========================================================================
def fig_lpips():
    l = load('lpips')
    order = ['shift', 'bright', 'noise', 'blur']
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.9), gridspec_kw={'wspace': 0.06})
    axes[0].imshow(plt.imread(DATA / 'lpips_original.png'))
    axes[0].set_title('original', fontsize=13, pad=8, color=DARK)
    axes[0].text(0.5, -0.08, 'MSE 0 · LPIPS 0', ha='center', va='top', fontsize=12, color=MID, transform=axes[0].transAxes)
    for ax, k in zip(axes[1:], order):
        v = l['variants'][k]
        ax.imshow(plt.imread(DATA / f'lpips_{k}.png'))
        ax.set_title(v['label'], fontsize=13, pad=8, color=DARK)
        ax.text(0.5, -0.08, f'MSE {v["mse"]:.3f}  (PSNR {v["psnr"]:.1f} dB)', ha='center', va='top', fontsize=11,
                color=MID, transform=ax.transAxes)
        c = nml_cmap(min(v['lpips'] / 0.9, 1.0))
        ax.text(0.5, -0.24, f'LPIPS {v["lpips"]:.2f}', ha='center', va='top', fontsize=15, color=c,
                transform=ax.transAxes, fontweight='bold')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.2)
    save_figure(fig, 'LPIPS_equal_mse')
    plt.close()
    print("6. LPIPS:", {k: (round(v['mse'], 4), round(v['lpips'], 3)) for k, v in l['variants'].items()})


# ===========================================================================
# 7. CLIP Score — image × caption similarity matrix with real CLIP cosines
# ===========================================================================
def fig_clip():
    c = load('clip')
    cos = np.array(c['cos'])
    w = c['w']
    images = c['images']
    caps = [cap for _, cap in c['captions']]
    tags = [t for t, _ in c['captions']]
    nI, nC = cos.shape
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(nI, nC + 1, width_ratios=[0.9] + [1] * nC, wspace=0.05, hspace=0.05)
    # thumbnails
    for i, k in enumerate(images):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(plt.imread(DATA / f'img_{k}.png'))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ax = fig.add_subplot(gs[:, 1:])
    score = w * np.clip(cos, 0, None)
    ax.imshow(score, cmap=CYAN_MAP, vmin=0.15, vmax=0.9, aspect='auto')
    for i in range(nI):
        for j in range(nC):
            v = score[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=14, color='white' if v > 0.6 else DARK)
    # outline the caption written for each image (the first four), and the two probes for the first image
    for i, k in enumerate(images):
        j = tags.index(k)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=DARK, lw=2.4))
    for probe in ('hopper_swap', 'hopper_neg'):
        j = tags.index(probe)
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 1, fill=False, edgecolor=end_color, lw=2.4, ls=(0, (4, 2))))
    ax.set_xticks(range(nC))
    ax.set_xticklabels([textwrap.fill(cp, 16) for cp in caps], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, nC - 0.5)
    ax.set_ylim(nI - 0.5, -0.5)
    ax.text(nC - 0.6, nI - 0.35, f'cell = CLIP-S = {w:g} · cos;  raw cosines span {cos.min():.2f}–{cos.max():.2f}',
            ha='right', va='top', fontsize=11, color=MID, clip_on=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.08)
    save_figure(fig, 'CLIP_Score_matrix')
    plt.close()
    print("7. CLIP-S diagonal:", [round(score[i, tags.index(k)], 2) for i, k in enumerate(images)],
          "swap/neg:", round(score[0, tags.index('hopper_swap')], 2), round(score[0, tags.index('hopper_neg')], 2))


# ===========================================================================
# 8. DSG — question graph for one prompt scored on two images (worked example)
# ===========================================================================
def fig_dsg():
    prompt = 'a red bicycle leaning against a blue wall'
    Q = {
        'q1': ('Is there a bicycle?', []),
        'q2': ('Is the bicycle red?', ['q1']),
        'q3': ('Is there a wall?', []),
        'q4': ('Is the wall blue?', ['q3']),
        'q5': ('Is the bicycle leaning\nagainst the wall?', ['q1', 'q3']),
    }
    pos = {'q1': (0.17, 0.80), 'q3': (0.83, 0.80), 'q2': (0.17, 0.34), 'q4': (0.83, 0.34), 'q5': (0.50, 0.34)}
    images = [
        ('image A: a blue bicycle leaning on a red wall', {'q1': 'yes', 'q2': 'no', 'q3': 'yes', 'q4': 'no', 'q5': 'yes'}),
        ('image B: a blue wall, no bicycle', {'q1': 'no', 'q2': 'yes', 'q3': 'yes', 'q4': 'yes', 'q5': 'yes'}),
    ]

    def dsg_score(ans):
        credit = {}
        for q, (_, parents) in Q.items():
            credit[q] = ans[q] == 'yes' and all(ans[p] == 'yes' for p in parents)
        skipped = [q for q, (_, parents) in Q.items() if any(ans[p] != 'yes' for p in parents)]
        return sum(credit.values()), len(Q), skipped

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={'wspace': 0.08})
    fills = {'yes': ('#dff4fb', start_color), 'no': ('#fbe3e3', end_color), 'skip': ('#f0f0f0', GREY_LINE)}
    out = {}
    for ax, (title, ans) in zip(axes, images):
        got, tot, skipped = dsg_score(ans)
        out[title] = (got, tot)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        # edges first
        for q, (_, parents) in Q.items():
            for p in parents:
                x0, y0 = pos[p]
                x1, y1 = pos[q]
                col = GREY_LINE if q in skipped else DARK
                ax.add_patch(FancyArrowPatch((x0, y0 - 0.075), (x1, y1 + 0.075), arrowstyle='-|>', mutation_scale=14,
                                             color=col, lw=1.3, shrinkA=0, shrinkB=0, zorder=2,
                                             connectionstyle='arc3,rad=0.0'))
        for q, (text, parents) in Q.items():
            x, y = pos[q]
            state = 'skip' if q in skipped else ans[q]
            face, edge = fills[state]
            ax.add_patch(FancyBboxPatch((x - 0.15, y - 0.075), 0.30, 0.15, boxstyle='round,pad=0.01,rounding_size=0.03',
                                        facecolor=face, edgecolor=edge, lw=2, zorder=3,
                                        linestyle='--' if state == 'skip' else '-'))
            ax.text(x, y + 0.012, text, ha='center', va='center', fontsize=11.5, color=DARK, zorder=4)
            if state == 'skip':
                lab = f'skipped (VQA said “{ans[q]}”)'
            else:
                lab = f'VQA: {ans[q]}'
            ax.text(x, y - 0.052, lab, ha='center', va='center', fontsize=10, color=edge, zorder=4, style='italic')
        ax.set_title(title, fontsize=13, pad=6, color=DARK)
        ax.text(0.5, 0.985, 'prompt: “' + prompt + '”', ha='center', va='top', fontsize=11.5, color=MID)
        ax.text(0.5, 0.12, f'DSG = {got} / {tot} = {got / tot:.2f}', ha='center', va='top', fontsize=15, color=DARK,
                fontweight='bold')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02)
    save_figure(fig, 'DSG_question_graph')
    plt.close()
    print("8. DSG:", out)


# ===========================================================================
# 9. VQAScore vs CLIP Score — one image, six captions, two scales
# ===========================================================================
def fig_vqascore():
    c = load('clip')
    v = load('vqascore')
    tags = [t for t, _ in c['captions']]
    caps = [cp for _, cp in c['captions']]
    i = c['images'].index('hopper')
    clip_s = c['w'] * np.clip(np.array(c['cos'][i]), 0, None)
    p_yes = np.array(v['p_yes'][v['images'].index('hopper')])
    # the caption written for the image, its word-order swap, its negation, and one wrong object
    keep = [tags.index(t) for t in ('hopper', 'hopper_swap', 'hopper_neg', 'dahlia')]
    order = [j for j in np.argsort(-p_yes) if j in keep]
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.34, 1.0], wspace=0.02)
    axI = fig.add_subplot(gs[0, 0])
    axI.imshow(plt.imread(DATA / 'img_hopper.png'))
    axI.axis('off')
    ax = fig.add_subplot(gs[0, 1])
    ys = np.arange(len(order))[::-1]
    for y, j in zip(ys, order):
        col = end_color if tags[j] in ('hopper_swap', 'hopper_neg') else DARK
        note = {'hopper': 'the caption written for the image', 'hopper_swap': 'word order swapped',
                'hopper_neg': 'negated', 'dahlia': 'a different object'}[tags[j]]
        ax.text(-0.06, y - 0.16, note, ha='right', va='center', fontsize=10.5, color=MID, style='italic')
        ax.plot([clip_s[j], p_yes[j]], [y, y], color=GREY, lw=2.5, zorder=2)
        ax.scatter(clip_s[j], y, s=110, color=GREY_LINE, zorder=4)
        ax.scatter(p_yes[j], y, s=130, color=start_color, zorder=5)
        ax.text(-0.06, y + 0.14, '“' + caps[j] + '”', ha='right', va='center', fontsize=11.5, color=col)
        if abs(p_yes[j] - clip_s[j]) < 0.1:
            ax.text(p_yes[j], y + 0.32, f'{p_yes[j]:.2f}', ha='center', va='center', fontsize=11.5, color=start_color)
            ax.text(clip_s[j], y - 0.32, f'{clip_s[j]:.2f}', ha='center', va='center', fontsize=11.5, color=MID)
        else:
            ax.text(p_yes[j] + 0.025, y, f'{p_yes[j]:.2f}', ha='left', va='center', fontsize=11.5, color=start_color)
            ax.text(clip_s[j] - 0.025, y, f'{clip_s[j]:.2f}', ha='right', va='center', fontsize=11.5, color=MID)
    top = ys[0] + 0.55
    ax.text(1.0, top + 0.5, 'VQAScore  P(“yes”)', ha='right', va='center', fontsize=12.5, color=start_color)
    ax.text(1.0, top + 0.9, 'CLIP-S', ha='right', va='center', fontsize=12.5, color=MID)
    ax.scatter([0.56], [top + 0.5], s=130, color=start_color)
    ax.scatter([0.83], [top + 0.9], s=110, color=GREY_LINE)
    ax.set_xlim(-0.95, 1.06)
    ax.set_ylim(-0.7, len(order) - 0.3 + 1.35)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='x', labelsize=12)
    ax.set_yticks([])
    ax.spines['bottom'].set_bounds(0, 1)
    despine(ax, keep=('bottom',))
    fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.12)
    save_figure(fig, 'VQAScore_vs_CLIP')
    plt.close()
    print("9. VQAScore:", [(tags[j], round(p_yes[j], 2), round(clip_s[j], 2)) for j in order])


# ===========================================================================
# 10. MOS — three rating distributions with the same mean
# ===========================================================================
def fig_mos():
    n = 20
    systems = {
        'consensus': [3] * n,
        'polarized': [1] * 10 + [5] * 10,
        'skewed': [2] * 8 + [3] * 6 + [4] * 4 + [5] * 2,
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True, gridspec_kw={'wspace': 0.12})
    out = {}
    for ax, (name, r) in zip(axes, systems.items()):
        r = np.array(r)
        mos, med, sd = r.mean(), np.median(r), r.std(ddof=1)
        ci = 1.96 * sd / np.sqrt(n)
        out[name] = (mos, med, ci)
        for rating in range(1, 6):
            k = int(np.sum(r == rating))
            ax.scatter([rating] * k, np.arange(k) + 1, s=95, color=start_color, zorder=3, linewidths=0)
        ax.axvline(mos, color=GREY_LINE, lw=1.3, ls=(0, (5, 3)), zorder=1)
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.4, 5.6)
        ax.set_ylim(0, 22.5)
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=12, length=0)
        despine(ax, keep=('bottom',))
        ax.set_title(f'MOS = {mos:.1f}', fontsize=16, pad=30, color=DARK)
        ax.text(0.5, 1.03, name, ha='center', va='bottom', fontsize=12.5, color=MID, transform=ax.transAxes)
        ax.text(0.97, 0.95, f'median {med:.0f}\n95% CI ± {ci:.2f}', ha='right', va='top', fontsize=12, color=DARK,
                transform=ax.transAxes, linespacing=1.5)
        ax.set_xlabel('listener rating', fontsize=12)
    axes[0].text(0.03, 0.95, f'{n} listeners\n(one dot each)', ha='left', va='top', fontsize=11, color=MID,
                 transform=axes[0].transAxes, linespacing=1.4)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.14)
    save_figure(fig, 'MOS_same_mean')
    plt.close()
    print("10. MOS:", {k: tuple(round(x, 2) for x in v) for k, v in out.items()})


# ===========================================================================
# 11. PESQ — alignment: what waveform SNR punishes and PESQ does not
# ===========================================================================
def fig_pesq():
    a = load('audio')
    order = ['clean', 'delay', 'gain', 'telephone', 'noise40', 'noise30', 'noise20', 'noise10']
    labels = {'clean': 'clean reference', 'delay': 'delayed by 20 ms', 'gain': 'gain −6 dB',
              'telephone': 'band-limited 300–3400 Hz', 'noise40': 'white noise, 40 dB SNR',
              'noise30': 'white noise, 30 dB SNR', 'noise20': 'white noise, 20 dB SNR', 'noise10': 'white noise, 10 dB SNR'}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.08})
    ys = np.arange(len(order))[::-1]
    for y, k in zip(ys, order):
        d = a[k]
        col = end_color if k in ('delay', 'gain') else start_color
        snr = d['snr_db']
        if snr is None:
            axL.text(44, y, '∞', ha='center', va='center', fontsize=15, color=col)
        else:
            axL.scatter(snr, y, s=120, color=col, zorder=4)
            axL.text(snr + (1.5 if snr < 30 else -1.5), y, f'{snr:.1f}', ha='left' if snr < 30 else 'right', va='center',
                     fontsize=11.5, color=col)
        axL.text(-9, y, labels[k], ha='right', va='center', fontsize=12.5, color=DARK)
        p = d['pesq_nb']
        axR.scatter(p, y, s=120, color=col, zorder=4)
        axR.text(p + 0.08, y, f'{p:.2f}', ha='left', va='center', fontsize=11.5, color=col)
    axL.set_xlim(-8, 46)
    axL.set_xticks([0, 10, 20, 30, 40])
    axL.spines['bottom'].set_bounds(0, 40)
    axL.set_xlabel('waveform SNR (dB)  —  how far apart the samples are', fontsize=12.5)
    axR.set_xlim(0.8, 5.0)
    axR.set_xticks([1, 2, 3, 4, 4.5])
    axR.spines['bottom'].set_bounds(1, 4.5)
    axR.set_xlabel('PESQ (MOS-LQO)  —  how different it sounds', fontsize=12.5)
    for ax in (axL, axR):
        ax.set_yticks([])
        ax.set_ylim(-0.7, len(order) + 0.2)
        ax.tick_params(axis='x', labelsize=12)
        despine(ax, keep=('bottom',))
        ax.axvline(0 if ax is axL else 4.5, color=GREY_LINE, lw=0.8, alpha=0.5, zorder=1)
    axR.text(4.5, len(order) - 0.45, 'transparent', ha='center', va='bottom', fontsize=11, color=MID)
    axL.text(0, len(order) - 0.45, 'signal = distortion', ha='center', va='bottom', fontsize=11, color=MID)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.9, bottom=0.14)
    save_figure(fig, 'PESQ_alignment')
    plt.close()
    print("11. PESQ:", {k: (a[k]['snr_db'] and round(a[k]['snr_db'], 1), round(a[k]['pesq_nb'], 2)) for k in order})


# ===========================================================================
# 12. STOI — band envelopes and the band × segment correlation map
# ===========================================================================
def third_octave_bands(fs, nfft, num_bands=15, min_freq=150):
    f = np.linspace(0, fs, nfft + 1)[: nfft // 2 + 1]
    k = np.arange(num_bands)
    cf = 2 ** (k / 3) * min_freq
    lo = cf * 2 ** (-1 / 6)
    hi = cf * 2 ** (1 / 6)
    H = np.zeros((num_bands, len(f)))
    for i in range(num_bands):
        H[i, (f >= lo[i]) & (f < hi[i])] = 1
    return H, cf


def stoi_intermediate(x, y, fs_in):
    """STOI as in Taal et al. (2011): 10 kHz, 256-point frames with 50% overlap,
    silent-frame removal, 15 one-third-octave bands, 30-frame segments,
    normalization + clipping at -15 dB, per-(band, segment) correlation."""
    fs = 10000
    x = signal.resample_poly(x, fs, fs_in)
    y = signal.resample_poly(y, fs, fs_in)
    N, hop, nseg, beta = 256, 128, 30, -15
    win = np.hanning(N + 2)[1:-1]
    frames = np.arange(0, len(x) - N, hop)
    ex = np.array([np.sum((x[i:i + N] * win) ** 2) for i in frames])
    keep = 10 * np.log10(ex + 1e-12) > 10 * np.log10(ex.max()) - 40
    frames = frames[keep]
    X = np.array([np.abs(np.fft.rfft(x[i:i + N] * win, N)) ** 2 for i in frames]).T
    Y = np.array([np.abs(np.fft.rfft(y[i:i + N] * win, N)) ** 2 for i in frames]).T
    H, cf = third_octave_bands(fs, N)
    Xb = np.sqrt(H @ X)   # bands x frames
    Yb = np.sqrt(H @ Y)
    J, M = Xb.shape[0], Xb.shape[1] - nseg + 1
    c = 10 ** (-beta / 20)
    d = np.zeros((J, Xb.shape[1] // nseg))
    for m in range(d.shape[1]):
        sl = slice(m * nseg, (m + 1) * nseg)
        xs, ys = Xb[:, sl], Yb[:, sl]
        alpha = np.linalg.norm(xs, axis=1, keepdims=True) / (np.linalg.norm(ys, axis=1, keepdims=True) + 1e-12)
        yn = np.minimum(ys * alpha, xs * (1 + c))
        xc = xs - xs.mean(axis=1, keepdims=True)
        yc = yn - yn.mean(axis=1, keepdims=True)
        d[:, m] = np.sum(xc * yc, axis=1) / (np.linalg.norm(xc, axis=1) * np.linalg.norm(yc, axis=1) + 1e-12)
    return d, Xb, Yb, cf, frames, fs, hop


def fig_stoi():
    a = load('audio')
    fs_in = 16000
    clean = np.load(DATA / 'audio_clean.npy')
    noisy = np.load(DATA / 'audio_noise0.npy')
    d, Xb, Yb, cf, frames, fs, hop = stoi_intermediate(clean, noisy, fs_in)
    my_stoi = float(d.mean())
    lib_stoi = a['noise0']['stoi']
    band = int(np.argmin(np.abs(cf - 1000)))   # the band nearest 1 kHz
    seg_len = 30
    n_show = 3                                    # three consecutive segments
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.25], hspace=0.5)
    axT = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    # top: one band's envelope, clean vs noisy, over three 384 ms segments (noisy normalized+clipped as STOI does)
    s0 = 2
    sl = slice(s0 * seg_len, (s0 + n_show) * seg_len)
    t = np.arange(sl.start, sl.stop) * hop / fs
    xs = Xb[band, sl]
    ys = Yb[band, sl]
    axT.plot(t, xs, color=start_color, lw=2.6, solid_capstyle='round', zorder=4)
    axT.plot(t, ys * (np.linalg.norm(xs) / np.linalg.norm(ys)), color=end_color, lw=2.2, solid_capstyle='round', zorder=3)
    for m in range(n_show):
        t0 = (s0 + m) * seg_len * hop / fs
        t1 = (s0 + m + 1) * seg_len * hop / fs
        axT.axvline(t1, color=GREY_LINE, lw=0.8, alpha=0.6, zorder=1)
        axT.text((t0 + t1) / 2, axT.get_ylim()[1] if False else xs.max() * 1.08, f'corr = {d[band, s0 + m]:.2f}',
                 ha='center', va='bottom', fontsize=12, color=DARK)
    axT.text(t[-1], xs.max() * 0.98, 'clean', color=start_color, fontsize=12.5, ha='right', va='top')
    axT.text(t[-1], xs.max() * 0.80, 'noisy, 0 dB SNR', color=end_color, fontsize=12.5, ha='right', va='top')
    axT.set_xlim(t[0], t[-1])
    axT.set_yticks([])
    axT.set_ylim(0, xs.max() * 1.3)
    axT.set_xlabel('time (s)  —  segments of 384 ms', fontsize=12)
    axT.set_ylabel(f'{cf[band]:.0f} Hz band\nenvelope', fontsize=11.5)
    despine(axT, keep=('bottom',))
    axT.tick_params(axis='x', labelsize=11)
    # bottom: band x segment correlation map
    im = axB.imshow(d, cmap=CYAN_MAP, vmin=0, vmax=1, aspect='auto', origin='lower')
    axB.set_yticks([0, 4, 8, 12, 14])
    axB.set_yticklabels([f'{cf[i]:.0f} Hz' for i in [0, 4, 8, 12, 14]], fontsize=11)
    axB.set_xticks(np.arange(0, d.shape[1], 2))
    axB.set_xticklabels([f'{i}' for i in np.arange(0, d.shape[1], 2)], fontsize=11)
    axB.set_xlabel('segment (384 ms each)', fontsize=12)
    axB.set_ylabel('one-third-octave band', fontsize=12)
    for s in axB.spines.values():
        s.set_visible(False)
    axB.tick_params(length=0)
    # mark the strip shown above
    axB.add_patch(plt.Rectangle((s0 - 0.5, band - 0.5), n_show, 1, fill=False, edgecolor=DARK, lw=1.8))
    axB.text(d.shape[1] - 0.5, d.shape[0] + 0.4, f'STOI = mean of all cells = {my_stoi:.2f}', ha='right', va='bottom',
             fontsize=13.5, color=DARK)
    cb = fig.colorbar(im, ax=axB, fraction=0.025, pad=0.01)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=10)
    cb.set_label('envelope correlation', fontsize=10.5, labelpad=6)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.09)
    save_figure(fig, 'STOI_envelopes')
    plt.close()
    print(f"12. STOI: re-implementation {my_stoi:.3f} vs pystoi {lib_stoi:.3f} (0 dB); "
          f"10 dB {a['noise10']['stoi']:.3f}, -5 dB {a['noise-5']['stoi']:.3f}, delay {a['delay']['stoi']:.3f}")


if __name__ == '__main__':
    which = sys.argv[1:] or ['perplexity', 'bertscore', 'mauve', 'is', 'fid', 'lpips', 'clip', 'dsg', 'vqascore',
                             'mos', 'pesq', 'stoi']
    for w in which:
        globals()[f'fig_{w}']()
    print("\nGenAI figures regenerated.")
