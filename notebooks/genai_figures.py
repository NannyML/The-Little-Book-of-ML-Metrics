"""Figures for the GenAI chapter, drawn at printed size.

Every number is computed here, exactly on constructed data (MAUVE, IS, FID,
MOS, DSG) or from model outputs stored in data/genai/*.json by
data/genai/gen_model_data.py (GPT-2, BERT, CLIP, BLIP, LPIPS, PESQ, STOI).
Run: uv run python notebooks/genai_figures.py [NAME ...]
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import linalg, signal
from scipy.stats import t as student_t
from style import *  # noqa: F401,F403

DATA = Path(__file__).resolve().parent / 'data' / 'genai'
CYAN_MAP = mcolors.LinearSegmentedColormap.from_list('cyan_map', ['#ffffff', NML_CYAN])


def load(name):
    with open(DATA / f'{name}.json') as f:
        return json.load(f)


def clean_tok(t):
    t = t.replace('Ġ', ' ').replace('##', '·')
    return t.strip() if t.strip() else t


def kl(a, b):
    m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / b[m])))


# ===========================================================================
def perplexity():
    p = load('perplexity')
    nat, swp = p['natural'], p['one_swap']
    fig, axes = book_figure(1.0, 2.62, 1, 2, sharex=True)
    for ax, d, key, name in zip(axes, [nat, swp], ['natural', 'swap'], ['Paris', 'Ohio']):
        s = np.array(d['surprisal'])
        toks = [clean_tok(t) for t in d['tokens']]
        y = np.arange(len(s))[::-1]
        colors = [NML_CYAN] * len(s)
        if key == 'swap':
            colors[toks.index('Ohio')] = NML_RED
            colors[toks.index('France')] = NML_PURPLE
        ax.barh(y, s, color=colors, height=0.66, linewidth=0, zorder=3)
        m = s.mean()
        ax.axvline(m, color=REF, lw=LW_THIN, ls=(0, (3, 2)), zorder=4)
        ax.text(m + 0.15, len(s) - 0.4, f'mean {num(m)}', color=MUTED, va='center', fontsize=SMALL_PT)
        ax.set_title(f'…in {name}, the capital of France.\nperplexity = exp({num(m)}) = {num(d["ppl"], 1)}',
                     loc='left', fontsize=TEXT_PT, linespacing=1.3)
        ax.set_yticks(y, labels=toks, fontsize=SMALL_PT)
        ax.set_xticks([0, 5, 10])
        ax.set_xlim(0, 10.6)
        ax.set_ylim(-0.6, len(s) - 0.1)
        ax.set_xlabel('surprisal (nats)')
        tidy_axes(ax)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        if key == 'swap':
            i, j = toks.index('Ohio'), toks.index('France')
            ax.annotate(f'p = {np.exp(-s[i]):.5f}', (s[i], y[i]), xytext=(-3, -9), textcoords='offset points',
                        ha='right', va='center', color=NML_RED, fontsize=SMALL_PT)
            ax.annotate('now surprising too', (s[j], y[j]), xytext=(4, 0), textcoords='offset points',
                        ha='left', va='center', color=NML_PURPLE, fontsize=SMALL_PT)
    save_figure(fig, 'Perplexity_surprisal')
    plt.close(fig)
    print(f"Perplexity natural {nat['ppl']:.2f} swap {swp['ppl']:.2f}")


# ===========================================================================
def bertscore():
    pair = load('bertscore')['pairs']['paraphrase']
    sim = np.array(pair['sim'])
    rt = [clean_tok(t) for t in pair['ref_tokens']]
    ct = [clean_tok(t) for t in pair['cand_tokens']]
    fig = book_canvas(1.0, 3.0)
    ax = fig.add_axes([0.17, 0.27, 0.8, 0.5])
    im = ax.imshow(sim, cmap=CYAN_MAP, vmin=0.2, vmax=1.0, aspect='auto')
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, num(sim[i, j]), ha='center', va='center', fontsize=SMALL_PT,
                    color=ink_for(im.cmap(im.norm(sim[i, j]))))
    for i, j in enumerate(sim.argmax(axis=1)):
        ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, lw=1.1))
    for j, i in enumerate(sim.argmax(axis=0)):
        ax.plot(j + 0.33, i - 0.28, marker='o', ms=3.2, color=NML_RED, zorder=5)
    ax.set_xticks(range(len(ct)), labels=ct)
    ax.set_yticks(range(len(rt)), labels=rt)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.0, 0.985, f'reference (rows): “{pair["ref"]}”', va='top', color=INK)
    fig.text(0.0, 0.92, f'candidate (columns): “{pair["cand"]}”', va='top', color=INK)
    key = fig.add_axes([0.17, 0.0, 0.8, 0.2])
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    bare_axes(key)
    key.add_patch(Rectangle((0.0, 0.76), 0.022, 0.14, fill=False, edgecolor=INK, lw=1.1))
    key.text(0.04, 0.83, f'row maximum: recall R = mean = {num(pair["R"], 3)}', va='center', color=INK)
    key.plot(0.011, 0.5, marker='o', ms=3.2, color=NML_RED)
    key.text(0.04, 0.5, f'column maximum: precision P = mean = {num(pair["P"], 3)}', va='center', color=INK)
    key.text(0.04, 0.17, f'BERTScore F = {num(pair["F"], 3)}; unigram precision = {num(pair["bleu1"])}',
             va='center', color=INK)
    save_figure(fig, 'BERTScore_matching', crop=False)
    plt.close(fig)
    print('BERTScore', round(pair['F'], 4), 'unigram', pair['bleu1'])


# ===========================================================================
def mauve_curve(P, Q, c=5, n=600):
    lam = np.linspace(1e-6, 1 - 1e-6, n)
    pts = [(0.0, 1.0)]
    for w in lam:
        R = w * P + (1 - w) * Q
        pts.append((np.exp(-c * kl(Q, R)), np.exp(-c * kl(P, R))))
    pts.append((1.0, 0.0))
    pts = np.array(pts)
    o = np.argsort(pts[:, 0])
    return pts[o], float(np.trapezoid(pts[o, 1], pts[o, 0]))


def mauve():
    P = np.array([.28, .22, .18, .14, .10, .08, 0, 0])
    Qs = [('close to human', np.array([.25, .24, .17, .15, .11, .08, 0, 0]), NML_CYAN),
          ('adds junk text', 0.75 * P + 0.25 * np.array([0, 0, 0, 0, 0, 0, .5, .5]), NML_RED),
          ('misses rare modes', np.array([.28, .22, .18, 0, 0, 0, 0, 0]) / .68, NML_PURPLE)]
    fig = book_canvas(1.0, 2.6, layout='constrained')
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0])
    af = fig.add_subplot(gs[:, 1])
    xb = np.arange(len(P))
    out = {}
    for row, (name, Q, col) in enumerate(Qs):
        ah = fig.add_subplot(gs[row, 0])
        ah.bar(xb - 0.19, P, width=0.36, color=LIGHT, linewidth=0)
        ah.bar(xb + 0.19, Q, width=0.36, color=col, linewidth=0)
        ah.set_xticks(xb, labels=[str(i + 1) for i in xb] if row == 2 else [])
        ah.set_yticks([0, 0.4], labels=['0.0', '0.4'])
        ah.set_ylim(0, 0.42)
        ah.set_title(f'{name}: MAUVE = {num(mauve_curve(P, Q)[1])}', loc='left', fontsize=TEXT_PT, color=col, pad=2)
        tidy_axes(ah)
        ah.tick_params(axis='x', length=0)
        if row == 2:
            ah.set_xlabel('embedding cluster')
        if row == 1:
            ah.set_ylabel('share of texts')
        pts, area = mauve_curve(P, Q)
        out[name] = (pts, area, col)
    jpts, jarea, jcol = out['adds junk text']
    af.fill_between(jpts[:, 0], 0, jpts[:, 1], color=jcol, alpha=0.15, linewidth=0)
    for name, (pts, area, col) in out.items():
        af.plot(pts[:, 0], pts[:, 1], color=col, clip_on=False)
    af.text(0.06, 0.08, f'shaded area under the\nred curve = {num(jarea)}', color=NML_RED, linespacing=1.2)
    af.set_xlim(0, 1)
    af.set_ylim(0, 1)
    af.set_aspect('equal')
    unit_ticks(af, 'x', 0.5)
    unit_ticks(af, 'y', 0.5)
    af.set_xlabel('exp(\u2212c KL(Q \u2016 R))')
    af.set_ylabel('exp(\u2212c KL(P \u2016 R))')
    tidy_axes(af)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=LIGHT, label='human text P'),
                        Rectangle((0, 0), 1, 1, color=MUTED, label='model text Q (colored)')],
               loc='outside upper left', ncol=2, handlelength=1.0)
    save_figure(fig, 'MAUVE_frontier')
    plt.close(fig)
    print('MAUVE', {k: round(v[1], 4) for k, v in out.items()})


# ===========================================================================
def inception():
    K = 4
    hi, lo = 0.94, 0.02
    panels = [('confident,\nvaried classes', np.array([[hi if j == i else lo for j in range(K)] for i in range(K)])),
              ('confident,\none class', np.array([[hi if j == 0 else lo for j in range(K)] for i in range(K)])),
              ('uncertain\npredictions', np.array([[0.40 if j == i else 0.20 for j in range(K)] for i in range(K)]))]
    fig, axes = book_figure(1.0, 2.3, 1, 3)
    out = {}
    for k, (ax, (name, pyx)) in enumerate(zip(axes, panels)):
        py = pyx.mean(axis=0)
        mkl = float(np.mean([kl(r, py) for r in pyx]))
        IS = float(np.exp(mkl))
        out[name] = (IS, mkl)
        grid = np.vstack([pyx, np.full((1, K), np.nan), py[None, :]])
        im = ax.imshow(grid, cmap=CYAN_MAP, vmin=0, vmax=1, aspect='auto')
        for i, row in enumerate(list(pyx) + [None, py]):
            if row is None:
                continue
            for j, v in enumerate(row):
                ax.text(j, i, f'{v:.2f}'.lstrip('0') if v < 1 else '1', ha='center', va='center',
                        fontsize=SMALL_PT, color=ink_for(im.cmap(im.norm(v))))
        ax.set_xticks(range(K), labels=[str(j + 1) for j in range(K)])
        ax.set_yticks(list(range(K)) + [K + 1],
                      labels=([f'image {i + 1}' for i in range(K)] + ['average']) if k == 0 else [''] * (K + 1))
        ax.set_xlabel('class')
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f'{name}\nIS = {num(IS)}', loc='left', fontsize=TEXT_PT, linespacing=1.3)
    save_figure(fig, 'IS_mechanism')
    plt.close(fig)
    print('IS', {k: tuple(round(x, 3) for x in v) for k, v in out.items()})


# ===========================================================================
def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a.T), np.cov(b.T)
    covmean = linalg.sqrtm(s1 @ s2).real
    mt = float(np.sum((mu1 - mu2) ** 2))
    ct = float(np.trace(s1 + s2 - 2 * covmean))
    return mt + ct, mt, ct


def fid():
    rng = np.random.default_rng(7)
    n = 500
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    real = rng.multivariate_normal([0, 0], cov, n)
    shifted = rng.multivariate_normal([1.0, 0.5], cov, n)
    collapsed = rng.multivariate_normal([0, 0], 0.05 * cov, n)
    d = np.array([0.85, 0.85])
    S = cov - np.outer(d, d)
    bimodal = np.vstack([rng.multivariate_normal(d, S, n // 2), rng.multivariate_normal(-d, S, n - n // 2)])
    panels = [('mean shifted', shifted), ('variety collapsed', collapsed),
              ('two clouds, same mean\nand covariance', bimodal)]
    fig, axes = book_figure(1.0, 2.25, 1, 3)
    out = {}
    for ax, (name, gen) in zip(axes, panels):
        F, mt, ct = frechet(real, gen)
        out[name.replace('\n', ' ')] = (F, mt, ct)
        ax.scatter(real[:, 0], real[:, 1], s=1.5, color=LIGHT, linewidths=0)
        ax.scatter(gen[:, 0], gen[:, 1], s=1.5, color=NML_PURPLE, linewidths=0, alpha=0.9)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.set_aspect('equal')
        bare_axes(ax)
        ax.set_title(f'{name}', loc='center', fontsize=TEXT_PT, linespacing=1.2)
        ax.set_xlabel(f'FID = {num(F)}\nmean term {num(mt, 3)}\ncovariance term {num(ct, 3)}', linespacing=1.3)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=3, color=LIGHT, label='real'),
                        Line2D([], [], marker='o', ls='', ms=3, color=NML_PURPLE, label='generated')],
               loc='outside upper center', ncol=2, handletextpad=0.2)
    save_figure(fig, 'FID_mechanism')
    plt.close(fig)
    print('FID', {k: tuple(round(x, 4) for x in v) for k, v in out.items()})


# ===========================================================================
def lpips():
    l = load('lpips')
    labels = {'shift': 'shift 4 px', 'bright': 'brighten +0.16', 'noise': 'noise \u03c3 = 0.16',
              'blur': 'blur \u03c3 = 16 px'}
    fig, axes = book_figure(1.0, 1.55, 1, 5)
    cells = [('original', None)] + [(k, l['variants'][k]) for k in ['shift', 'bright', 'noise', 'blur']]
    for ax, (k, v) in zip(axes, cells):
        im = plt.imread(DATA / f'lpips_{k}.png')
        h, w = im.shape[:2]
        ax.imshow(im[h // 5: h * 4 // 5, w // 5: w * 4 // 5])     # central 60%, enlarged
        bare_axes(ax)
        ax.set_title('original' if v is None else labels[k], fontsize=SMALL_PT)
        if v is None:
            ax.set_xlabel('LPIPS = 0', color=INK)
        else:
            ax.set_xlabel(f'LPIPS = {num(v["lpips"])}', color=NML_RED if v['lpips'] > 0.5 else NML_CYAN)
    mse = np.mean([v['mse'] for v in l['variants'].values()])
    psnr = np.mean([v['psnr'] for v in l['variants'].values()])
    fig.supxlabel(f'all four distortions: MSE = {num(mse, 3)}, PSNR = {num(psnr, 1)} dB', fontsize=TEXT_PT, color=MUTED)
    save_figure(fig, 'LPIPS_equal_mse')
    plt.close(fig)
    print('LPIPS', {k: round(v['lpips'], 3) for k, v in l['variants'].items()}, 'MSE', round(mse, 4))


# ===========================================================================
def clip_score():
    c = load('clip')
    cos = np.array(c['cos'])                 # images x captions
    score = c['w'] * np.clip(cos, 0, None)
    tags = [t for t, _ in c['captions']]
    caps = [cp for _, cp in c['captions']]
    images = c['images']
    nI, nC = cos.shape
    fig = book_canvas(1.0, 3.05)
    top = 0.80
    axm = fig.add_axes([0.42, 0.09, 0.56, top - 0.09])
    data = score.T                           # captions x images
    im = axm.imshow(data, cmap=CYAN_MAP, vmin=0.1, vmax=0.9, aspect='auto')
    for i in range(nC):
        for j in range(nI):
            axm.text(j, i, num(data[i, j]), ha='center', va='center', fontsize=SMALL_PT,
                     color=ink_for(im.cmap(im.norm(data[i, j]))))
    for j, k in enumerate(images):
        i = tags.index(k)
        axm.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=INK, lw=1.1))
    for probe in ('hopper_swap', 'hopper_neg'):
        i = tags.index(probe)
        axm.add_patch(Rectangle((-0.47, i - 0.47), 0.94, 0.94, fill=False, edgecolor=NML_RED, lw=1.1,
                                ls=(0, (2, 1.5))))
    axm.set_xticks([])
    import textwrap
    axm.set_yticks(range(nC), labels=['\n'.join(textwrap.wrap(f'“{cp}”', 27)) for cp in caps])
    axm.tick_params(length=0)
    for side in axm.spines.values():
        side.set_visible(False)
    w = 0.56 / nI
    for j, k in enumerate(images):
        a = fig.add_axes([0.42 + j * w + 0.006, top + 0.01, w - 0.012, 0.18])
        a.imshow(plt.imread(DATA / f'img_{k}.png'))
        bare_axes(a)
    fig.text(0.0, 0.0, f'cell = CLIP-S = {c["w"]:g} × cosine; raw cosines span '
                       f'{num(cos.min())}–{num(cos.max())}', color=MUTED, va='bottom')
    save_figure(fig, 'CLIP_Score_matrix', crop=False)
    plt.close(fig)
    print('CLIP diag', [round(score[i, tags.index(k)], 3) for i, k in enumerate(images)],
          'swap', round(score[0, tags.index('hopper_swap')], 3), 'neg', round(score[0, tags.index('hopper_neg')], 3),
          'min', round(cos.min(), 3), 'max', round(cos.max(), 3))


# ===========================================================================
def dsg():
    Q = {'q1': ('Is there a\nbicycle?', []), 'q2': ('Is the bicycle\nred?', ['q1']), 'q3': ('Is there\na wall?', []),
         'q4': ('Is the wall\nblue?', ['q3']), 'q5': ('Is it leaning\non the wall?', ['q1', 'q3'])}
    pos = {'q1': (0.2, 0.74), 'q3': (0.8, 0.74), 'q2': (0.17, 0.34), 'q4': (0.83, 0.34), 'q5': (0.5, 0.34)}
    images = [('image A: wrong colors', {'q1': 'yes', 'q2': 'no', 'q3': 'yes', 'q4': 'no', 'q5': 'yes'}),
              ('image B: no bicycle', {'q1': 'no', 'q2': 'yes', 'q3': 'yes', 'q4': 'yes', 'q5': 'yes'})]
    fills = {'yes': (CYAN_TINT, NML_CYAN), 'no': (RED_TINT, NML_RED), 'skip': ('#f2f2f2', REF)}
    fig, axes = book_figure(1.0, 2.3, 1, 2)
    fig.suptitle('prompt: “a red bicycle leaning against a blue wall”\n', fontsize=TEXT_PT, color=MUTED, linespacing=1.2)
    out = {}
    bw, bh = 0.31, 0.24
    for ax, (title, ans) in zip(axes, images):
        skipped = [q for q, (_, par) in Q.items() if any(ans[p] != 'yes' for p in par)]
        got = sum(ans[q] == 'yes' and q not in skipped for q in Q)
        out[title] = got
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        bare_axes(ax)
        for q, (_, par) in Q.items():
            for k, p in enumerate(par):
                (x0, y0), (x1, y1) = pos[p], pos[q]
                dx = 0 if len(par) == 1 else (-0.04 if k == 0 else 0.04)
                ax.add_patch(FancyArrowPatch((x0, y0 - bh / 2), (x1 + dx, y1 + bh / 2), arrowstyle='-|>',
                                             mutation_scale=6, color=REF if q in skipped else INK, lw=0.7,
                                             shrinkA=0, shrinkB=0, zorder=2))
        for q, (text, _) in Q.items():
            x, y = pos[q]
            state = 'skip' if q in skipped else ans[q]
            face, edge = fills[state]
            ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle='round,pad=0,rounding_size=0.02',
                                        facecolor=face, edgecolor=edge, lw=0.8, zorder=3,
                                        linestyle=(0, (2, 1.5)) if state == 'skip' else '-'))
            ax.text(x, y + 0.035, text, ha='center', va='center', color=INK, zorder=4, linespacing=1.1,
                    fontsize=SMALL_PT)
            ax.text(x, y - 0.08, 'skipped' if state == 'skip' else ans[q], ha='center', va='center',
                    color=edge, zorder=4, fontsize=SMALL_PT)
        ax.set_title(title, fontsize=TEXT_PT)
        ax.text(0.5, 0.05, f'DSG = {got}/5 = {num(got / 5)}', ha='center', va='center', color=INK,
                fontweight='semibold')
    save_figure(fig, 'DSG_question_graph')
    plt.close(fig)
    print('DSG', out)


# ===========================================================================
def vqascore():
    import textwrap
    c, v = load('clip'), load('vqascore')
    tags = [t for t, _ in c['captions']]
    caps = [cp for _, cp in c['captions']]
    i = c['images'].index('hopper')
    clip_s = c['w'] * np.clip(np.array(c['cos'][i]), 0, None)
    p_yes = np.array(v['p_yes'][v['images'].index('hopper')])
    keep = [tags.index(t) for t in ('hopper', 'hopper_swap', 'hopper_neg', 'dahlia')]
    order = [j for j in np.argsort(-p_yes) if j in keep]
    fig = book_canvas(1.0, 2.2)
    axI = fig.add_axes([0.0, 0.12, 0.2, 0.74])
    axI.imshow(plt.imread(DATA / 'img_hopper.png'))
    bare_axes(axI)
    axA = fig.add_axes([0.6, 0.18, 0.17, 0.66])
    axB = fig.add_axes([0.82, 0.18, 0.17, 0.66])
    ys = np.arange(len(order))[::-1]
    for ax, vals, col, title in [(axA, p_yes, NML_CYAN, 'BLIP yes/no\nscore'), (axB, clip_s, MUTED, 'CLIP-S')]:
        for y, j in zip(ys, order):
            ax.scatter(vals[j], y, s=14, color=col, zorder=3, linewidths=0)
            ax.text(vals[j], y + 0.22, num(vals[j]), ha='center', va='bottom', color=INK, fontsize=SMALL_PT)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(order) - 0.3)
        ax.set_xticks([0, 0.5, 1.0], labels=['0.0', '0.5', '1.0'])
        ax.set_title(title, fontsize=TEXT_PT, linespacing=1.2)
        tidy_axes(ax, left=False)
    axA.set_yticks(ys, labels=['\n'.join(textwrap.wrap(f'\u201c{caps[j]}\u201d', 26)) for j in order])
    axA.tick_params(axis='y', length=0, labelleft=True, pad=6)
    for lab, j in zip(axA.get_yticklabels(), order):
        lab.set_color(NML_RED if tags[j] in ('hopper_swap', 'hopper_neg') else INK)
    save_figure(fig, 'VQAScore_vs_CLIP', crop=False)
    plt.close(fig)
    print('VQAScore', [(tags[j], round(p_yes[j], 3), round(clip_s[j], 3)) for j in order])


# ===========================================================================
def mos():
    n = 20
    systems = {'consensus': [3] * n, 'polarized': [1] * 10 + [5] * 10,
               'skewed': [2] * 8 + [3] * 6 + [4] * 4 + [5] * 2}
    tcrit = float(student_t.ppf(0.975, n - 1))
    fig, axes = book_figure(1.0, 2.35, 1, 3, sharey=True)
    out = {}
    for ax, (name, r) in zip(axes, systems.items()):
        r = np.array(r)
        m, med, sd = r.mean(), np.median(r), r.std(ddof=1)
        half = tcrit * sd / np.sqrt(n)
        out[name] = (m, med, half)
        for rating in range(1, 6):
            k = int(np.sum(r == rating))
            ax.scatter([rating] * k, np.arange(k) + 1, s=7, color=NML_CYAN, linewidths=0, zorder=3)
        ax.axvline(m, color=REF, lw=LW_THIN, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0, 21)
        ax.set_title(f'{name}\nMOS = {num(m, 1)} ± {num(half)}\nmedian = {med:.0f}', loc='left',
                     fontsize=TEXT_PT, linespacing=1.3)
        ax.set_xlabel('listener rating')
        tidy_axes(ax, left=False)
    axes[2].text(0.6, 20.5, f'{n} listeners,\none dot each', color=MUTED, va='top', ha='left')
    save_figure(fig, 'MOS_same_mean')
    plt.close(fig)
    print('MOS', {k: tuple(round(x, 3) for x in v) for k, v in out.items()}, 't', round(tcrit, 3))


# ===========================================================================
def pesq():
    a = load('audio')
    order = ['clean', 'delay', 'gain', 'telephone', 'noise40', 'noise30', 'noise20', 'noise10']
    labels = {'clean': 'clean reference', 'delay': 'delayed by 20 ms', 'gain': 'gain −6 dB',
              'telephone': 'band-limited 300–3400 Hz', 'noise40': 'white noise, 40 dB SNR',
              'noise30': 'white noise, 30 dB SNR', 'noise20': 'white noise, 20 dB SNR',
              'noise10': 'white noise, 10 dB SNR'}
    misleads = {'delay', 'gain', 'telephone'}
    pmax = max(a[k]['pesq_nb'] for k in order)
    fig = book_canvas(1.0, 2.05)
    axL = fig.add_axes([0.42, 0.21, 0.27, 0.72])
    axR = fig.add_axes([0.73, 0.21, 0.25, 0.72])
    ys = np.arange(len(order))[::-1]
    for y, k in zip(ys, order):
        d = a[k]
        col = NML_RED if k in misleads else NML_CYAN
        if d['snr_db'] is None:
            axL.text(44, y, '∞', ha='center', va='center', color=col)
        else:
            axL.scatter(d['snr_db'], y, s=12, color=col, zorder=4, linewidths=0)
            axL.annotate(num(d['snr_db'], 1), (d['snr_db'], y), xytext=(4, 0), textcoords='offset points',
                         va='center', color=col, fontsize=SMALL_PT)
        axR.scatter(d['pesq_nb'], y, s=12, color=col, zorder=4, linewidths=0)
        axR.annotate(num(d['pesq_nb']), (d['pesq_nb'], y), xytext=(-4, 0), textcoords='offset points',
                     va='center', ha='right', color=col, fontsize=SMALL_PT)
    axL.set_xlim(-5, 46)
    axL.set_xticks([0, 20, 40])
    axL.axvline(0, color=REF, lw=LW_THIN, zorder=1)
    axL.set_xlabel('waveform SNR (dB)')
    axR.set_xlim(1, 4.7)
    axR.set_xticks([1, 2, 3, 4])
    axR.axvline(pmax, color=REF, lw=LW_THIN, zorder=1, ls=(0, (3, 2)))
    axR.text(pmax, len(order) - 0.35, f'maximum {num(pmax)}', ha='right', va='bottom', color=MUTED,
             fontsize=SMALL_PT)
    axR.set_xlabel('PESQ (MOS-LQO)')
    for ax in (axL, axR):
        ax.set_yticks([])
        ax.set_ylim(-0.6, len(order) - 0.1)
        tidy_axes(ax, left=False)
    axL.set_yticks(ys, labels=[labels[k] for k in order])
    axL.tick_params(axis='y', length=0, labelleft=True, pad=4)
    save_figure(fig, 'PESQ_alignment', crop=False)
    plt.close(fig)
    print('PESQ', {k: (a[k]['snr_db'] and round(a[k]['snr_db'], 2), round(a[k]['pesq_nb'], 3)) for k in order})


# ===========================================================================
def third_octave_bands(fs, nfft, num_bands=15, min_freq=150):
    f = np.linspace(0, fs, nfft + 1)[: nfft // 2 + 1]
    cf = 2 ** (np.arange(num_bands) / 3) * min_freq
    H = np.zeros((num_bands, len(f)))
    for i in range(num_bands):
        H[i, (f >= cf[i] * 2 ** (-1 / 6)) & (f < cf[i] * 2 ** (1 / 6))] = 1
    return H, cf


def stoi_blocks(x, y, fs_in):
    """Simplified STOI walkthrough after Taal et al. (2011): 10 kHz, 256-sample
    frames, silent-frame removal, 15 one-third-octave bands, non-overlapping
    30-frame segments, per-segment normalization and clipping at -15 dB."""
    fs = 10000
    x = signal.resample_poly(x, fs, fs_in)
    y = signal.resample_poly(y, fs, fs_in)
    N, hop, nseg, beta = 256, 128, 30, -15
    win = np.hanning(N + 2)[1:-1]
    frames = np.arange(0, len(x) - N, hop)
    ex = np.array([np.sum((x[i:i + N] * win) ** 2) for i in frames])
    frames = frames[10 * np.log10(ex + 1e-12) > 10 * np.log10(ex.max()) - 40]
    X = np.array([np.abs(np.fft.rfft(x[i:i + N] * win, N)) ** 2 for i in frames]).T
    Y = np.array([np.abs(np.fft.rfft(y[i:i + N] * win, N)) ** 2 for i in frames]).T
    H, cf = third_octave_bands(fs, N)
    Xb, Yb = np.sqrt(H @ X), np.sqrt(H @ Y)
    c = 10 ** (-beta / 20)
    nblocks = Xb.shape[1] // nseg
    d = np.zeros((Xb.shape[0], nblocks))
    Yn = np.zeros_like(Yb)
    for m in range(nblocks):
        sl = slice(m * nseg, (m + 1) * nseg)
        xs, ys = Xb[:, sl], Yb[:, sl]
        alpha = np.linalg.norm(xs, axis=1, keepdims=True) / (np.linalg.norm(ys, axis=1, keepdims=True) + 1e-12)
        yn = np.minimum(ys * alpha, xs * (1 + c))
        Yn[:, sl] = yn
        xc, yc = xs - xs.mean(1, keepdims=True), yn - yn.mean(1, keepdims=True)
        d[:, m] = np.sum(xc * yc, 1) / (np.linalg.norm(xc, axis=1) * np.linalg.norm(yc, axis=1) + 1e-12)
    return d, Xb, Yn, cf, hop / fs


def stoi():
    a = load('audio')
    clean = np.load(DATA / 'audio_clean.npy')
    noisy = np.load(DATA / 'audio_noise0.npy')
    d, Xb, Yn, cf, frame_s = stoi_blocks(clean, noisy, 16000)
    band = int(np.argmin(np.abs(cf - 1000)))
    s0, n_show, L = 2, 3, 30
    fig, (axT, axB) = book_figure(1.0, 2.33, 2, 1, gridspec_kw={'height_ratios': [1, 1.25]})
    sl = slice(s0 * L, (s0 + n_show) * L)
    tt = np.arange(sl.start, sl.stop) * frame_s
    axT.plot(tt, Xb[band, sl], color=NML_CYAN, lw=LW - 0.2)
    axT.plot(tt, Yn[band, sl], color=NML_RED, lw=LW - 0.3)
    top = max(Xb[band, sl].max(), Yn[band, sl].max())
    for m in range(n_show):
        t0, t1 = (s0 + m) * L * frame_s, (s0 + m + 1) * L * frame_s
        if m:
            axT.axvline(t0, color=REF, lw=LW_THIN, zorder=1)
        axT.text((t0 + t1) / 2, top * 1.12, f'correlation = {num(d[band, s0 + m])}', ha='center', va='bottom',
                 color=INK)
    axT.text(tt[-1], top * 1.0, 'clean', color=NML_CYAN, va='top', ha='right')
    axT.text(tt[-1], top * 0.84, 'noisy, 0 dB SNR\n(normalized, clipped)', color=NML_RED, va='top', ha='right',
             linespacing=1.2)
    axT.set_xlim(tt[0], tt[-1])
    axT.set_ylim(0, top * 1.3)
    axT.set_yticks([])
    axT.set_xlabel('analysis time after silence removal (s); three 384 ms segments')
    axT.set_ylabel(f'{cf[band]:.0f} Hz\nenvelope', linespacing=1.2)
    tidy_axes(axT, left=True)
    norm = mcolors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=1)
    cmap = mcolors.LinearSegmentedColormap.from_list('stoi', [NML_RED, '#ffffff', NML_CYAN])
    im = axB.imshow(d, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    axB.set_yticks([0, 7, 14], labels=[f'{cf[i]:.0f} Hz' for i in [0, 7, 14]])
    axB.set_xticks(np.arange(0, d.shape[1], 4))
    axB.set_xlabel('segment (384 ms each)')
    axB.set_ylabel('band')
    for side in axB.spines.values():
        side.set_visible(False)
    axB.tick_params(length=0)
    axB.add_patch(Rectangle((s0 - 0.5, band - 0.5), n_show, 1, fill=False, edgecolor=INK, lw=1.0))
    axB.set_title(f'mean of all cells = {num(d.mean())} (reference implementation: {num(a["noise0"]["stoi"])})',
                  loc='left', fontsize=TEXT_PT)
    cb = fig.colorbar(im, ax=axB, fraction=0.05, pad=0.02, aspect=12)
    cb.set_ticks([-0.2, 0, 0.5, 1.0], labels=['−0.2', '0.0', '0.5', '1.0'])
    cb.set_label('envelope correlation')
    cb.outline.set_linewidth(0.6)
    save_figure(fig, 'STOI_envelopes')
    plt.close(fig)
    neg = d[d < 0]
    print(f'STOI blocks mean {d.mean():.4f} lib {a["noise0"]["stoi"]:.4f}; band {cf[band]:.0f} '
          f'corr {[round(d[band, s0 + m], 3) for m in range(n_show)]}; negatives {np.round(neg, 3).tolist()}')


FIGURES = {'Perplexity_surprisal': perplexity, 'BERTScore_matching': bertscore, 'MAUVE_frontier': mauve,
           'IS_mechanism': inception, 'FID_mechanism': fid, 'LPIPS_equal_mse': lpips,
           'CLIP_Score_matrix': clip_score, 'DSG_question_graph': dsg, 'VQAScore_vs_CLIP': vqascore,
           'MOS_same_mean': mos, 'PESQ_alignment': pesq, 'STOI_envelopes': stoi}

if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
