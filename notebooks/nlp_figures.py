"""Figures for the NLP chapter, drawn at printed size.

Token diagrams share one layout: axes measured in points, word boxes sized from
the rendered text, row labels right-aligned in one column, and one color code:
cyan = exact match, purple = synonym or moved phrase, red = not matched or an
edit, light gray = reference words. Every score is computed from the sentences
drawn. Run: uv run python notebooks/nlp_figures.py [NAME ...]
"""
import itertools
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch, Patch
from style import *  # noqa: F401,F403

MATCH, SYN, MISS, REFC = NML_CYAN, NML_PURPLE, NML_RED, '#e4e4e4'


# ---------------------------------------------------------------------------
# metric primitives
# ---------------------------------------------------------------------------
def toks(s):
    return s.lower().split()


def ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def mod_precision(c, r, n):
    cc, rr = ngrams(c, n), ngrams(r, n)
    total = sum(cc.values())
    return sum(min(k, rr.get(g, 0)) for g, k in cc.items()) / total if total else 0.0


def bleu(cand, ref, N=4):
    c, r = toks(cand), toks(ref)
    ps = [mod_precision(c, r, n) for n in range(1, N + 1)]
    bp = 1.0 if len(c) > len(r) else float(np.exp(1 - len(r) / len(c)))
    return ps, (0.0 if min(ps) == 0 else bp * float(np.exp(np.mean(np.log(ps)))))


def meteor(cand, ref, equiv, alpha=0.9, gamma=0.5, beta=3.0):
    """Original METEOR scoring on a one-to-one alignment: exact matches first,
    then synonym/stem pairs; among alignments of equal size, the one with the
    fewest chunks is used (as METEOR does)."""
    c, r = toks(cand), toks(ref)
    options = []
    for ci, w in enumerate(c):
        exact = [ri for ri, rw in enumerate(r) if rw == w]
        options.append([(ri, 'exact') for ri in exact] or
                       [(ri, 'equiv') for ri, rw in enumerate(r) if equiv.get(w) == rw] or [None])
    best = None
    for combo in itertools.product(*options):
        used = [o[0] for o in combo if o]
        if len(used) != len(set(used)):
            continue
        pairs = [(ci, o[0], o[1]) for ci, o in enumerate(combo) if o]
        chunks = sum(1 for k, (ci, ri, _) in enumerate(pairs)
                     if k == 0 or not (ci == pairs[k - 1][0] + 1 and ri == pairs[k - 1][1] + 1))
        key = (-len(pairs), chunks)
        if best is None or key < best[0]:
            best = (key, pairs, chunks)
    _, pairs, chunks = best
    m = len(pairs)
    P, R = m / len(c), m / len(r)
    fmean = P * R / (alpha * P + (1 - alpha) * R)
    return fmean * (1 - gamma * (chunks / m) ** beta), pairs, chunks


def rouge1(cand, ref):
    c, r = ngrams(toks(cand), 1), ngrams(toks(ref), 1)
    overlap = sum(min(k, r.get(g, 0)) for g, k in c.items())
    return overlap / sum(c.values()), overlap / sum(r.values())


def edit_align(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), int)
    dp[:, 0], dp[0, :] = np.arange(m + 1), np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = m, n, []
    while i > 0 or j > 0:
        cost = 0 if (i and j and ref[i - 1] == hyp[j - 1]) else 1
        if i and j and dp[i, j] == dp[i - 1, j - 1] + cost:
            ops.append(('C' if cost == 0 else 'S', ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i and dp[i, j] == dp[i - 1, j] + 1:
            ops.append(('D', ref[i - 1], None)); i -= 1
        else:
            ops.append(('I', None, hyp[j - 1])); j -= 1
    return ops[::-1]


def squad_norm(s):
    s = re.sub(r'[^a-z0-9 ]', ' ', s.lower())
    return [w for w in s.split() if w not in {'a', 'an', 'the'}]


def token_f1(pred, gold):
    p, g = squad_norm(pred), squad_norm(gold)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    pr, rc = overlap / len(p), overlap / len(g)
    return 2 * pr * rc / (pr + rc)


# ---------------------------------------------------------------------------
# token layout in points
# ---------------------------------------------------------------------------
PAD = 2.0        # box padding around the text, points
GAP = 2.6        # space between boxes, points
LABEL_X = 52     # right edge of the row-label column, points


class Tokens:
    def __init__(self, width, height):
        self.fig = book_canvas(width, height)
        self.W, self.H = TEXTWIDTH_IN * width * 72, height * 72
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        bare_axes(self.ax)
        self.r = self.fig.canvas.get_renderer()

    def width_of(self, s, **kw):
        t = self.ax.text(0, 0, s, fontsize=kw.get('fontsize', TEXT_PT))
        w = t.get_window_extent(self.r).width * 72 / self.fig.dpi
        t.remove()
        return w

    def box(self, x, y, word, face, ink='white', w=None):
        tw = self.width_of(word)
        w = w or tw + 2 * PAD
        h = TEXT_PT + 2 * PAD
        self.ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=1.6',
                                         facecolor=face, edgecolor='none', zorder=2))
        self.ax.text(x + w / 2, y, word, ha='center', va='center_baseline', color=ink, zorder=3)
        return w

    def slot(self, x, y, w):
        h = TEXT_PT + 2 * PAD
        self.ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=1.6',
                                         facecolor='none', edgecolor=REF, lw=0.6, ls=(0, (2, 1.5)), zorder=2))

    def row(self, words, y, faces, inks=None, x0=LABEL_X + 6, label=None):
        inks = inks or ['white'] * len(words)
        xs, x = [], x0
        for w, f, k in zip(words, faces, inks):
            bw = self.box(x, y, w, f, k)
            xs.append((x, x + bw))
            x += bw + GAP
        if label:
            self.label(y, label)
        return xs

    def label(self, y, text):
        self.ax.text(LABEL_X, y, text, ha='right', va='center_baseline', color=MUTED)

    def text(self, x, y, s, **kw):
        kw.setdefault('va', 'center_baseline')
        return self.ax.text(x, y, s, **kw)

    def save(self, name):
        save_figure(self.fig, name, crop=False)
        plt.close(self.fig)


def token_legend(fig, items, ncol=3):
    fig.legend(handles=[Patch(color=c, label=l) for c, l in items], loc='lower left',
               bbox_to_anchor=(LABEL_X / (fig.get_figwidth() * 72) + 0.005, 0.0), ncol=ncol,
               handlelength=1.0, columnspacing=1.0, borderaxespad=0.1)


# ===========================================================================
SENT = 'the quick brown fox jumps over the lazy dog'


def bleu_figure():
    cand = 'the quick brown cat jumps over the tired dog'
    ps, score = bleu(cand, SENT)
    amean = float(np.mean(ps))
    fig = book_canvas(1.0, 2.55)
    W = TEXTWIDTH_IN
    top = fig.add_axes([0, 0.73, 1, 0.27])
    top.set_xlim(0, W * 72)
    top.set_ylim(0, 0.27 * 2.55 * 72)
    bare_axes(top)
    T = Tokens.__new__(Tokens)
    T.fig, T.ax, T.r = fig, top, fig.canvas.get_renderer()
    rw, cw = toks(SENT), toks(cand)
    T.row(rw, 37, [REFC] * 9, [INK] * 9, label='reference')
    T.row(cw, 15, [MISS if a != b else MATCH for a, b in zip(rw, cw)], label='output')
    ax = fig.add_axes([0.135, 0.15, 0.5, 0.52])
    xs = np.arange(1, 5)
    ax.bar(xs, ps, width=0.6, color=NML_CYAN, linewidth=0)
    for x, p in zip(xs, ps):
        if p > 0:
            ax.text(x, p + 0.02, num(p), ha='center', va='bottom', color=INK)
    ax.scatter([4], [0], s=22, facecolors='white', edgecolors=NML_RED, linewidths=1.0, zorder=5,
               clip_on=False)
    ax.plot([0.6, 4.4], [amean, amean], color=REF, lw=LW_THIN, ls=(0, (3, 2)))
    ax.text(4.55, amean, f'arithmetic mean = {num(amean)}\n(not used by BLEU)', color=MUTED,
            va='center', linespacing=1.25)
    ax.annotate('BLEU = geometric\nmean = 0.00', (4.0, 0.0), xytext=(4.55, 0.14), color=NML_RED,
                va='center', linespacing=1.25, annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color=NML_RED, lw=LW_THIN, shrinkA=1, shrinkB=3))
    ax.set_xticks(xs, labels=['1', '2', '3', '4'])
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0, 1.0)
    unit_ticks(ax, 'y', 0.5)
    ax.set_xlabel('n-gram length $n$')
    ax.set_ylabel('modified n-gram\nprecision $p_n$')
    tidy_axes(ax)
    save_figure(fig, 'BLEU_ngram_precision', crop=False)
    plt.close(fig)
    print('BLEU', [round(p, 3) for p in ps], 'AM', round(amean, 3), 'BLEU', score)


def meteor_figure():
    equiv = {'fast': 'quick', 'leaps': 'jumps', 'idle': 'lazy'}
    cands = [('exact match', SENT),
             ('synonyms, same order', 'the fast brown fox leaps over the idle dog'),
             ('same words, reordered', 'over the lazy dog jumps the quick brown fox')]
    T = Tokens(1.0, 2.1)
    ys = [124, 94, 64, 34]
    names = ['reference'] + [n for n, _ in cands]
    x0 = 2
    for y, name in zip(ys, names):
        T.text(x0, y + 11, name, color=MUTED)
    T.row(toks(SENT), ys[0], [REFC] * 9, [INK] * 9, x0=x0)
    bx, mx = T.W - 46, T.W - 2
    T.text(bx, ys[0] + 11, 'BLEU', color=MUTED, ha='right')
    T.text(mx, ys[0] + 11, 'METEOR', color=MUTED, ha='right')
    for (name, cand), y in zip(cands, ys[1:]):
        m, pairs, chunks = meteor(cand, SENT, equiv)
        kind = {ci: k for ci, _, k in pairs}
        c = toks(cand)
        T.row(c, y, [MATCH if kind.get(i) == 'exact' else SYN if kind.get(i) == 'equiv' else MISS
                     for i in range(len(c))], x0=x0)
        b = bleu(cand, SENT)[1]
        T.text(bx, y, num(b), ha='right', color=INK)
        T.text(mx, y, num(m), ha='right', color=INK)
        T.text(mx, y + 11, f'{chunks} chunk{"s" if chunks > 1 else ""}', ha='right', color=MUTED,
               fontsize=SMALL_PT)
        print(f'METEOR {name}: BLEU {b:.4f} METEOR {m:.4f} chunks {chunks} matches {len(pairs)}')
    T.fig.legend(handles=[Patch(color=MATCH, label='exact match'), Patch(color=SYN, label='synonym match')],
                 loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=2, handlelength=1.0,
                 borderaxespad=0.1)
    T.save('METEOR_vs_BLEU')


def rouge_figure():
    ref = 'the central bank raised interest rates to curb inflation'
    core = 'the bank raised rates to curb inflation'
    filler = ('in a widely expected move that analysts said had been telegraphed for weeks during '
              'numerous public speeches').split()
    lens, P, R = [], [], []
    for k in range(len(filler) + 1):
        p, r = rouge1(core + ' ' + ' '.join(filler[:k]), ref)
        lens.append(len(toks(core)) + k); P.append(p); R.append(r)
    lens, P, R = map(np.array, (lens, P, R))
    fig, ax = book_figure(0.82, 2.35)
    ax.plot(lens, R, color=NML_CYAN)
    ax.plot(lens, P, color=NML_RED)
    label_end(ax, lens[-1], R[-1], 'ROUGE-1\nrecall', NML_CYAN, linespacing=1.2)
    label_end(ax, lens[-1], P[-1], 'ROUGE-1\nprecision', NML_RED, linespacing=1.2)
    for i in (0, -1):
        for v, col in [(R[i], NML_CYAN), (P[i], NML_RED)]:
            ax.scatter([lens[i]], [v], s=16, facecolors='white', edgecolors=col, linewidths=0.9, zorder=5)
    ax.text(lens[0] + 0.6, 1.2, f'concise summary ({lens[0]} words):\nrecall {num(R[0])}, precision {num(P[0])}',
            va='top', color=INK, linespacing=1.25)
    ax.text(lens[-1] - 0.3, 0.6, f'padded ({lens[-1]} words):\nrecall {num(R[-1])},\nprecision {num(P[-1])}',
            ha='right', va='center', color=INK, linespacing=1.25)
    ax.set_xlim(lens[0] - 0.5, lens[-1] + 0.5)
    ax.set_ylim(0, 1.25)
    ax.set_xticks([7, 12, 17, 22, 24])
    unit_ticks(ax, 'y', 0.5)
    ax.spines['left'].set_bounds(0, 1)
    ax.set_xlabel('summary length (words)')
    ax.set_ylabel('ROUGE-1 score')
    tidy_axes(ax)
    save_figure(fig, 'ROUGE_variants')
    plt.close(fig)
    print(f'ROUGE recall {R[0]:.3f} precision {P[0]:.3f}->{P[-1]:.3f}')


def ter_figure():
    ref, hyp = toks('police arrested the suspect on monday morning'), toks('on monday morning police arrested the suspect')
    phrase = {'on', 'monday', 'morning'}
    n = len(ref)
    T = Tokens(1.0, 1.5)
    yr, yh = 84, 46
    rx = T.row(ref, yr, [SYN if w in phrase else MATCH for w in ref], label='reference')
    hx = T.row(hyp, yh, [SYN if w in phrase else MATCH for w in hyp], label='output')
    src = (hx[0][0] + hx[2][1]) / 2
    dst = (rx[4][0] + rx[6][1]) / 2
    T.ax.annotate('', xy=(dst, yr - 8), xytext=(src, yh + 8),
                  arrowprops=dict(arrowstyle='-|>', color=REF, lw=LW_THIN + 0.2, mutation_scale=7,
                                  connectionstyle='arc3,rad=-0.08'))
    T.text((src + dst) / 2, (yr + yh) / 2, 'one shift', ha='center', color=MUTED, zorder=4,
           bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    T.text(2, 24, f'without shifts: 3 deletions + 3 insertions = 6 edits; edit rate = 6/{n} = {num(6 / n)}',
           color=MUTED)
    T.text(2, 9, f'with one shift: 1 edit; TER = 1/{n} = {num(1 / n)}', color=SYN)
    T.save('TER_edit_breakdown')


def em_figure():
    gold = 'Martin Luther King'
    cands = ['Martin Luther King', 'Martin Luther King Jr.', 'Dr. King', 'Malcolm X']
    g = set(squad_norm(gold))
    T = Tokens(1.0, 1.75)
    ys = [104, 82, 64, 46, 28]
    T.row(gold.split(), ys[0], [REFC] * 3, [INK] * 3, label='gold answer')
    em_x, f1_x = T.W - 66, T.W - 2
    T.text(em_x, ys[0], 'EM', color=MUTED, ha='right')
    T.text(f1_x, ys[0], 'token F1', color=MUTED, ha='right')
    for c, y in zip(cands, ys[1:]):
        words = c.split()
        T.row(words, y, [MATCH if (squad_norm(w) and squad_norm(w)[0] in g) else MISS for w in words])
        em = int(squad_norm(c) == squad_norm(gold))
        f1 = token_f1(c, gold)
        T.text(em_x, y, str(em), ha='right', color=INK)
        T.text(f1_x, y, num(f1), ha='right', color=INK)
        print(f'EM {c}: EM {em} F1 {f1:.4f}')
    T.label(sum(ys[1:]) / 4, 'predictions')
    token_legend(T.fig, [(MATCH, 'token in the gold answer'), (MISS, 'token not in the gold answer')], ncol=2)
    T.save('EM_vs_F1')


def wer_figure():
    ref, hyp = toks(SENT), toks('the very quick brown box jumps over the lazy')
    ops = edit_align(ref, hyp)
    S = sum(o == 'S' for o, _, _ in ops)
    D = sum(o == 'D' for o, _, _ in ops)
    I = sum(o == 'I' for o, _, _ in ops)
    N = len(ref)
    T = Tokens(1.0, 1.45)
    yr, yh = 82, 42
    x = LABEL_X + 6
    for op, a, b in ops:
        w = max(T.width_of(a or ''), T.width_of(b or '')) + 2 * PAD
        face = MATCH if op == 'C' else MISS
        T.box(x, yr, a, face, w=w) if a else T.slot(x, yr, w)
        T.box(x, yh, b, face, w=w) if b else T.slot(x, yh, w)
        if op != 'C':
            T.text(x + w / 2, (yr + yh) / 2, {'S': 'sub', 'D': 'del', 'I': 'ins'}[op], ha='center',
                   color=NML_RED)
        x += w + GAP
    T.label(yr, 'reference')
    T.label(yh, 'output')
    T.text(LABEL_X + 6, 12, f'WER = (S + D + I) / N = ({S} + {D} + {I}) / {N} = {num((S + D + I) / N)}',
           color=INK)
    T.save('WER_vs_errors')
    print(f'WER S {S} D {D} I {I} N {N}')


FIGURES = {'BLEU_ngram_precision': bleu_figure, 'METEOR_vs_BLEU': meteor_figure,
           'ROUGE_variants': rouge_figure, 'TER_edit_breakdown': ter_figure,
           'EM_vs_F1': em_figure, 'WER_vs_errors': wer_figure}

if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
