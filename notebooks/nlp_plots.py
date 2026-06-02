"""Generate figures for the NLP chapter.

Every metric is hand-computed from the sentence pairs the figure draws — no
hardcoded scores. Each figure shows the metric's *mechanism* (geometric-mean
collapse, synonym matching, recall-vs-length, phrase shifts, token overlap,
edit alignment), not just an output number.

Run from the notebooks/ directory:  uv run python nlp_plots.py
"""
import sys
sys.path.insert(0, '.')

import re
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from style import *

GRAY = '#9a9a9a'
LIGHT = '#d9d9d9'
INK = '#33333a'


# ---------------------------------------------------------------------------
# metric primitives (all computed, nothing hardcoded)
# ---------------------------------------------------------------------------
def toks(s):
    return s.lower().split()


def ngram_counts(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def modified_precision(cand, ref, n):
    c, r = ngram_counts(cand, n), ngram_counts(ref, n)
    total = sum(c.values())
    if total == 0:
        return 0, 0, 0.0
    clipped = sum(min(cnt, r.get(g, 0)) for g, cnt in c.items())
    return clipped, total, clipped / total


def brevity_penalty(c_len, r_len):
    if c_len > r_len:
        return 1.0
    if c_len == 0:
        return 0.0
    return float(np.exp(1 - r_len / c_len))


def bleu(cand_s, ref_s, N=4):
    c, r = toks(cand_s), toks(ref_s)
    ps = [modified_precision(c, r, n)[2] for n in range(1, N + 1)]
    bp = brevity_penalty(len(c), len(r))
    score = 0.0 if min(ps) <= 0 else bp * float(np.exp(np.mean(np.log(ps))))
    return bp, ps, score


def lcs_len(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_n(cand_s, ref_s, n):
    c, r = ngram_counts(toks(cand_s), n), ngram_counts(toks(ref_s), n)
    overlap = sum(min(cnt, r.get(g, 0)) for g, cnt in c.items())
    rec = overlap / sum(r.values()) if sum(r.values()) else 0.0
    prec = overlap / sum(c.values()) if sum(c.values()) else 0.0
    return prec, rec


def edit_align(ref, hyp):
    """Word-level Levenshtein with backtrace. Returns list of (op, ref_w, hyp_w),
    op in {C,S,D,I}. Diagonal preferred, so matches/subs stay aligned."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j, ops = m, n, []
    while i > 0 or j > 0:
        cost = 0 if (i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]) else 1
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + cost:
            ops.append(('C' if cost == 0 else 'S', ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(('D', ref[i - 1], None))
            i -= 1
        else:
            ops.append(('I', None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


ARTICLES = {'a', 'an', 'the'}


def squad_norm(s):
    s = re.sub(r'[^a-z0-9 ]', ' ', s.lower())
    return [w for w in s.split() if w not in ARTICLES]


def exact_match(pred, gold):
    return int(squad_norm(pred) == squad_norm(gold))


def token_f1(pred, gold):
    p, g = squad_norm(pred), squad_norm(gold)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def meteor_align(cand, ref, equiv):
    """One-to-one alignment: exact matches first, then synonym/stem (equiv) maps.
    Returns matches as (cand_idx, ref_idx, kind) with kind in {exact, equiv}."""
    used = [False] * len(ref)
    matched_c = {}
    for ci, w in enumerate(cand):
        for ri, rw in enumerate(ref):
            if not used[ri] and w == rw:
                used[ri] = True
                matched_c[ci] = (ri, 'exact')
                break
    for ci, w in enumerate(cand):
        if ci in matched_c or w not in equiv:
            continue
        for ri, rw in enumerate(ref):
            if not used[ri] and rw == equiv[w]:
                used[ri] = True
                matched_c[ci] = (ri, 'equiv')
                break
    return [(ci, ri, kind) for ci, (ri, kind) in sorted(matched_c.items())]


def meteor(cand_s, ref_s, equiv, alpha=0.9, gamma=0.5, beta=3.0):
    c, r = toks(cand_s), toks(ref_s)
    matches = meteor_align(c, r, equiv)
    m = len(matches)
    if m == 0:
        return 0.0, []
    P, R = m / len(c), m / len(r)
    fmean = P * R / (alpha * P + (1 - alpha) * R)
    chunks, prev = 0, None
    for ci, ri, _ in matches:
        if not (prev and ci == prev[0] + 1 and ri == prev[1] + 1):
            chunks += 1
        prev = (ci, ri)
    penalty = gamma * (chunks / m) ** beta
    return fmean * (1 - penalty), matches


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def despine(ax, keep=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)


def word_row(ax, words, y, facecolors, textcolors=None, x0=0.0, fs=13.5,
             gap=0.7, pad=1.6):
    """Lay out words as rounded boxes left-to-right in character units.
    Returns the x-centers and the right edge."""
    x, centers = x0, []
    for i, w in enumerate(words):
        width = len(w) + pad
        cx = x + width / 2
        tc = 'white' if textcolors is None else textcolors[i]
        ax.text(cx, y, w, ha='center', va='center', fontsize=fs, color=tc,
                zorder=4, fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.32', facecolor=facecolors[i],
                          edgecolor='none'))
        centers.append(cx)
        x += width + gap
    return centers, x


# ===========================================================================
# 1. BLEU — the geometric mean collapses when one n-gram precision is zero
# ===========================================================================
REF = "the quick brown fox jumps over the lazy dog"
CAND_BLEU = "the quick brown cat jumps over the tired dog"   # fox->cat, lazy->tired
bp, ps, score = bleu(CAND_BLEU, REF)
amean = float(np.mean(ps))

fig = plt.figure(figsize=(9.4, 7.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.45], hspace=0.32)

# --- top: the sentence pair, substituted words flagged ---
axt = fig.add_subplot(gs[0])
axt.axis('off')
axt.set_xlim(0, 1)
axt.set_ylim(0, 1)
ref_w, cand_w = toks(REF), toks(CAND_BLEU)
sub_mask = [rw != cw for rw, cw in zip(ref_w, cand_w)]
axt.text(0.0, 0.86, 'reference', fontsize=11, color=GRAY, ha='left')
word_row(axt, ref_w, 0.58,
         facecolors=[LIGHT] * len(ref_w), textcolors=[INK] * len(ref_w),
         x0=0.5, fs=12.0, gap=0.6)
axt.text(0.0, 0.30, 'candidate', fontsize=11, color=GRAY, ha='left')
word_row(axt, cand_w, 0.04,
         facecolors=[end_color if s else LIGHT for s in sub_mask],
         textcolors=['white' if s else INK for s in sub_mask],
         x0=0.5, fs=12.0, gap=0.6)
# rescale the two rows into [0,1] x
axt.set_xlim(0, max(sum(len(w) + 2.2 for w in ref_w),
                    sum(len(w) + 2.2 for w in cand_w)) + 0.5)

# --- bottom: the four precisions + the two means ---
ax = fig.add_subplot(gs[1])
xs = np.arange(1, 5)
bars = ax.bar(xs, ps, width=0.62, color=start_color, edgecolor='none', zorder=3)
for x, p in zip(xs, ps):
    if p > 0:
        ax.text(x, p + 0.022, f'{p:.2f}', ha='center', va='bottom',
                fontsize=13, color=start_color, fontweight='bold')
ax.scatter([4], [0], s=90, facecolor='white', edgecolor=end_color,
           linewidth=2.2, zorder=5)

# arithmetic-mean divider; label sits clear above the dotted line
ax.axhline(amean, color=GRAY, lw=1.5, ls=(0, (5, 3)), zorder=2)
ax.text(6.5, amean + 0.075, f'arithmetic mean = {amean:.2f}\n(not how BLEU combines them)',
        va='bottom', ha='right', fontsize=10.5, color=GRAY, linespacing=1.5)

# one zero precision collapses the geometric mean to 0
ax.annotate('BLEU = geometric\nmean = 0.00', xy=(4.0, 0.0), xytext=(4.55, 0.30),
            va='center', ha='left', fontsize=11, color=end_color, fontweight='bold',
            linespacing=1.5,
            arrowprops=dict(arrowstyle='-|>', color=end_color, lw=1.7,
                            shrinkA=4, shrinkB=8,
                            connectionstyle='arc3,rad=0.12'))

ax.set_xticks(xs)
ax.set_xticklabels(['$p_1$\nunigram', '$p_2$\nbigram', '$p_3$\ntrigram', '$p_4$\n4-gram'])
ax.set_ylabel('modified n-gram precision')
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(0.4, 6.7)
ax.spines['bottom'].set_bounds(1, 4)
ax.spines['left'].set_bounds(0, 1)
despine(ax)
ax.tick_params(labelsize=12.5)
fig.subplots_adjust(left=0.10, right=0.985, top=0.985, bottom=0.075)
save_figure(fig, 'BLEU_ngram_precision')
plt.close()
print(f"1. BLEU  p={[round(p,3) for p in ps]}  AM={amean:.3f}  BLEU={score:.3f}  BP={bp:.2f}")


# ===========================================================================
# 2. METEOR vs BLEU — synonyms BLEU cannot see, and METEOR's order penalty
# ===========================================================================
EQUIV = {'fast': 'quick', 'speedy': 'quick', 'leaps': 'jumps', 'hopped': 'jumps',
         'idle': 'lazy', 'sleepy': 'lazy', 'hound': 'dog'}
cands = [
    ('exact match', "the quick brown fox jumps over the lazy dog"),
    ('synonyms, same order', "the fast brown fox leaps over the idle dog"),
    ('paraphrase, reordered', "the sleepy hound hopped over the speedy fox"),
]

fig, ax = plt.subplots(figsize=(12.8, 5.6))
ax.axis('off')
KIND_C = {'exact': start_color, 'equiv': middle_color, None: '#e3e3e3'}
x0 = 1.0
ref_y = len(cands) * 1.25 + 0.45
# reference strip (descriptor sits above the words, never beside them)
ax.text(x0, ref_y + 0.52, 'reference', fontsize=10.5, color=GRAY, ha='left')
ends = [word_row(ax, toks(REF), ref_y, [LIGHT] * 9, textcolors=[INK] * 9,
                 x0=x0, fs=12.5)[1]]
rowinfo = []
for row, (name, cand_s) in enumerate(cands):
    yy = (len(cands) - 1 - row) * 1.25
    c = toks(cand_s)
    kind_of = {ci: kind for ci, ri, kind in meteor(cand_s, REF, EQUIV)[1]}
    facecolors = [KIND_C[kind_of.get(i)] for i in range(len(c))]
    txtcolors = ['white' if kind_of.get(i) else '#8a8a8a' for i in range(len(c))]
    ax.text(x0, yy + 0.52, name, fontsize=10.5, color=GRAY, ha='left')
    ends.append(word_row(ax, c, yy, facecolors, textcolors=txtcolors, x0=x0, fs=12.5)[1])
    rowinfo.append((yy, bleu(cand_s, REF)[2], meteor(cand_s, REF, EQUIV)[0]))

score_x = max(ends) + 3
for yy, b, m in rowinfo:
    ax.text(score_x, yy, f'BLEU {b:.2f}', fontsize=13, va='center', ha='left',
            color=end_color, fontweight='bold')
    ax.text(score_x + 16, yy, f'METEOR {m:.2f}', fontsize=13, va='center',
            ha='left', color=middle_color, fontweight='bold')
ax.set_xlim(-0.5, score_x + 34)
ax.set_ylim(-1.0, ref_y + 1.1)
handles = [mpatches.Patch(color=start_color, label='exact match'),
           mpatches.Patch(color=middle_color, label='synonym / stem match'),
           mpatches.Patch(color='#e3e3e3', label='no match')]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=3,
          frameon=False, fontsize=11.5, handlelength=1.1)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
save_figure(fig, 'METEOR_vs_BLEU')
plt.close()
for name, cand_s in cands:
    print(f"2. METEOR  {name:24s}  BLEU={bleu(cand_s,REF)[2]:.3f}  METEOR={meteor(cand_s,REF,EQUIV)[0]:.3f}")


# ===========================================================================
# 3. ROUGE — recall never penalizes length; padding only costs precision
# ===========================================================================
REF_SUM = "the central bank raised interest rates to curb inflation"
CORE = "the bank raised rates to curb inflation"          # covers the reference content
FILLER = ("in a widely expected move that analysts said had been "
          "telegraphed for weeks during numerous public speeches").split()

ks = np.arange(0, len(FILLER) + 1)
recs, precs = [], []
for k in ks:
    cand = CORE + ' ' + ' '.join(FILLER[:k])
    p, r = rouge_n(cand, REF_SUM, 1)
    precs.append(p)
    recs.append(r)
recs, precs = np.array(recs), np.array(precs)

fig, ax = plt.subplots(figsize=(9.4, 6.2))
cand_lens = np.array([len(toks(CORE)) + k for k in ks])
ax.fill_between(cand_lens, precs, recs, color=start_color, alpha=0.10, zorder=1)
ax.plot(cand_lens, recs, color=start_color, lw=4, solid_capstyle='round', zorder=3)
ax.plot(cand_lens, precs, color=end_color, lw=4, solid_capstyle='round', zorder=3)
ax.text(cand_lens[-1], recs[-1] + 0.015, 'ROUGE-1 recall',
        ha='right', va='bottom', fontsize=14, color=start_color)
ax.text(cand_lens[-1], precs[-1] - 0.02, 'precision (not reported by ROUGE)',
        ha='right', va='top', fontsize=14, color=end_color)
mid = len(cand_lens) // 2
ax.annotate('padding the summary never lowers recall',
            xy=(cand_lens[mid], recs[mid]), xytext=(cand_lens[mid], 0.90),
            fontsize=11.5, color=GRAY, ha='center', va='bottom',
            arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=1.3,
                            connectionstyle='arc3,rad=0'))
# mark the two named candidates
for x, lbl in [(cand_lens[0], 'concise'), (cand_lens[-1], 'padded')]:
    ax.scatter([x], [recs[0]], s=70, facecolor='white', edgecolor=start_color,
               linewidth=2, zorder=5)
    ax.text(x, recs[0] + 0.11, lbl, ha='right', va='bottom', fontsize=11.5, color=INK)
ax.set_xlabel('summary length (words)')
ax.set_ylabel('ROUGE-1 score')
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlim(cand_lens[0] - 0.5, cand_lens[-1] + 0.5)
ax.spines['left'].set_bounds(0, 1)
ax.spines['bottom'].set_bounds(cand_lens[0], cand_lens[-1])
despine(ax)
ax.tick_params(labelsize=12.5)
fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.11)
save_figure(fig, 'ROUGE_variants')
plt.close()
print(f"3. ROUGE  recall {recs[0]:.2f}->{recs[-1]:.2f}  precision {precs[0]:.2f}->{precs[-1]:.2f}")


# ===========================================================================
# 4. TER — a shift relocates a whole phrase for the cost of one edit
# ===========================================================================
TER_REF = "police arrested the suspect on monday morning"
TER_HYP = "on monday morning police arrested the suspect"
ref_t, hyp_t = toks(TER_REF), toks(TER_HYP)
N = len(ref_t)
phrase = ['on', 'monday', 'morning']
# no-shift accounting: the phrase must be deleted (3) and re-inserted (3)
noshift_edits = 2 * len(phrase)
ter_noshift = noshift_edits / N
ter_shift = 1 / N

fig, ax = plt.subplots(figsize=(12.6, 5.0))
ax.axis('off')


def is_phrase(w):
    return w in phrase


y_hyp, y_ref = 1.0, 3.2
hyp_centers, hyp_end = word_row(
    ax, hyp_t, y_hyp,
    [middle_color if is_phrase(w) else start_color for w in hyp_t],
    textcolors=['white'] * len(hyp_t), x0=0.5, fs=14, gap=0.8)
ref_centers, ref_end = word_row(
    ax, ref_t, y_ref,
    [middle_color if is_phrase(w) else start_color for w in ref_t],
    textcolors=['white'] * len(ref_t), x0=0.5, fs=14, gap=0.8)
ax.text(-0.2, y_ref, 'reference', fontsize=11.5, color=GRAY, ha='right', va='center')
ax.text(-0.2, y_hyp, 'hypothesis', fontsize=11.5, color=GRAY, ha='right', va='center')

# arrow: the misplaced phrase lifts from the front of the hypothesis and drops
# onto its reference position at the end — one move, regardless of phrase length.
# the chord is wide, so a small rad keeps the convex arc inside the two rows.
src = float(np.mean(hyp_centers[0:3]))     # center of the phrase in the hypothesis
dst = float(np.mean(ref_centers[4:7]))     # where it belongs in the reference
ax.annotate('', xy=(dst, y_ref - 0.4), xytext=(src, y_hyp + 0.4),
            arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=2.2, mutation_scale=20,
                            connectionstyle='arc3,rad=-0.06'))
ax.text((src + dst) / 2, y_ref + 0.5, 'one shift moves the whole phrase',
        fontsize=12, color=GRAY, ha='center', va='center')

ax.text(0.5, -0.55,
        f'as plain word edits:  {len(phrase)} deletions + {len(phrase)} insertions '
        f'= {noshift_edits}   $\\rightarrow$   TER = {noshift_edits}/{N} = {ter_noshift:.2f}',
        fontsize=12.5, color=GRAY, ha='left', va='center')
ax.text(0.5, -1.25,
        f'with a shift:  1 phrase moved = 1 edit   $\\rightarrow$   '
        f'TER = 1/{N} = {ter_shift:.2f}',
        fontsize=12.5, color=middle_color, ha='left', va='center', fontweight='bold')

ax.set_xlim(-3.5, max(hyp_end, ref_end) + 1)
ax.set_ylim(-1.7, y_ref + 1.1)
fig.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.02)
save_figure(fig, 'TER_edit_breakdown')
plt.close()
print(f"4. TER  no-shift={ter_noshift:.3f} ({noshift_edits} edits)  shift={ter_shift:.3f}")


# ===========================================================================
# 5. EM vs token-F1 — all-or-nothing vs graceful partial credit
# ===========================================================================
GOLD = "Martin Luther King"
em_cands = [
    "Martin Luther King",
    "Martin Luther King Jr.",
    "Dr. King",
    "Malcolm X",
]
gold_set = squad_norm(GOLD)

fig, ax = plt.subplots(figsize=(12, 5.2))
ax.axis('off')
nrows = len(em_cands)
PITCH, x0 = 9.0, 12.0          # fixed column grid; tokens left-aligned per column


def col_row(words, y, facecolors, textcolors, fs=13):
    for i, w in enumerate(words):
        ax.text(x0 + i * PITCH, y, w, ha='left', va='center', fontsize=fs,
                color=textcolors[i], zorder=4, fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.32', facecolor=facecolors[i],
                          edgecolor='none'))


gold_words = GOLD.split()
gold_y = nrows + 0.4
ax.text(x0 - 5.5, gold_y, 'gold answer', fontsize=11, color=GRAY, ha='right', va='center')
col_row(gold_words, gold_y, [LIGHT] * len(gold_words), [INK] * len(gold_words))
score_x = x0 + 4 * PITCH
for row, cand in enumerate(em_cands):
    yy = nrows - 1 - row
    words = cand.split()
    facecolors, txt = [], []
    for w in words:
        nw = squad_norm(w)
        hit = bool(nw) and nw[0] in gold_set
        facecolors.append(start_color if hit else middle_color)
        txt.append('white')
    col_row(words, yy, facecolors, txt)
    em = exact_match(cand, GOLD)
    f1 = token_f1(cand, GOLD)
    ax.text(score_x, yy, f'EM {em}', fontsize=12.5, va='center', ha='left',
            color=start_color if em else end_color, fontweight='bold')
    ax.text(score_x + 8.5, yy, f'token-F1 {f1:.2f}', fontsize=12.5, va='center',
            ha='left', color=middle_color, fontweight='bold')

ax.set_xlim(-6, score_x + 24)
ax.set_ylim(-0.7, gold_y + 0.6)
handles = [mpatches.Patch(color=start_color, label='token shared with gold'),
           mpatches.Patch(color=middle_color, label='token not in gold')]
ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, -0.1), ncol=2,
          frameon=False, fontsize=11.5, handlelength=1.2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.06)
save_figure(fig, 'EM_vs_F1')
plt.close()
for cand in em_cands:
    print(f"5. EM  {cand:24s}  EM={exact_match(cand,GOLD)}  F1={token_f1(cand,GOLD):.3f}")


# ===========================================================================
# 6. WER — the best alignment splits errors into substitutions / deletions /
#    insertions
# ===========================================================================
WER_REF = "the quick brown fox jumps over the lazy dog"
WER_HYP = "the very quick brown box jumps over the lazy"   # ins 'very', fox->box, drop 'dog'
ops = edit_align(toks(WER_REF), toks(WER_HYP))
S = sum(o == 'S' for o, _, _ in ops)
D = sum(o == 'D' for o, _, _ in ops)
I = sum(o == 'I' for o, _, _ in ops)
Nref = len(toks(WER_REF))
wer = (S + D + I) / Nref

OP_COLOR = {'C': start_color, 'S': end_color, 'D': '#b58900', 'I': middle_color}
OP_NAME = {'S': 'sub', 'D': 'del', 'I': 'ins'}

fig, ax = plt.subplots(figsize=(12.6, 4.8))
ax.axis('off')
x = 0.5
y_ref, y_hyp = 2.7, 1.0
for op, rw, hw in ops:
    w = max(len(rw or ''), len(hw or '')) + 1.8
    cx = x + w / 2
    col = OP_COLOR[op]
    if rw is not None:
        ax.text(cx, y_ref, rw, ha='center', va='center', fontsize=13.5,
                color='white', zorder=4, fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=col if op != 'C' else start_color, edgecolor='none'))
    else:
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w / 2 + 0.3, y_ref - 0.32), w - 0.6, 0.64,
            boxstyle='round,pad=0.02', facecolor='none', edgecolor=LIGHT,
            linewidth=1.3, linestyle=(0, (3, 2)), zorder=2))
    if hw is not None:
        ax.text(cx, y_hyp, hw, ha='center', va='center', fontsize=13.5,
                color='white', zorder=4, fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=col if op != 'C' else start_color, edgecolor='none'))
    else:
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w / 2 + 0.3, y_hyp - 0.32), w - 0.6, 0.64,
            boxstyle='round,pad=0.02', facecolor='none', edgecolor=LIGHT,
            linewidth=1.3, linestyle=(0, (3, 2)), zorder=2))
    if op != 'C':
        ax.text(cx, (y_ref + y_hyp) / 2, OP_NAME[op], ha='center', va='center',
                fontsize=10.5, color=col, fontweight='bold', zorder=5,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none'))
    x += w + 0.7

ax.text(-0.2, y_ref, 'reference', fontsize=11.5, color=GRAY, ha='right', va='center')
ax.text(-0.2, y_hyp, 'hypothesis', fontsize=11.5, color=GRAY, ha='right', va='center')
ax.text(0.5, -0.35,
        f'WER = (S + D + I) / N = ({S} + {D} + {I}) / {Nref} = {wer:.2f}',
        fontsize=13.5, color=INK, ha='left', va='center')
ax.set_xlim(-3.5, x + 1)
ax.set_ylim(-0.9, 3.5)
fig.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.02)
save_figure(fig, 'WER_vs_errors')
plt.close()
print(f"6. WER  S={S} D={D} I={I} N={Nref}  WER={wer:.3f}")

print("\nAll NLP figures regenerated.")
