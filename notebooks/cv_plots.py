"""Generate figures for the Computer Vision chapter.

Redesigned to show each metric's *mechanism* (not just an output number) and to
compute every displayed value from real data — no hardcoded scores.

Run from the notebooks/ directory:  uv run python cv_plots.py
"""
import sys
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
from style import *
import cv2

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


def psnr(distorted, ref, L=255.0):
    mse = np.mean((distorted.astype(np.float64) - ref.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(L * L / mse)


def ssim(a, b, L=255.0):
    """Canonical Wang et al. (2004) SSIM with an 11x11 Gaussian window."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    win, sig = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(a, win, sig)
    mu_b = cv2.GaussianBlur(b, win, sig)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = cv2.GaussianBlur(a * a, win, sig) - mu_a2
    vb = cv2.GaussianBlur(b * b, win, sig) - mu_b2
    vab = cv2.GaussianBlur(a * b, win, sig) - mu_ab
    m = ((2 * mu_ab + C1) * (2 * vab + C2)) / ((mu_a2 + mu_b2 + C1) * (va + vb + C2))
    return float(m.mean())


def tune(make, lo, hi, ref, target=24.0, iters=44):
    """Bisection on a distortion strength so that PSNR(make(strength)) ~= target.
    Assumes PSNR decreases monotonically as strength grows."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if psnr(make(mid), ref) > target:
            lo = mid          # too little distortion -> stronger
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_div2k(gray=False, width=520):
    path = FIGURES_DIR / 'DIV2K_0803.png'
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    im = cv2.imread(str(path), flag)
    scale = width / im.shape[1]
    im = cv2.resize(im, (width, int(im.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if not gray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float64)


# ===========================================================================
# 1. Pixel Accuracy — why the majority class dictates the score
#    Variable-width bars: width = share of pixels, height = per-class accuracy.
#    Bar AREA = that class's contribution to PA. PA (frequency-weighted) rides
#    the fat background bar; mPA (equal-weighted) is dragged down by the thin
#    minority bars.
# ===========================================================================
fig, ax = plt.subplots(figsize=(12.5, 5.8))
names = ['Background', 'Class 1', 'Class 2']
shares = ['85% of pixels', '10% of pixels', '5% of pixels']
freq = np.array([0.85, 0.10, 0.05])
acc = np.array([0.98, 0.45, 0.20])
bar_colors = ['#d4d4d4', start_color, middle_color]

pa = float(np.sum(freq * acc))   # frequency-weighted average  == Pixel Accuracy
mpa = float(np.mean(acc))        # equal-weighted average      == Mean Pixel Accuracy

edges = np.concatenate([[0.0], np.cumsum(freq)])
for i in range(len(freq)):
    left, w, h = edges[i], freq[i], acc[i]
    ax.bar(left, h, width=w, align='edge', color=bar_colors[i],
           edgecolor='white', linewidth=2.5, zorder=2)
    ax.text(left + w / 2, h + 0.022, f'{h:.0%}', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#6f6f6f' if i == 0 else bar_colors[i])

# aggregation lines
ax.axhline(pa, color=NML_DARK_RED, lw=2.5, zorder=3)
ax.axhline(mpa, color=NML_PURPLE, lw=2.5, ls=(0, (5, 3)), zorder=3)
ax.text(1.015, pa, f'Pixel Accuracy = {pa:.0%}', va='center', ha='left',
        color=NML_DARK_RED, fontsize=12.5, fontweight='bold')
ax.text(1.015, mpa, f'Mean Pixel\nAccuracy = {mpa:.0%}', va='center', ha='left',
        color=NML_PURPLE, fontsize=12.5, fontweight='bold')

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_ylabel('per-class accuracy')
ax.set_xticks([])
ax.margins(x=0)
despine(ax, keep=('left',))
ax.annotate('bar width = share of pixels   ·   area = contribution to Pixel Accuracy',
            xy=(0.5, 1.04), xycoords='axes fraction', ha='center', va='bottom',
            fontsize=11, color='#7a7a7a')
legend_handles = [matplotlib.patches.Patch(facecolor=bar_colors[i], edgecolor='#bbbbbb',
                  label=f'{names[i]} — {shares[i]}') for i in range(3)]
ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.03),
          ncol=3, frameon=False, fontsize=11.5)
fig.subplots_adjust(left=0.085, right=0.79, top=0.88, bottom=0.13)
save_figure(fig, 'Pixel_Accuracy_imbalance')
plt.close()
print(f"1. Pixel Accuracy: PA={pa:.3f}  mPA={mpa:.3f}")


# ===========================================================================
# 2. OKS — smooth Gaussian similarity vs the hard PCK cutoff
#    Fixes the dead x-range (was 0..3 with everything flat past 0.7) and
#    overlays PCK's pass/fail step so the "OKS = smooth PCK" point is visible.
# ===========================================================================
fig, ax = plt.subplots(figsize=(9.2, 6.4))
d = np.linspace(0, 0.8, 500)
for k, color, label in [
    (0.05, start_color, 'k = 0.05  (strict — eyes)'),
    (0.10, middle_color, 'k = 0.10  (moderate — shoulders)'),
    (0.20, end_color, 'k = 0.20  (lenient — hips)'),
]:
    ax.plot(d, np.exp(-d ** 2 / (2 * k ** 2)), color=color, lw=4,
            solid_capstyle='round', label=label, zorder=3)

# PCK-style hard threshold (pass/fail) for contrast
thr = 0.30
ax.plot([0, thr, thr, 0.8], [1, 1, 0, 0], color='#9a9a9a', lw=2.2,
        ls=(0, (5, 3)), zorder=2, label='PCK: hard pass/fail')
ax.text(thr + 0.012, 0.55, 'a hard threshold accepts\neverything left of the line,\nrejects everything right',
        fontsize=10, color='#7a7a7a', va='center')

ax.set_xlabel('normalized keypoint error  ($d$ / scale)')
ax.set_ylabel('similarity score')
ax.set_xlim(0, 0.8)
ax.set_ylim(0, 1.04)
ax.legend(fontsize=12, frameon=False, loc='upper right')
despine(ax)
ax.grid(axis='y', alpha=0.15)
fig.tight_layout()
save_figure(fig, 'OKS_gaussian_falloff')
plt.close()
print("2. OKS: done")


# ===========================================================================
# 3. Sørensen–Dice — overlap of two real masks, Dice computed (not hardcoded)
#    Two equal discs (prediction vs ground truth) pushed apart; Dice and IoU
#    measured from the actual pixels.
# ===========================================================================
fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
N = 500
yy, xx = np.mgrid[0:N, 0:N]
r = 0.26 * N
cyc = N / 2
c_inter = mcolors.to_rgb(start_color)   # overlap
c_gt = mcolors.to_rgb(end_color)        # ground-truth only (missed)
c_pred = mcolors.to_rgb(middle_color)   # prediction only (false positive)

for ax, off_frac, title in [
    (axs[0], 0.060, 'High overlap'),
    (axs[1], 0.185, 'Moderate overlap'),
    (axs[2], 0.330, 'Low overlap'),
]:
    off = off_frac * N
    gt = (xx - (cyc - off / 2)) ** 2 + (yy - cyc) ** 2 <= r ** 2
    pred = (xx - (cyc + off / 2)) ** 2 + (yy - cyc) ** 2 <= r ** 2
    inter = gt & pred
    dice = 2 * inter.sum() / (gt.sum() + pred.sum())
    iou = inter.sum() / (gt | pred).sum()

    disp = np.ones((N, N, 3))
    disp[gt & ~pred] = c_gt
    disp[pred & ~gt] = c_pred
    disp[inter] = c_inter
    ax.imshow(disp, interpolation='nearest')
    ax.set_title(f'{title}\nDice = {dice:.2f}   ·   IoU = {iou:.2f}',
                 fontsize=15, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

handles = [
    matplotlib.patches.Patch(color=c_inter, label='Overlap  (counts toward Dice)'),
    matplotlib.patches.Patch(color=c_gt, label='Ground truth only  (missed)'),
    matplotlib.patches.Patch(color=c_pred, label='Prediction only  (false positive)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
           fontsize=12, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.12, wspace=0.08)
save_figure(fig, 'Dice_overlap_examples')
plt.close()
print("3. Dice: done")


# ===========================================================================
# 4. PQ — the SQ x RQ plane with iso-PQ contours
#    PQ = SQ * RQ, so models live on a plane and equal-PQ curves are hyperbolas.
#    Models B and C sacrifice opposite halves yet land on the same low contour.
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.6, 7.6))
g = np.linspace(0.001, 1, 400)
RQ, SQ = np.meshgrid(g, g)
PQ = SQ * RQ
ax.imshow(PQ, origin='lower', extent=[0, 1, 0, 1], cmap=nml_cmap.reversed(),
          alpha=0.16, aspect='auto')
cs = ax.contour(RQ, SQ, PQ, levels=[0.2, 0.4, 0.6, 0.8],
                colors='#7a7a7a', linewidths=1.1)
ax.clabel(cs, fmt='PQ = %.1f', fontsize=10.5, inline_spacing=8)

# labels extend toward the open interior, away from the plot edges
models = [
    ('A — good all round', 0.90, 0.85, (0.87, 0.85), 'right', 'center'),
    ('B — segments well,\nmisses objects', 0.50, 0.90, (0.47, 0.88), 'right', 'center'),
    ('C — finds objects,\nsloppy masks', 0.85, 0.55, (0.82, 0.55), 'right', 'center'),
    ('D — poor', 0.40, 0.50, (0.43, 0.45), 'left', 'top'),
]
for label, rqv, sqv, (tx, ty), ha, va in models:
    ax.scatter(rqv, sqv, s=190, color=NML_PURPLE, edgecolor='white',
               linewidth=1.8, zorder=5)
    ax.annotate(f'{label}\nPQ = {rqv * sqv:.2f}', (rqv, sqv), xytext=(tx, ty),
                ha=ha, va=va, fontsize=11, fontweight='bold', zorder=6)

ax.set_xlabel('Recognition Quality  (RQ — were the objects found?)')
ax.set_ylabel('Segmentation Quality  (SQ — how well are they outlined?)')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
despine(ax)
fig.tight_layout()
save_figure(fig, 'PQ_decomposition')
plt.close()
print("4. PQ: done")


# ===========================================================================
# 5. SSIM vs PSNR — same PSNR, very different SSIM (Wang et al. demonstration)
#    One image, three distortions tuned to the SAME PSNR. PSNR can't tell them
#    apart; SSIM tracks how bad each actually looks.
# ===========================================================================
ref = load_div2k(gray=True, width=440)
ref = ref[40:-40, 60:-60]   # central crop with texture
TARGET = 24.0


def f_shift(c):
    return np.clip(ref + c, 0, 255)


def f_blur(s):
    return cv2.GaussianBlur(ref, (0, 0), s)


def _noise(std):
    return np.clip(ref + RNG.normal(0, std, ref.shape), 0, 255)


# noise is stochastic; freeze one realization per strength via a fixed draw
_noise_unit = RNG.normal(0, 1, ref.shape)


def f_noise(std):
    return np.clip(ref + std * _noise_unit, 0, 255)


c = tune(f_shift, 0, 120, ref, TARGET)
s = tune(f_blur, 0.3, 12, ref, TARGET)
ns = tune(f_noise, 0, 120, ref, TARGET)

panels = [
    ('Original', ref),
    ('Brightness shift', f_shift(c)),
    ('Blur', f_blur(s)),
    ('Gaussian noise', f_noise(ns)),
]
fig, axs = plt.subplots(1, 4, figsize=(16, 5.2))
for ax, (name, img) in zip(axs, panels):
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    if name == 'Original':
        sub = 'PSNR = ∞   ·   SSIM = 1.00'
    else:
        sub = f'PSNR = {psnr(img, ref):.1f} dB   ·   SSIM = {ssim(img, ref):.2f}'
    ax.set_title(f'{name}\n{sub}', fontsize=13.5, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.suptitle('Three distortions at the same PSNR — SSIM still separates them by how they look',
             fontsize=14, y=1.02)
fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.06)
save_figure(fig, 'SSIM_vs_PSNR')
plt.close()
print(f"5. SSIM: shift c={c:.1f} blur s={s:.2f} noise std={ns:.1f}")


# ===========================================================================
# 6. PSNR — correct dB on a degrading image + the log law
#    Left: one image at rising noise, PSNR computed correctly (the old figure
#    subtracted uint8 arrays and underflowed). Right: PSNR is linear in
#    log(MSE), so equal dB steps mean multiplicative error.
# ===========================================================================
img = load_div2k(gray=False, width=420)
_unit = RNG.normal(0, 1, img.shape)
sigmas = [0, 12, 28, 60]

fig = plt.figure(figsize=(13, 6.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.45, 1], hspace=0.4, wspace=0.06)

pts = []
for i, sg in enumerate(sigmas):
    noisy = np.clip(img + sg * _unit, 0, 255) if sg else img
    p = psnr(noisy, img)
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(noisy.astype(np.uint8))
    ax.set_title('Original' if sg == 0 else f'{p:.0f} dB', fontsize=14,
                 pad=6, fontweight='bold' if sg == 0 else 'normal')
    ax.axis('off')
    if sg:
        pts.append((np.mean((noisy - img) ** 2), p))

# bottom (full width): PSNR is linear in log(MSE) — a straight line
axc = fig.add_subplot(gs[1, :])
mse_grid = np.logspace(0.3, 4, 200)
axc.plot(mse_grid, 10 * np.log10(255.0 ** 2 / mse_grid), color=middle_color, lw=3,
         solid_capstyle='round', zorder=2)
mse_pts, p_pts = zip(*pts)
axc.scatter(mse_pts, p_pts, s=95, color=NML_DARK_RED, edgecolor='white',
            linewidth=1.5, zorder=4)
for anchor, txt in [(30, 'clean'), (20, 'artifacts visible')]:
    axc.axhline(anchor, color='#9a9a9a', lw=1, ls=(0, (4, 3)), zorder=1)
    axc.text(2.4, anchor + 0.8, f'{anchor} dB — {txt}', ha='left', va='bottom',
             fontsize=11, color='#7a7a7a')
axc.set_xscale('log')
axc.set_xlabel('mean squared error  (log scale)')
axc.set_ylabel('PSNR (dB)')
axc.set_xlim(2, 1e4)
axc.set_ylim(8, 46)
despine(axc)
axc.grid(alpha=0.15)
fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.11)
save_figure(fig, 'PSNR_plot')
plt.close()
print("6. PSNR: done")

print("\nAll CV figures regenerated.")
