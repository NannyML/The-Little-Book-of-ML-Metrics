"""Figures for the Computer Vision chapter, drawn at printed size.

Every displayed value is computed here, from the masks, images and keypoints
drawn. Run: uv run python notebooks/cv_figures.py [NAME ...]
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import cv2
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from style import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
def psnr(distorted, ref, L=255.0):
    mse = np.mean((distorted.astype(np.float64) - ref.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(L * L / mse)


def ssim(a, b, L=255.0):
    """Wang et al. (2004) SSIM with an 11x11 Gaussian window (sigma 1.5)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    va = blur(a * a) - mu_a ** 2
    vb = blur(b * b) - mu_b ** 2
    vab = blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(m.mean())


def tune(make, lo, hi, ref, target=24.0, iters=44):
    """Bisection on a distortion strength so that PSNR(make(strength)) ~= target."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if psnr(make(mid), ref) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def photo(gray=False, width=560):
    """The DIV2K flower, cropped 3:2 to the flower head so detail shows in print."""
    im = cv2.imread(str(FIGURES_DIR / 'DIV2K_0803.png'),
                    cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    im = im[560:1410, 380:1655]
    im = cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1]))), interpolation=cv2.INTER_AREA)
    if not gray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float64)


def image_grid(height):
    fig, axes = book_figure(1.0, height, 2, 2)
    for ax in axes.flat:
        bare_axes(ax)
    return fig, axes.flat


# ===========================================================================
def pixel_accuracy():
    names = ['background', 'class 1', 'class 2']
    freq = np.array([0.85, 0.10, 0.05])
    acc = np.array([0.98, 0.45, 0.20])
    colors = ['#cfecef', NML_CYAN, NML_PURPLE]
    pa, mpa = float(freq @ acc), float(acc.mean())
    fig, ax = book_figure(1.0, 2.55)
    edges = np.r_[0.0, np.cumsum(freq)]
    for left, w, h, c in zip(edges[:-1], freq, acc, colors):
        ax.bar(left, h, width=w, align='edge', color=c, edgecolor='white', linewidth=0.8)
        if w < 0.08:
            ax.text(left + w + 0.008, h / 2, f'{h:.0%}', ha='left', va='center', color=INK)
        else:
            ax.text(left + w / 2, h + 0.02, f'{h:.0%}', ha='center', va='bottom', color=INK)
    ax.plot([0, 1], [pa, pa], color=NML_RED, lw=LW_THIN + 0.3)
    ax.plot([0, 1], [mpa, mpa], color=NML_PURPLE, lw=LW_THIN + 0.3, ls=(0, (3, 2)))
    ax.text(0.02, pa - 0.025, f'pixel accuracy = {num(100 * pa, 1)}%', va='top', color=NML_RED)
    ax.text(0.02, mpa - 0.025, f'mean pixel accuracy = {num(100 * mpa, 1)}%', va='top', color=NML_PURPLE)
    ax.set_xlim(0, 1.07)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=['0%', '25%', '50%', '75%', '100%'])
    ax.set_xticks([])
    ax.set_ylabel('pixels labeled correctly\nwithin each class')
    ax.set_xlabel('share of all pixels (bar width)')
    tidy_axes(ax, bottom=False)
    fig.legend(handles=[Patch(color=c, label=f'{n}: {f:.0%} of pixels') for c, n, f in zip(colors, names, freq)],
               loc='outside lower center', ncol=2, handlelength=1.0)
    save_figure(fig, 'Pixel_Accuracy_imbalance')
    plt.close(fig)
    print(f'Pixel accuracy {pa:.4f} mean {mpa:.4f}')


# ===========================================================================
def pck():
    truth = np.array([[0, 1.7], [-.35, 1.6], [.35, 1.6], [-.50, 1.1], [.50, 1.1],
                      [-.60, .6], [.60, .6], [0, .7], [-.20, .65], [.20, .65],
                      [-.20, 0], [.20, 0], [-.20, -.6], [.20, -.6]])
    pred = truth + np.array([.04, .03])
    pred[1] = truth[1] + [-.25, .15]
    pred[5] = truth[5] + [-.55, .05]
    pred[6] = truth[6] + [.12, -.08]
    pred[13] = truth[13] + [.32, 0]
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 7), (7, 8), (7, 9), (8, 10),
             (9, 11), (10, 12), (11, 13)]
    L = float(np.linalg.norm(truth[0] - truth[7]))
    dist = np.linalg.norm(pred - truth, axis=1)
    circled = [1, 5, 13]          # the three joints that miss at alpha = 0.2
    fig, axes = book_figure(1.0, 2.55, 1, 3)
    counts = []
    for i, ax in enumerate(axes):
        for a, b in edges:
            ax.plot(*truth[[a, b]].T, color=REF, lw=0.8, zorder=1)
        ax.scatter(*truth.T, s=7, color=INK, zorder=3, linewidths=0)
        ax.add_patch(Circle((0, 2.03), .16, fill=False, edgecolor=REF, lw=0.8))
        if i == 0:
            ax.annotate('', (.88, 1.7), (.88, .7), arrowprops=dict(arrowstyle='<->', color=NML_PURPLE,
                        lw=LW_THIN + 0.1, shrinkA=0, shrinkB=0, mutation_scale=6))
            for yy in (1.7, .7):
                ax.plot([0, .84], [yy, yy], lw=0.5, color=NML_PURPLE, ls=':')
            ax.text(.95, 1.2, '$L$', va='center', color=NML_PURPLE)
            ax.set_title('reference', fontsize=TEXT_PT)
        else:
            alpha = [.2, .5][i - 1]
            ok = dist <= alpha * L
            counts.append(int(ok.sum()))
            for a, b in edges:
                ax.plot(*pred[[a, b]].T, color=NML_CYAN, lw=0.7, alpha=0.6, zorder=2)
            ax.scatter(*pred.T, s=10, c=np.where(ok, NML_CYAN, NML_RED), zorder=4, linewidths=0)
            for j in circled:
                ax.add_patch(Circle(truth[j], alpha * L, fill=False, ec=NML_PURPLE, lw=0.7))
            ax.set_title(f'$\\alpha$ = {alpha:.1f}: {ok.sum()}/{len(ok)} correct', fontsize=TEXT_PT)
        ax.set(xlim=(-1.3, 1.25), ylim=(-1.2, 2.25), aspect='equal')
        bare_axes(ax)
    handles = [Line2D([], [], marker='o', ls='', ms=3.5, color=INK, label='reference joint'),
               Line2D([], [], marker='o', ls='', ms=3.5, color=NML_CYAN, label='within tolerance'),
               Line2D([], [], marker='o', ls='', ms=3.5, color=NML_RED, label='outside tolerance'),
               Line2D([], [], marker='o', ls='', ms=7, mfc='none', mec=NML_PURPLE,
                      label='tolerance circle, radius $\\alpha L$')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, handletextpad=0.3)
    assert L == 1 and counts == [11, 13], counts
    save_figure(fig, 'PCK_threshold_visual')
    plt.close(fig)
    print('PCK distances', np.round(dist, 3).tolist(), counts)


# ===========================================================================
def oks():
    d = np.linspace(0, 0.8, 500)
    fig, ax = book_figure(0.82, 2.4)
    thr = 0.30
    ax.plot([0, thr, thr, 0.8], [1, 1, 0, 0], color=REF, lw=LW_THIN + 0.2, ls=(0, (3, 2)), zorder=2)
    ax.text(thr + 0.012, 0.5, 'hypothetical\nPCK cutoff\nat 0.30', color=MUTED, va='center')
    # COCO's per-keypoint constants k_i = 2 sigma_i
    rows = [(0.05, NML_CYAN, 'eyes, $k$ = 0.05'), (0.158, NML_PURPLE, 'shoulders, $k$ = 0.158'),
            (0.214, NML_RED, 'hips, $k$ = 0.214')]
    for k, c, _ in rows:
        ax.plot(d, np.exp(-d ** 2 / (2 * k ** 2)), color=c, zorder=3)
    ax.legend(handles=[Line2D([], [], color=c, label=t) for _, c, t in rows], loc='upper right',
              bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], labels=[num(v, 1) for v in [0, 0.2, 0.4, 0.6, 0.8]])
    unit_ticks(ax, 'y', 0.5)
    ax.set_xlabel('normalized keypoint error $d/s$')
    ax.set_ylabel('keypoint similarity')
    tidy_axes(ax)
    save_figure(fig, 'OKS_gaussian_falloff')
    plt.close(fig)


# ===========================================================================
def psnr_figure():
    rng = np.random.default_rng(42)
    img = photo()
    unit = rng.normal(0, 1, img.shape)
    fig, axes = image_grid(3.1)
    out = []
    for ax, sg in zip(axes, [0, 12, 28, 60]):
        noisy = np.clip(img + sg * unit, 0, 255) if sg else img
        p = psnr(noisy, img)
        ax.imshow(noisy.astype(np.uint8))
        ax.set_title('original: PSNR = ∞' if sg == 0 else f'PSNR = {num(p, 1)} dB', fontsize=TEXT_PT)
        if sg:
            out.append((round(float(np.mean((noisy - img) ** 2)), 2), round(p, 3)))
    save_figure(fig, 'PSNR_plot')
    plt.close(fig)
    print('PSNR (MSE, dB):', json.dumps(out))


def ssim_figure():
    rng = np.random.default_rng(7)
    ref = photo(gray=True)
    unit = rng.normal(0, 1, ref.shape)
    makes = {'brightness shift': lambda c: np.clip(ref + c, 0, 255),
             'blur': lambda s: cv2.GaussianBlur(ref, (0, 0), s),
             'Gaussian noise': lambda s: np.clip(ref + s * unit, 0, 255)}
    strengths = {'brightness shift': tune(makes['brightness shift'], 0, 120, ref),
                 'blur': tune(makes['blur'], 0.3, 20, ref),
                 'Gaussian noise': tune(makes['Gaussian noise'], 0, 120, ref)}
    fig, axes = image_grid(3.35)
    panels = [('original', ref)] + [(n, makes[n](strengths[n])) for n in makes]
    out = {}
    for ax, (name, im) in zip(axes, panels):
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        if name == 'original':
            title = 'original: PSNR = ∞, SSIM = 1.00'
        else:
            p, s = psnr(im, ref), ssim(im, ref)
            out[name] = (round(p, 2), round(s, 4))
            title = f'{name}\nPSNR = {p:.0f} dB, SSIM = {num(s)}'
        ax.set_title(title, fontsize=TEXT_PT, linespacing=1.3)
    save_figure(fig, 'SSIM_vs_PSNR')
    plt.close(fig)
    print('SSIM', out, {k: round(v, 3) for k, v in strengths.items()})


# ===========================================================================
def dice():
    N = 500
    yy, xx = np.mgrid[0:N, 0:N]
    r, cy = 0.26 * N, N / 2
    fig, axes = book_figure(1.0, 1.7, 1, 3)
    span = int(r + 0.330 * N / 2) + 5            # half-width of the widest pair
    box = (slice(int(cy - r) - 5, int(cy + r) + 6), slice(int(cy) - span, int(cy) + span + 1))
    for ax, off_frac, name in zip(axes, [0.060, 0.185, 0.330], ['high', 'moderate', 'low']):
        off = off_frac * N
        pred = (xx - (cy - off / 2)) ** 2 + (yy - cy) ** 2 <= r ** 2
        gt = (xx - (cy + off / 2)) ** 2 + (yy - cy) ** 2 <= r ** 2
        inter = gt & pred
        d = 2 * inter.sum() / (gt.sum() + pred.sum())
        iou = inter.sum() / (gt | pred).sum()
        disp = np.ones((N, N, 3))
        disp[pred & ~gt] = mcolors.to_rgb(PRED_ONLY)
        disp[gt & ~pred] = mcolors.to_rgb(TRUTH_ONLY)
        disp[inter] = mcolors.to_rgb(OVERLAP)
        disp = disp[box]
        ax.imshow(disp, interpolation='nearest')
        ax.set_title(f'{name} overlap\nDice = {num(d)}, IoU = {num(iou)}', fontsize=TEXT_PT, linespacing=1.3)
        bare_axes(ax)
        print(f'Dice {name}: Dice {d:.4f} IoU {iou:.4f}')
    handles = [Patch(color=OVERLAP, label='overlap (true positives)'),
               Patch(color=PRED_ONLY, label='prediction only (false positives)'),
               Patch(color=TRUTH_ONLY, label='ground truth only (false negatives)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, handlelength=1.0)
    save_figure(fig, 'Dice_overlap_examples')
    plt.close(fig)


# ===========================================================================
def pq():
    fig, ax = book_figure(0.82, 2.75)
    g = np.linspace(0.002, 1, 500)
    RQ, SQ = np.meshgrid(g, np.linspace(0.5, 1, 300))
    levels = [0.2, 0.4, 0.6, 0.8]
    cs = ax.contour(RQ, SQ, RQ * SQ, levels=levels, colors=[REF], linewidths=LW_THIN)
    for v in levels:
        ax.text(v, 1.012, f'PQ = {v:g}', ha='center', va='bottom', color=MUTED, fontsize=SMALL_PT)
    models = [('A', 'good all round', 0.90, 0.85, (-6, 0), 'right'),
              ('B', 'good masks, missed objects', 0.50, 0.90, (6, 0), 'left'),
              ('C', 'found objects, rough masks', 0.85, 0.55, (-6, 0), 'right')]
    lines = []
    for letter, desc, rq, sq, off, ha in models:
        ax.scatter(rq, sq, s=26, color=NML_PURPLE, edgecolors='white', linewidths=0.8, zorder=5)
        ax.annotate(letter, (rq, sq), xytext=off, textcoords='offset points', ha=ha, va='center',
                    color=INK, fontweight='semibold', zorder=6,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
        lines.append(f'{letter}: {desc}, PQ = {num(rq * sq)}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1)
    unit_ticks(ax, 'x', 0.5)
    ax.set_yticks([0.5, 0.75, 1.0], labels=['0.50', '0.75', '1.00'])
    ax.set_xlabel('recognition quality (RQ)')
    ax.set_ylabel('segmentation quality (SQ)')
    tidy_axes(ax)
    fig.supxlabel('\n'.join(lines), fontsize=TEXT_PT, ha='left', x=0.02, linespacing=1.35)
    save_figure(fig, 'PQ_decomposition')
    plt.close(fig)


FIGURES = {'Pixel_Accuracy_imbalance': pixel_accuracy, 'PCK_threshold_visual': pck,
           'OKS_gaussian_falloff': oks, 'PSNR_plot': psnr_figure, 'Dice_overlap_examples': dice,
           'PQ_decomposition': pq, 'SSIM_vs_PSNR': ssim_figure}

if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
