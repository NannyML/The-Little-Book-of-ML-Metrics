"""Two-dimensional figures for the Classification chapter.

Balanced Accuracy, F-beta, ROC AUC, PR AUC, Brier Score, Log Loss, Jaccard,
D-squared Log Loss, P4 and Cohen's kappa. Every printed number is computed here.
Run: uv run python notebooks/classification_figures.py [NAME ...]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Circle
from scipy.optimize import brentq
from style import *  # noqa: F401,F403


def auc(x, y):
    return float(np.trapezoid(y, x))


# ---------------------------------------------------------------------------
def balanced_accuracy():
    # One model, one pair of class recalls, evaluated on two class mixes.
    sens, spec = 0.79, 0.85
    rows = []
    for pos in (0.05, 0.5):
        acc = pos * sens + (1 - pos) * spec
        bal = (sens + spec) / 2
        rows.append(((1 - pos, 0.5), (acc, bal)))   # (always negative, model)
    fig, axes = book_figure(1.0, 1.95, 1, 2, sharey=True)
    titles = ['95% of cases negative', 'balanced classes (50/50)']
    x = np.arange(2)
    w = 0.34
    for ax, ((n_acc, n_bal), (m_acc, m_bal)), title in zip(axes, rows, titles):
        for dx, vals, col in [(-w / 2 - 0.01, (n_acc, n_bal), NML_RED), (w / 2 + 0.01, (m_acc, m_bal), NML_CYAN)]:
            ax.bar(x + dx, vals, w, color=col, linewidth=0)
            for xi, v in zip(x + dx, vals):
                ax.text(xi, v + 0.025, num(v), ha='center', va='bottom', color=INK)
        ax.set_xticks(x, labels=['accuracy', 'balanced\naccuracy'])
        ax.set_title(title, loc='left', fontsize=TEXT_PT)
        ax.set_ylim(0, 1.1)
        unit_ticks(ax, 'y', 0.5)
        tidy_axes(ax)
        ax.tick_params(axis='x', length=0)
    axes[0].set_ylabel('score')
    handles = [matplotlib.patches.Patch(color=NML_RED, label='always predicts negative'),
               matplotlib.patches.Patch(color=NML_CYAN, label='example model')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, handlelength=1.0)
    save_figure(fig, 'Balanced_Accuracy_comparison')
    plt.close(fig)
    print('Balanced accuracy rows', [[round(v, 3) for pair in r for v in pair] for r in rows])


# ---------------------------------------------------------------------------
def f_beta():
    recall = np.linspace(0.005, 1.0, 400)
    precision = 0.8
    fig, ax = book_figure(0.82, 2.55)
    curves = [(0.5, NML_CYAN, '$\\beta$ = 0.5, favors precision'),
              (1.0, NML_PURPLE, '$\\beta$ = 1 (F1)'),
              (2.0, NML_RED, '$\\beta$ = 2, favors recall')]
    ends = []
    for beta, col, _ in curves:
        f = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        ax.plot(recall, f, color=col)
        ends.append(f[-1])
    from dedicated_figures import spread
    for (beta, col, lab), y in zip(curves, spread(ends, 0.075)):
        ax.text(1.02, y, lab, color=col, va='center')
    ax.scatter([0.8], [0.8], s=18, facecolors='white', edgecolors=INK, linewidths=0.8, zorder=5)
    note(ax, 0.97, 0.33, 'curves meet where\nrecall = precision = 0.8', xy=(0.8, 0.775), ha='right', va='center')
    note(ax, 0.98, 0.06, 'precision fixed at 0.8', ha='right', va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    unit_ticks(ax, 'x', 0.2)
    unit_ticks(ax, 'y', 0.2)
    ax.set_xlabel('recall')
    ax.set_ylabel('F-beta score')
    tidy_axes(ax)
    save_figure(fig, 'F_beta_curves')
    plt.close(fig)


# ---------------------------------------------------------------------------
def roc():
    fpr_good = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0])
    tpr_good = np.array([0, 0.5, 0.75, 0.88, 0.95, 0.98, 1.0])
    fpr_weak = np.array([0, 0.1, 0.25, 0.45, 0.65, 0.85, 1.0])
    tpr_weak = np.array([0, 0.22, 0.45, 0.64, 0.80, 0.93, 1.0])
    a_good, a_weak = auc(fpr_good, tpr_good), auc(fpr_weak, tpr_weak)
    fig, ax = book_figure(0.82, 2.6)
    ax.fill_between(fpr_good, tpr_good, color=NML_CYAN, alpha=0.10, linewidth=0, zorder=1)
    ax.plot([0, 1], [0, 1], color=REF, lw=LW_THIN, ls='--', zorder=2)
    ax.plot(fpr_good, tpr_good, color=NML_CYAN, zorder=3)
    ax.plot(fpr_weak, tpr_weak, color=NML_RED, zorder=3)
    ax.text(0.2, 0.99, f'good model, AUC = {num(a_good)}', color=NML_CYAN, va='bottom', ha='left')
    ax.text(0.09, 0.63, f'weak model\nAUC = {num(a_weak)}', color=NML_RED, va='center', ha='left')
    ax.text(0.97, 0.36, 'random ranking\nAUC = 0.50', color=MUTED, va='top', ha='right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    unit_ticks(ax, 'x', 0.2)
    unit_ticks(ax, 'y', 0.2)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate (recall)')
    tidy_axes(ax)
    save_figure(fig, 'ROC_AUC_curves')
    plt.close(fig)
    print(f'ROC AUC good {a_good:.4f} weak {a_weak:.4f}')


# ---------------------------------------------------------------------------
def pr():
    prevalence = 0.10
    recall_good = np.array([0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])
    prec_good = np.array([1.0, 0.95, 0.92, 0.88, 0.80, 0.65, 0.40])
    recall_weak = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    prec_weak = np.array([0.6, 0.45, 0.35, 0.25, 0.18, 0.10])
    a_good, a_weak = auc(recall_good, prec_good), auc(recall_weak, prec_weak)
    fig, ax = book_figure(0.82, 2.6)
    ax.fill_between(recall_good, prec_good, color=NML_CYAN, alpha=0.10, linewidth=0, zorder=1)
    ax.axhline(prevalence, color=REF, lw=LW_THIN, ls='--', zorder=2)
    ax.plot(recall_good, prec_good, color=NML_CYAN, zorder=3)
    ax.plot(recall_weak, prec_weak, color=NML_RED, zorder=3)
    ax.text(0.05, 1.0, f'good model, PR AUC = {num(a_good)}', color=NML_CYAN, va='bottom')
    ax.text(0.3, 0.47, f'weak model, PR AUC = {num(a_weak)}', color=NML_RED, va='bottom')
    ax.text(0.02, prevalence + 0.02, 'random ranking: precision = 0.10\n(10% of cases positive)',
            color=MUTED, ha='left', va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.06)
    unit_ticks(ax, 'x', 0.2)
    unit_ticks(ax, 'y', 0.2)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    tidy_axes(ax)
    save_figure(fig, 'PR_AUC_curves')
    plt.close(fig)
    print(f'PR AUC good {a_good:.4f} weak {a_weak:.4f}')


# ---------------------------------------------------------------------------
def _loss_curves(name, y1, y0, ylim, yticks, ylabel, mark, mark_text, text_xy):
    p = np.linspace(0.001, 0.999, 600)
    fig, ax = book_figure(0.82, 2.55)
    ax.plot(p, y1(p), color=NML_CYAN)
    ax.plot(p, y0(p), color=NML_RED)
    ax.scatter([mark[0]], [mark[1]], s=18, facecolors='white', edgecolors=NML_RED,
               linewidths=0.9, zorder=5)
    note(ax, *text_xy, mark_text, xy=mark, ha='right', va='center')
    return fig, ax, p


def brier():
    fig, ax, p = _loss_curves('Brier_Score_curves', lambda p: (1 - p) ** 2, lambda p: p ** 2,
                              1, None, None, (0.7, 0.49),
                              '70% rain forecast,\ndry day: 0.49', (0.98, 0.25))
    ax.text(0.03, 0.45, 'outcome = 1', color=NML_CYAN, va='center')
    ax.text(0.97, 0.45, 'outcome = 0', color=NML_RED, va='center', ha='right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    unit_ticks(ax, 'x', 0.2)
    unit_ticks(ax, 'y', 0.2)
    ax.set_xlabel('predicted probability $p$ of the positive class')
    ax.set_ylabel('Brier loss for one prediction')
    tidy_axes(ax)
    save_figure(fig, 'Brier_Score_curves')
    plt.close(fig)


def log_loss():
    fig, ax, p = _loss_curves('Log_Loss_curves', lambda p: -np.log(p), lambda p: -np.log(1 - p),
                              5, None, None, (0.99, -np.log(0.01)),
                              '99% spam,\nnot spam: 4.6', (0.84, 4.15))
    ax.text(0.06, 4.6, 'outcome = 1', color=NML_CYAN, va='center')
    ax.text(0.97, 0.55, 'outcome = 0', color=NML_RED, va='center', ha='right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    unit_ticks(ax, 'x', 0.2)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel('predicted probability $p$ of the positive class')
    ax.set_ylabel('log loss for one prediction')
    tidy_axes(ax)
    save_figure(fig, 'Log_Loss_curves')
    plt.close(fig)


# ---------------------------------------------------------------------------
def lens_area(d, r=1.0):
    if d >= 2 * r:
        return 0.0
    return 2 * r * r * np.arccos(d / (2 * r)) - d / 2 * np.sqrt(4 * r * r - d * d)


def jaccard_of(d):
    a = lens_area(d)
    return a / (2 * np.pi - a)


def overlap_panel(ax, d, title):
    """Two unit circles: prediction (left, cyan) and ground truth (right, purple)."""
    from matplotlib.patches import Circle
    pred = Circle((-d / 2, 0), 1, facecolor=PRED_ONLY, edgecolor='none', zorder=1)
    truth = Circle((d / 2, 0), 1, facecolor=TRUTH_ONLY, edgecolor='none', zorder=1)
    both = Circle((d / 2, 0), 1, facecolor=OVERLAP, edgecolor='none', zorder=2)
    ax.add_patch(pred)
    ax.add_patch(truth)
    ax.add_patch(both)
    both.set_clip_path(Circle((-d / 2, 0), 1, transform=ax.transData))
    ax.set_xlim(-1 - d / 2 - 0.05, 1 + d / 2 + 0.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    bare_axes(ax)
    ax.set_title(title, fontsize=TEXT_PT)


def overlap_legend(fig, labels):
    handles = [matplotlib.patches.Patch(color=c, label=l)
               for c, l in zip([OVERLAP, PRED_ONLY, TRUTH_ONLY], labels)]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, handlelength=1.0)


def jaccard():
    targets = [0.6, 0.12]
    ds = [brentq(lambda d, t=t: jaccard_of(d) - t, 0, 1.999) for t in targets]
    fig, axes = book_figure(1.0, 1.75, 1, 2)
    for ax, d, name in zip(axes, ds, ['high overlap', 'low overlap']):
        overlap_panel(ax, d, f'{name}: Jaccard = {num(jaccard_of(d))}')
    overlap_legend(fig, ['overlap (both)', 'prediction only', 'ground truth only'])
    save_figure(fig, 'Jaccard_overlap')
    plt.close(fig)
    print('Jaccard', [round(jaccard_of(d), 4) for d in ds], 'distances', [round(d, 4) for d in ds])


# ---------------------------------------------------------------------------
def d2_log_loss():
    def h(p):
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    model_ll = np.linspace(0.0, 1.0, 200)
    fig, ax = book_figure(0.82, 2.55)
    ax.plot([0, 1], [0, 0], color=REF, lw=LW_THIN, zorder=0)
    rows = [(0.5, '50/50 balanced', NML_CYAN), (0.2, '80/20 imbalanced', NML_PURPLE),
            (0.05, '95/5 heavy imbalance', NML_RED)]
    for p, label, col in rows:
        null = h(p)
        d2 = 1 - model_ll / null
        ax.plot(model_ll, d2, color=col, clip_on=True)
        ax.scatter([null], [0], s=16, facecolors='white', edgecolors=col, linewidths=0.9, zorder=5)
        text = f'{label}\nbaseline {num(null)}'
        end_y = 1 - 1.0 / null
        if end_y >= -1.5:
            ax.text(1.02, end_y, text, color=col, va='center', linespacing=1.25)
        else:
            ax.text(null * 2.5 + 0.03, -1.38, text, color=col, va='center', linespacing=1.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.0)
    unit_ticks(ax, 'x', 0.2)
    ax.set_yticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0])
    minus_axis(ax, 'y', 1)
    ax.set_xlabel('model log loss')
    ax.set_ylabel('D-squared log loss score')
    tidy_axes(ax)
    note(ax, 0.99, 0.97, 'open circles: the model\nloss equals the baseline,\nso the score is 0', ha='right', va='top')
    save_figure(fig, 'D2_Log_Loss_curve')
    plt.close(fig)


# ---------------------------------------------------------------------------
def p4():
    TP, FP, FN = 80.0, 20.0, 20.0
    spec = np.linspace(0.02, 0.99, 300)
    TN = FP * spec / (1 - spec)
    f1 = 2 * TP / (2 * TP + FP + FN)
    p4v = 4 * TP * TN / (4 * TP * TN + (TP + TN) * (FP + FN))
    fig, ax = book_figure(0.82, 2.6)
    ax.plot(spec, np.full_like(spec, f1), color=NML_RED)
    ax.plot(spec, p4v, color=NML_CYAN)
    label_end(ax, spec[-1], f1, 'F1', NML_RED, dy=-4)
    label_end(ax, spec[-1], p4v[-1], 'P4', NML_CYAN, dy=4)
    ax.scatter([0.8], [f1], s=18, facecolors='white', edgecolors=INK, linewidths=0.9, zorder=5)
    note(ax, 0.97, 0.6, 'F1 = P4 at\nspecificity = 0.80', xy=(0.805, 0.785), ha='right', va='top')
    gx = 0.3
    gp = float(np.interp(gx, spec, p4v))
    ax.annotate('', xy=(gx, f1), xytext=(gx, gp),
                arrowprops=dict(arrowstyle='<->', color=REF, lw=LW_THIN, shrinkA=1, shrinkB=1,
                                mutation_scale=6))
    note(ax, gx - 0.03, (f1 + gp) / 2 + 0.03, 'F1 does not\nchange with\ntrue negatives', ha='right', va='center')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    unit_ticks(ax, 'x', 0.2)
    unit_ticks(ax, 'y', 0.2)
    ax.set_xlabel('specificity = TN / (TN + FP)')
    ax.set_ylabel('score')
    tidy_axes(ax)
    save_figure(fig, 'P4_vs_F1')
    plt.close(fig)


# ---------------------------------------------------------------------------
def kappa():
    p = np.linspace(0.5, 0.995, 200)
    raw = p ** 2 + (1 - p) ** 2
    fig, ax = book_figure(0.82, 2.6)
    ax.plot(p, raw, color=NML_RED)
    ax.plot(p, np.zeros_like(p), color=NML_CYAN)
    label_end(ax, p[-1], raw[-1], 'raw agreement', NML_RED)
    label_end(ax, p[-1], 0, "Cohen's $\\kappa$", NML_CYAN)
    hx = 0.9
    hr = hx ** 2 + (1 - hx) ** 2
    for y, col in [(hr, NML_RED), (0, NML_CYAN)]:
        ax.scatter([hx], [y], s=18, facecolors='white', edgecolors=col, linewidths=0.9, zorder=5)
    ax.annotate('', xy=(hx, hr - 0.02), xytext=(hx, 0.02),
                arrowprops=dict(arrowstyle='<->', color=REF, lw=LW_THIN, shrinkA=1, shrinkB=1,
                                mutation_scale=6))
    note(ax, hx - 0.015, hr / 2, f'at $p$ = 0.90:\nraw = {num(hr)}\n$\\kappa$ = 0', ha='right', va='center')
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=[num(v, 1) for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    unit_ticks(ax, 'y', 0.2)
    ax.set_xlabel('majority-class proportion $p$')
    ax.set_ylabel('agreement')
    tidy_axes(ax)
    save_figure(fig, 'Cohens_Kappa_levels')
    plt.close(fig)


FIGURES = {'Balanced_Accuracy_comparison': balanced_accuracy, 'F_beta_curves': f_beta,
           'ROC_AUC_curves': roc, 'PR_AUC_curves': pr, 'Brier_Score_curves': brier,
           'Log_Loss_curves': log_loss, 'Jaccard_overlap': jaccard, 'D2_Log_Loss_curve': d2_log_loss,
           'P4_vs_F1': p4, 'Cohens_Kappa_levels': kappa}

if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
