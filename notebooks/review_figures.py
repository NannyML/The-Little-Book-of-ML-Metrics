"""MDA, Pinball Loss, Explained Variance and Confusion Matrix figures.

Drawn at printed size (style.book_figure); every number printed on a figure is
computed here. Run: uv run python notebooks/review_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
from style import *  # noqa: F401,F403
from matplotlib.patches import Patch, Rectangle


# ===========================================================================
# MDA: direction right or wrong, day by day
# ===========================================================================
def fig_mda():
    # Eight days of a price and the forecast for each day, made the day before.
    actual = np.array([100.0, 100.6, 98.8, 97.9, 100.6, 100.8, 98.9, 97.7, 98.7])
    forecast_move = np.array([0.7, -0.6, -1.6, 2.4, -0.2, 4.6, -2.6, 0.3])
    forecast = actual[:-1] + forecast_move
    a_move = np.diff(actual)                  # y_t - y_(t-1)
    f_move = forecast - actual[:-1]           # forecast_t - y_(t-1)
    hit = np.sign(a_move) == np.sign(f_move)
    days = np.arange(1, len(a_move) + 1)
    fig, ax = book_figure(1.0, 2.2)
    w = 0.34
    ax.bar(days - w / 2 - 0.02, a_move, w, color=MUTED, linewidth=0)
    ax.bar(days + w / 2 + 0.02, f_move, w, color=np.where(hit, NML_CYAN, NML_RED), linewidth=0)
    ax.axhline(0, color=INK, lw=LW_THIN, zorder=3)
    big, small = days[~hit][np.argmax(np.abs(f_move[~hit]))], days[~hit][np.argmin(np.abs(a_move[~hit]))]
    ax.text(big + 0.55, f_move[big - 1] - 0.1, f'day {small} and day {big}\ncount one miss each',
            va='top', color=MUTED)
    ax.set_xticks(days)
    ax.set_xlim(0.4, days[-1] + 0.6)
    ax.set_yticks([-2, 0, 2, 4])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: num(v, 0, sign=True)))
    ax.set_xlabel('day')
    ax.set_ylabel('change from the\nprevious day\'s price')
    ax.set_title(f'MDA = {hit.sum()}/{len(hit)} = {num(hit.mean())}', loc='left', fontsize=TEXT_PT)
    tidy_axes(ax)
    ax.tick_params(axis='x', length=0)
    fig.legend(handles=[Patch(color=MUTED, label='actual move'),
                        Patch(color=NML_CYAN, label='forecast: same direction'),
                        Patch(color=NML_RED, label='forecast: opposite')],
               loc='outside lower center', ncol=3, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    save_figure(fig, 'MDA_direction')
    plt.close(fig)
    print(f'MDA {hit.sum()}/{len(hit)} = {hit.mean():.3f}; moves {np.round(a_move, 2).tolist()}; '
          f'misses on days {days[~hit].tolist()}')


# ===========================================================================
# Pinball loss: the tilted V, three quantiles
# ===========================================================================
def fig_pinball():
    u = np.linspace(-10, 10, 401)
    fig, ax = book_figure(1.0, 2.45)
    ax.axvline(0, color=REF, lw=LW_THIN, zorder=1)
    for q, col in [(0.1, NML_CYAN), (0.5, NML_PURPLE), (0.9, NML_RED)]:
        loss = np.where(u >= 0, q * u, (q - 1) * u)
        ax.plot(u, loss, color=col, zorder=3)
        label_end(ax, 10, q * 10, f'$q$ = {q}', col)
    note(ax, -5, 9.9, 'over-prediction ($\\hat{Y} > Y$):\nloss grows by $1 - q$ per unit', ha='center', va='top')
    note(ax, 5, 9.9, 'under-prediction ($Y > \\hat{Y}$):\nloss grows by $q$ per unit', ha='center', va='top')
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(0, 10)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel('residual $Y - \\hat{Y}$')
    ax.set_ylabel('loss')
    tidy_axes(ax)
    save_figure(fig, 'Pinball_Loss')
    plt.close(fig)


# ===========================================================================
# Explained variance vs R^2: three models on the same targets
# ===========================================================================
def fig_evs():
    rng = np.random.default_rng(3)
    n = 80
    x = rng.uniform(0, 10, n)
    y = 2 * x + rng.normal(0, 1.5, n)
    models = {
        'good': y + rng.normal(0, 1.2, n),
        'biased: shifted up': y + 6 + rng.normal(0, 1.2, n),
        'noisy': y + rng.normal(0, 5.0, n),
    }

    def r2(y, yh):
        return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def evs(y, yh):
        return 1 - np.var(y - yh) / np.var(y)

    fig, axes = book_figure(1.0, 2.15, 1, 3, sharey=True)
    for ax, (name, yh), col in zip(axes, models.items(), [NML_CYAN, NML_RED, NML_PURPLE]):
        e, r = evs(y, yh), r2(y, yh)
        ax.plot((-15, 30), (-15, 30), color=REF, lw=LW_THIN, zorder=1)
        ax.scatter(y, yh, s=5, color=col, alpha=0.9, zorder=3, linewidths=0)
        ax.set_title(f'{name}\nEVS = {num(e)}   $R^2$ = {num(r)}', loc='left', fontsize=TEXT_PT,
                     color=INK, linespacing=1.35)
        ax.title.set_color(INK)
        ax.set_xlim(-2, 25)
        ax.set_ylim(-15, 31)
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([-10, 0, 10, 20, 30])
        ax.set_xlabel('actual')
        tidy_axes(ax)
        minus_axis(ax, 'y', 0)
        print(f'EVS {name}: EVS {e:.3f} R2 {r:.3f}')
    axes[0].set_ylabel('predicted')
    bias = (models['biased: shifted up'] - y).mean()
    axes[1].annotate('', xy=(22.6, 22.6), xytext=(22.6, 22.6 + bias),
                     arrowprops=dict(arrowstyle='<->', color=NML_RED, lw=LW_THIN + 0.1,
                                     shrinkA=0, shrinkB=0, mutation_scale=6))
    axes[1].text(23.2, 22.6 + bias / 2, f'+{num(bias, 1)}', color=NML_RED, va='center')
    save_figure(fig, 'EVS_vs_R2_three_models')
    plt.close(fig)
    print('EVS bias', round(bias, 2))


# ===========================================================================
# Confusion matrix: ten predictions become four cells, and the cells become the metrics
# ===========================================================================
def fig_confusion():
    actual = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0])
    kind = np.where(actual == 1, np.where(pred == 1, 'TP', 'FN'), np.where(pred == 1, 'FP', 'TN'))
    tp, fn, fp, tn = [(kind == k).sum() for k in ('TP', 'FN', 'FP', 'TN')]
    n = len(actual)
    fig = book_canvas(1.0, 2.35)
    axL = fig.add_axes([0.0, 0.0, 0.35, 1.0])
    axR = fig.add_axes([0.37, 0.0, 0.63, 1.0])
    for ax in (axL, axR):
        bare_axes(ax)
    # left: the ten predictions, one row each
    axL.set_xlim(0, 3.5)
    axL.set_ylim(-0.6, n + 0.9)
    for x, txt in zip([0.5, 1.75, 2.95], ['actual', 'predicted', 'cell']):
        axL.text(x, n + 0.3, txt, ha='center', va='center', color=MUTED)
    for i in range(n):
        yy = n - 1 - i
        ok = actual[i] == pred[i]
        axL.add_patch(Rectangle((0.02, yy - 0.42), 3.46, 0.84,
                                facecolor=CYAN_TINT if ok else RED_TINT, edgecolor='none'))
        axL.text(0.5, yy, str(actual[i]), ha='center', va='center', color=INK)
        axL.text(1.75, yy, str(pred[i]), ha='center', va='center', color=INK)
        axL.text(2.95, yy, kind[i], ha='center', va='center', color=INK)
    # right: the matrix and the four reads of it
    axR.set_xlim(-1.0, 4.45)
    axR.set_ylim(-1.55, 2.75)
    cells = {(0, 1): ('TP', tp, True), (1, 1): ('FN', fn, False),
             (0, 0): ('FP', fp, False), (1, 0): ('TN', tn, True)}
    for (cx, cy), (name, cnt, ok) in cells.items():
        axR.add_patch(Rectangle((cx + 0.03, cy + 0.03), 0.94, 0.94,
                                facecolor=CYAN_TINT if ok else RED_TINT, edgecolor='none'))
        axR.text(cx + 0.5, cy + 0.6, str(cnt), ha='center', va='center', fontsize=11, color=INK)
        axR.text(cx + 0.5, cy + 0.24, name, ha='center', va='center', color=MUTED)
    # the diagonal holds the correct predictions: accuracy reads it
    axR.add_patch(Rectangle((0.0, 1.0), 1.0, 1.0, fill=False, edgecolor=INK, lw=0.7, ls=(0, (2, 1.5))))
    axR.add_patch(Rectangle((1.0, 0.0), 1.0, 1.0, fill=False, edgecolor=INK, lw=0.7, ls=(0, (2, 1.5))))
    axR.text(1.0, 2.5, 'predicted', ha='center', va='center', color=INK)
    axR.text(0.5, 2.17, '1', ha='center', va='center', color=INK)
    axR.text(1.5, 2.17, '0', ha='center', va='center', color=INK)
    axR.text(-0.1, 1.5, 'actual 1', ha='right', va='center', color=INK)
    axR.text(-0.1, 0.5, 'actual 0', ha='right', va='center', color=INK)
    axR.plot([2.12, 2.12], [1.06, 1.94], color=MUTED, lw=LW_THIN)
    axR.text(2.22, 1.5, f'recall = TP / (TP + FN)\n= {tp}/{tp + fn} = {num(tp / (tp + fn))}',
             ha='left', va='center', color=INK, linespacing=1.3)
    axR.plot([2.12, 2.12], [0.06, 0.94], color=MUTED, lw=LW_THIN)
    axR.text(2.22, 0.5, f'FPR = FP / (FP + TN)\n= {fp}/{fp + tn} = {num(fp / (fp + tn))}',
             ha='left', va='center', color=INK, linespacing=1.3)
    axR.plot([0.06, 0.94], [-0.1, -0.1], color=MUTED, lw=LW_THIN)
    axR.text(-0.98, -0.22, f'precision = TP / (TP + FP)\n= {tp}/{tp + fp} = {num(tp / (tp + fp))}',
             ha='left', va='top', color=INK, linespacing=1.3)
    axR.annotate(f'accuracy = (TP + TN) / {n}\n= {num((tp + tn) / n)}', xy=(2.0, 0.08),
                 xytext=(2.22, -0.22), ha='left', va='top', color=INK, linespacing=1.3,
                 arrowprops=dict(arrowstyle='-', color=MUTED, lw=LW_THIN, shrinkA=0, shrinkB=1))
    save_figure(fig, 'Confusion_Matrix_cells')
    plt.close(fig)
    print(f'Confusion: TP {tp} FN {fn} FP {fp} TN {tn}')


if __name__ == '__main__':
    wanted = set(sys.argv[1:])
    for name, f in [('MDA_direction', fig_mda), ('Pinball_Loss', fig_pinball),
                    ('EVS_vs_R2_three_models', fig_evs), ('Confusion_Matrix_cells', fig_confusion)]:
        if not wanted or name in wanted:
            f()
