"""Regression figures for MASE and wMAPE, drawn at printed size.

MASE: model errors against the naive lag-1 forecast on one simulated series.
wMAPE: the channel table, comparing wMAPE with MAPE.
(The clustering and CG figures that used to live here are drawn by
clustering_figures.py and ranking_figures.py.)

Every number printed is computed here.
Run: uv run python notebooks/dedicated_figures.py [NAME ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
from style import *

RNG = np.random.default_rng(3)


# ===========================================================================
# MASE — model errors against the naive lag-1 forecast
# ===========================================================================
def spread(vals, gap):
    """Nudge values apart so labels stacked on them don't overlap (keeps order)."""
    order = np.argsort(vals)
    out = np.array(vals, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def fig_mase():
    rng = np.random.default_rng(0)
    T = 11
    t = np.arange(T)
    level = 50 + 12 * np.sin(2 * np.pi * t / 10)
    yv = level + rng.normal(0, 3.0, T)
    naive = np.r_[np.nan, yv[:-1]]
    model = level + rng.normal(0, 2.0, T)
    e_model = np.abs(yv - model)
    e_naive = np.abs(yv - naive)
    mae_model = e_model[1:].mean()
    mae_naive = np.nanmean(e_naive)
    ratio = mae_model / mae_naive
    fig, (axT, axB) = book_figure(1.0, 2.75, 2, 1, sharex=True,
                                  gridspec_kw={'height_ratios': [1.2, 1]})
    axT.plot(t, naive, color=REF, lw=LW_THIN + 0.2, ls=(0, (3, 2)), zorder=2)
    axT.plot(t, model, color=NML_CYAN, zorder=3)
    axT.plot(t, yv, color=INK, lw=LW_THIN + 0.3, marker='o', ms=2.2, zorder=4)
    ends = spread([yv[-1], model[-1], naive[-1]], 5.0)
    for yy_, txt, col in zip(ends, ['actual', 'model forecast', "naive: yesterday's value"],
                             [INK, NML_CYAN, MUTED]):
        axT.text(T + 0.3, yy_, txt, color=col, va='center')
    axT.set_ylabel('demand')
    axT.set_yticks([40, 50, 60])
    axT.set_ylim(33, 67)
    tidy_axes(axT, bottom=False)
    w = 0.42
    axB.bar(t[1:] - w / 2, e_naive[1:], width=w, color=LIGHT, zorder=3, linewidth=0)
    axB.bar(t[1:] + w / 2, e_model[1:], width=w, color=NML_CYAN, zorder=3, linewidth=0)
    axB.plot([0, T - 1], [mae_naive] * 2, color=REF, lw=LW_THIN + 0.2, ls=(0, (3, 2)), zorder=4)
    axB.plot([0, T - 1], [mae_model] * 2, color=NML_CYAN, lw=LW_THIN + 0.2, ls=(0, (3, 2)), zorder=4)
    lab = spread([mae_model, mae_naive], 1.8)
    axB.text(T + 0.3, lab[1], f'naive MAE = {num(mae_naive)}', color=MUTED, va='center')
    axB.text(T + 0.3, lab[0], f'model MAE = {num(mae_model)}', color=NML_CYAN, va='center')
    axB.set_title(f'model MAE / naive MAE = {num(mae_model)} / {num(mae_naive)} = {num(ratio)}'
                  f'  (same {T - 1} days)', loc='left', fontsize=TEXT_PT)
    axB.set_ylabel('absolute error')
    axB.set_xlabel('day')
    axB.set_yticks([0, 5, 10])
    axB.set_ylim(0, 10)
    axB.set_xlim(-1, T)
    axB.set_xticks(t)
    tidy_axes(axB)
    save_figure(fig, 'MASE_naive_baseline')
    plt.close(fig)
    print(f'MASE model MAE {mae_model:.3f} naive MAE {mae_naive:.3f} ratio {ratio:.3f}')


# ===========================================================================
# wMAPE: MAPE per channel vs each channel's contribution to wMAPE
# ===========================================================================
def fig_wmape():
    channels = ['B2B', 'Online', 'Marketplace', 'Retail']
    actual = np.array([110_000, 90_000, 120_000, 1_000_000], float)
    forecast = np.array([99_010, 79_990, 110_040, 900_000], float)
    err = np.abs(actual - forecast)
    mape_i = 100 * err / actual
    weight = actual / actual.sum()
    contrib = 100 * err / actual.sum()
    wmape = contrib.sum()
    mape = mape_i.mean()
    fig, (axL, axR) = book_figure(1.0, 1.95, 1, 2, sharey=True, sharex=True)
    yy = np.arange(len(channels))[::-1]
    axL.barh(yy, mape_i, color=LIGHT, height=0.62, zorder=3)
    for yi, v in zip(yy, mape_i):
        axL.text(v + 0.3, yi, f'{num(v, 1)}%', va='center', color=INK)
    axR.barh(yy, contrib, color=NML_CYAN, height=0.62, zorder=3)
    for yi, v, wgt in zip(yy, contrib, weight):
        axR.text(v + 0.3, yi, f'{num(v)}  ({100 * wgt:.0f}% of revenue)', va='center', color=INK)
    axL.set_title(f'MAPE per channel\nmean {num(mape, 1)}%', loc='left', fontsize=TEXT_PT, linespacing=1.3)
    axR.set_title(f'contribution to wMAPE\nsum {num(wmape, 1)}%', loc='left', fontsize=TEXT_PT, linespacing=1.3)
    axL.set_yticks(yy, labels=channels)
    axL.set_xlabel('error / actual (%)')
    axR.set_xlabel('error / total actual (points)')
    for ax in (axL, axR):
        ax.set_xlim(0, 12.5)
        ax.set_xticks([0, 5, 10])
        tidy_axes(ax)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axR.tick_params(axis='y', labelleft=False)
    save_figure(fig, 'wMAPE_compare_MAPE')
    plt.close(fig)
    print('wMAPE', dict(mape_i=np.round(mape_i, 1), weight=np.round(100 * weight, 1),
                        contrib=np.round(contrib, 2), wmape=round(wmape, 2), mape=round(mape, 2)))


FIGURES = {'MASE_naive_baseline': fig_mase, 'wMAPE_compare_MAPE': fig_wmape}


if __name__ == '__main__':
    # Only MASE draws from RNG, so each figure can be rebuilt on its own.
    wanted = set(sys.argv[1:])
    for name, make in FIGURES.items():
        if not wanted or name in wanted:
            make()
