"""Tufte-style bar chart for Mutual Info Score figure.

Same data as the original (good vs bad clustering across 6 external metrics),
but Tufte conventions: direct value labels on each bar, no legend, condition
labels above the leftmost pair, range-frame axis, MI highlighted as the page
subject. Uses brand NannyML cyan + red.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mutual_info_score, adjusted_rand_score, fowlkes_mallows_score,
    completeness_score, homogeneity_score, v_measure_score,
)

from style import NML_CYAN, NML_RED, save_figure

X_good, y_good = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
X_bad,  y_bad  = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=42)
km_good = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_good)
km_bad  = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_bad)

metric_funcs = [
    ('MI',           mutual_info_score),
    ('ARI',          adjusted_rand_score),
    ('FMI',          fowlkes_mallows_score),
    ('Completeness', completeness_score),
    ('Homogeneity',  homogeneity_score),
    ('V-Measure',    v_measure_score),
]
names = [n for n, _ in metric_funcs]
good = np.array([f(y_good, km_good.labels_) for _, f in metric_funcs])
bad  = np.array([f(y_bad,  km_bad.labels_)  for _, f in metric_funcs])

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(names))
width = 0.36
mi_idx = names.index('MI')

ax.bar(x - width/2, good, width, color=NML_CYAN, edgecolor='none', alpha=0.9)
ax.bar(x + width/2, bad,  width, color=NML_RED,  edgecolor='none', alpha=0.9)

# Direct value labels above each bar
for xi, val in zip(x - width/2, good):
    ax.text(xi, val + 0.02, f'{val:.2f}', ha='center', va='bottom',
            fontsize=14, color=NML_CYAN)
for xi, val in zip(x + width/2, bad):
    ax.text(xi, val + 0.02, f'{val:.2f}', ha='center', va='bottom',
            fontsize=14, color=NML_RED)

# Small in-plot legend in the upper-right corner, two rows
label_y_top = max(good.max(), bad.max()) + 0.22
ax.text(x[-1] + 0.45, label_y_top,
        'well-separated', ha='right', va='center',
        fontsize=13, color=NML_CYAN, style='italic')
ax.text(x[-1] + 0.45, label_y_top - 0.07,
        'overlapping',    ha='right', va='center',
        fontsize=13, color=NML_RED,  style='italic')

# MI callout, top-left, kept short to avoid wrap
ax.text(-0.45, label_y_top,
        'only MI exceeds $1$', ha='left', va='center',
        fontsize=13, color=NML_CYAN)
ax.text(-0.45, label_y_top - 0.07,
        '(bounded by $\\log k \\approx 1.1$)', ha='left', va='center',
        fontsize=11, color=NML_CYAN, alpha=0.8)

# X-tick labels; bold MI
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=16)
for tick_label, n in zip(ax.get_xticklabels(), names):
    if n == 'MI':
        tick_label.set_color(NML_CYAN)
        tick_label.set_fontweight('semibold')

# Range frame: drop top/right, bound left to data
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#999999')
ax.spines['bottom'].set_color('#999999')
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'],
                   fontsize=12, color='#666666')
ax.tick_params(axis='both', length=3, color='#999999')
ax.spines['left'].set_bounds(0, 1.0)
ax.set_ylim(0, label_y_top + 0.08)

fig.tight_layout()
save_figure(fig, 'Clustering_external_metrics_comparison')
plt.close()
print("MI Tufte bar chart: done")
