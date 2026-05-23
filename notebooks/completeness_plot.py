"""Tufte-style three-panel showing the completeness vs homogeneity tradeoff."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import completeness_score, homogeneity_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import NML_CYAN, NML_PURPLE, NML_RED, nml_cmap, save_figure

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)

# A: k = 3 (recovers truth)
pred_a = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X).labels_
c_a = completeness_score(y, pred_a)
h_a = homogeneity_score(y, pred_a)

# B: everything in one cluster (Completeness = 1, Homogeneity = 0)
pred_b = np.zeros(len(y), dtype=int)
c_b = completeness_score(y, pred_b)
h_b = homogeneity_score(y, pred_b)

# C: over-split (each class chopped into many pure pieces; Homogeneity = 1, Completeness drops)
pred_c = KMeans(n_clusters=9, random_state=42, n_init=10).fit(X).labels_
c_c = completeness_score(y, pred_c)
h_c = homogeneity_score(y, pred_c)

print(f"A k=3:      C={c_a:.2f}  H={h_a:.2f}")
print(f"B one cluster: C={c_b:.2f}  H={h_b:.2f}")
print(f"C k=9:      C={c_c:.2f}  H={h_c:.2f}")

palette_3 = [NML_CYAN, NML_PURPLE, NML_RED]
palette_9 = [nml_cmap(i / 8) for i in range(9)]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

panels = [
    (axes[0], pred_a, palette_3, c_a, h_a, "$k = 3$ (true count)"),
    (axes[1], pred_b, [NML_PURPLE], c_b, h_b, "one big cluster"),
    (axes[2], pred_c, palette_9, c_c, h_c, "$k = 9$ (over-split)"),
]

for ax, pred, palette, comp, hom, label in panels:
    for c in np.unique(pred):
        ax.scatter(
            X[pred == c, 0],
            X[pred == c, 1],
            color=palette[int(c) % len(palette)],
            s=26,
            alpha=0.8,
            linewidths=0,
        )
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=15, loc="center", pad=14, color="#333333")
    ax.text(
        0.5,
        -0.08,
        f"Completeness = {comp:.2f}\nHomogeneity  = {hom:.2f}",
        transform=ax.transAxes,
        fontsize=15,
        family="monospace",
        color="#111111",
        ha="center",
        va="top",
    )

plt.tight_layout()
save_figure(fig, "Completeness_Score_tradeoff")
