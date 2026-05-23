"""Tufte-style scatter showing why ARI exists: random labels give surprisingly high RI."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import NML_CYAN, NML_PURPLE, NML_RED, save_figure

colors = [NML_CYAN, NML_PURPLE, NML_RED]

# Well-separated clusters: k-means recovers them
X_sep, y_sep = make_blobs(
    n_samples=300, centers=3, cluster_std=0.6, random_state=42
)
pred_sep = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_sep).labels_
ri_sep = rand_score(y_sep, pred_sep)
ari_sep = adjusted_rand_score(y_sep, pred_sep)

# Overlapping clusters: k-means partial success
X_over, y_over = make_blobs(
    n_samples=300, centers=3, cluster_std=3.8, random_state=42
)
pred_over = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_over).labels_
ri_over = rand_score(y_over, pred_over)
ari_over = adjusted_rand_score(y_over, pred_over)

# Random labels on the same easy data: shows how RI inflates by chance
rng = np.random.default_rng(7)
pred_rand = rng.integers(0, 3, size=len(y_sep))
ri_rand = rand_score(y_sep, pred_rand)
ari_rand = adjusted_rand_score(y_sep, pred_rand)

print(f"well-separated:  RI={ri_sep:.3f}  ARI={ari_sep:.3f}")
print(f"overlapping:     RI={ri_over:.3f}  ARI={ari_over:.3f}")
print(f"random labels:   RI={ri_rand:.3f}  ARI={ari_rand:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

panels = [
    (axes[0], X_sep, pred_sep, ri_sep, ari_sep, "well-separated"),
    (axes[1], X_over, pred_over, ri_over, ari_over, "overlapping"),
    (axes[2], X_sep, pred_rand, ri_rand, ari_rand, "random labels"),
]

for ax, X, pred, ri, ari, label in panels:
    for c in range(3):
        ax.scatter(
            X[pred == c, 0],
            X[pred == c, 1],
            c=colors[c],
            s=28,
            alpha=0.75,
            linewidths=0,
        )
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=16, loc="left", pad=10, color="#333333")
    ax.text(
        0.03,
        0.97,
        f"RI  = {ri:.2f}\nARI = {ari:+.2f}",
        transform=ax.transAxes,
        fontsize=17,
        family="monospace",
        verticalalignment="top",
        color="#111111",
    )

plt.tight_layout()
save_figure(fig, "Rand_Index_separation")
