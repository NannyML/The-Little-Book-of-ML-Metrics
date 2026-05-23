"""Tufte-style elbow plot for the CH Index."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import NML_CYAN, NML_PURPLE, save_figure

X, _ = make_blobs(
    n_samples=600, centers=4, cluster_std=0.8, random_state=42
)

ks = np.arange(2, 10)
scores = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    scores.append(calinski_harabasz_score(X, km.labels_))

scores = np.array(scores)
best_idx = int(np.argmax(scores))
best_k = int(ks[best_idx])

print(f"CH per k: {dict(zip(ks.tolist(), scores.round(1).tolist()))}")
print(f"best k = {best_k}, CH = {scores[best_idx]:.1f}")

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.plot(
    ks,
    scores,
    color=NML_CYAN,
    linewidth=3.5,
    marker="o",
    markersize=9,
    markerfacecolor=NML_CYAN,
    markeredgecolor="white",
    markeredgewidth=1.5,
    solid_capstyle="round",
    clip_on=False,
)

ax.scatter(
    [best_k],
    [scores[best_idx]],
    s=180,
    facecolor="white",
    edgecolor=NML_PURPLE,
    linewidth=2.5,
    zorder=5,
    clip_on=False,
)
ax.annotate(
    f"peak at k = {best_k}",
    xy=(best_k, scores[best_idx]),
    xytext=(best_k + 0.6, scores[best_idx] - 50),
    fontsize=16,
    color=NML_PURPLE,
    arrowprops=dict(arrowstyle="-", color=NML_PURPLE, linewidth=1.2),
)

for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#666666")
ax.spines["bottom"].set_color("#666666")
ax.tick_params(axis="both", colors="#444444", labelsize=14, length=4)
ax.set_xlabel("number of clusters $k$", fontsize=15, color="#222222")
ax.set_ylabel("CH index", fontsize=15, color="#222222")
ax.set_xticks(ks)
ax.set_ylim(0, scores.max() * 1.1)

plt.tight_layout()
save_figure(fig, "Calinski_Harabasz_elbow")
