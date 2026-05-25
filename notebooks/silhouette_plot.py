"""Tufte-style silhouette plot for the Silhouette Score page."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import NML_CYAN, NML_PURPLE, NML_RED, save_figure

X_good, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
X_bad, _ = make_blobs(n_samples=300, centers=3, cluster_std=2.5, random_state=42)

km_good = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_good)
km_bad = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_bad)

colors_list = [NML_CYAN, NML_PURPLE, NML_RED]


def plot_panel(ax, X, labels, title):
    sil_values = silhouette_samples(X, labels)
    avg = silhouette_score(X, labels)
    n_clusters = len(set(labels))

    y_lower = 5
    y_center_positions = []
    cluster_labels = []
    for i in range(n_clusters):
        cluster_sil = np.sort(sil_values[labels == i])
        n_in_cluster = len(cluster_sil)
        y_upper = y_lower + n_in_cluster
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_sil,
            facecolor=colors_list[i],
            alpha=0.7,
            linewidth=0,
        )
        y_center_positions.append((y_lower + y_upper) / 2)
        cluster_labels.append(f"cluster {i}")
        y_lower = y_upper + 12

    ax.axvline(x=avg, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(
        avg,
        y_lower + 4,
        f"average = {avg:.2f}",
        color="#222222",
        fontsize=13,
        ha="center",
        va="bottom",
    )

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.set_yticks(y_center_positions)
    ax.set_yticklabels(cluster_labels, fontsize=13, color="#222222")
    ax.tick_params(axis="x", colors="#444444", labelsize=13, length=4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("silhouette coefficient $s(i)$", fontsize=14, color="#222222")
    ax.set_xlim(-0.3, 1)
    ax.set_ylim(0, y_lower + 20)
    ax.set_title(title, fontsize=15, loc="left", pad=12, color="#333333")


fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.2})

plot_panel(
    axes[0],
    X_good,
    km_good.labels_,
    f"well-separated (avg $S = {silhouette_score(X_good, km_good.labels_):.2f}$)",
)
plot_panel(
    axes[1],
    X_bad,
    km_bad.labels_,
    f"overlapping (avg $S = {silhouette_score(X_bad, km_bad.labels_):.2f}$)",
)

save_figure(fig, "Silhouette_Score_comparison")
