"""
NannyML Book Style Module

Shared plotting configuration for The Little Book of ML Metrics.
Import this module in any notebook or script to get consistent styling.

Usage:
    from style import *
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ---------------------------------------------------------------------------
# NannyML color palette
# ---------------------------------------------------------------------------
NML_CYAN = "#0AA7D4"
NML_PURPLE = "#3B0280"
NML_RED = "#DD4040"
NML_DARK_RED = "#CB0202"

# Convenience aliases matching the notebooks' original variable names
start_color = NML_CYAN
middle_color = NML_PURPLE
end_color = NML_RED
end_end_color = NML_DARK_RED

# Ordered palette for multi-line plots (good → bad semantic ordering)
PALETTE = [NML_CYAN, NML_PURPLE, NML_RED, NML_DARK_RED]
PALETTE_REVERSED = PALETTE[::-1]

# ---------------------------------------------------------------------------
# NannyML colormap
# ---------------------------------------------------------------------------
_num_colors = 10
_gradient = mcolors.LinearSegmentedColormap.from_list(
    "custom_gradient",
    [NML_CYAN, NML_PURPLE, NML_RED, NML_DARK_RED],
    N=_num_colors,
)
_gradient_colors = [
    mcolors.rgb2hex(_gradient(i / _num_colors)) for i in range(_num_colors)
]
nml_cmap = mcolors.LinearSegmentedColormap.from_list("nml_cmap", _gradient_colors)

# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
plt.rcParams["axes.labelpad"] = 15
plt.rcParams.update({"font.size": 16})

# ---------------------------------------------------------------------------
# Standard figure sizes
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE = (6.4 * 1.5, 4.8 * 1.5)   # (9.6, 7.2) — cross-sections, 3D
FIGSIZE_HEATMAP = (12, 6)                    # heatmaps / imshow
FIGSIZE_COMPARISON = (14, 6)                 # side-by-side subplots
FIGSIZE_LARGE = (12, 10)                     # classification 3D / 2D multi-line
FIGSIZE_SMALL = (7, 3)                       # compact inline plots (e.g. Pinball)

# ---------------------------------------------------------------------------
# Print-size rule (2026-09-06).  The book's text width is 3.7 in.  A figure
# included at \textwidth is scaled by 3.7 / figure_width, so its text shrinks by
# the same factor.  Rule: text on the printed page must be at least 6.5 pt, the
# size of a footnote.  With FIG_W = 8 in that means every font size >= 14 pt;
# use BOOK_FONT for labels and never go below BOOK_FONT_MIN.  Figures narrower
# than \textwidth (0.6\textwidth etc.) need proportionally larger fonts:
# printed_pt(fontsize, fig_width, frac) tells you what a size will print at.
# ---------------------------------------------------------------------------
PRINT_WIDTH_IN = 3.7
FIG_W = 8.0
BOOK_FONT = 15
BOOK_FONT_MIN = 14


def printed_pt(fontsize, fig_width_in=FIG_W, width_frac=1.0):
    """Point size a figure font prints at when the figure is included at width_frac * textwidth."""
    return fontsize * PRINT_WIDTH_IN * width_frac / fig_width_in

# Default line style
LINE_KW = dict(linewidth=6, solid_capstyle="round")

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parent.parent / "book" / "figures"


def save_figure(fig, name: str, *, dpi: int = 300):
    """Save a figure to book/figures/<name>.png with book-standard settings."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure factory helpers
# ---------------------------------------------------------------------------

def create_line_plot(figsize=FIGSIZE_SINGLE):
    """Create a figure + axes for a 2D line / cross-section plot."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_3d_surface(figsize=FIGSIZE_SINGLE):
    """Create a figure + 3D axes for a surface plot."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def create_heatmap(figsize=FIGSIZE_HEATMAP):
    """Create a figure + axes for a heatmap / imshow plot."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_comparison(ncols=2, figsize=FIGSIZE_COMPARISON):
    """Create a figure with side-by-side subplots."""
    fig, axs = plt.subplots(1, ncols, figsize=figsize)
    return fig, axs


# ---------------------------------------------------------------------------
# Colorbar helper
# ---------------------------------------------------------------------------

def add_colorbar(fig, mappable, label: str, *, pad=0.01, nbins=4, labelpad=15):
    """Add a consistently-styled colorbar to a figure."""
    cbar = fig.colorbar(mappable, pad=pad)
    cbar.ax.locator_params(nbins=nbins)
    cbar.set_label(label, labelpad=labelpad)
    return cbar


# ---------------------------------------------------------------------------
# Preview colormap (useful inside notebooks)
# ---------------------------------------------------------------------------

def show_colormap():
    """Display the NannyML colormap as a horizontal bar."""
    plt.imshow([[0, 1]], aspect="auto", cmap=nml_cmap)
    plt.gca().set_visible(False)
    plt.colorbar(cmap=nml_cmap, orientation="horizontal")
    plt.show()
