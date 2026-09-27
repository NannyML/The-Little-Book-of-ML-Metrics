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
from matplotlib import font_manager
from matplotlib.text import Text

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
# Inter is bundled with the book so plotting does not depend on system fonts.
_INTER_DIR = Path(__file__).resolve().parent.parent / "book" / "fonts" / "Inter"
for _face in ("Regular", "Italic", "SemiBold", "SemiBoldItalic"):
    font_manager.fontManager.addfont(str(_INTER_DIR / f"Inter-{_face}.otf"))

PLOT_FONT = font_manager.FontProperties(fname=str(_INTER_DIR / "Inter-Regular.otf")).get_name()
PLOT_FONT_SETTINGS = {
    "font.family": PLOT_FONT,
    "font.size": 16,
    "font.weight": "normal",
    "mathtext.fontset": "custom",
    "mathtext.rm": PLOT_FONT,
    "mathtext.it": f"{PLOT_FONT}:italic",
    "mathtext.bf": f"{PLOT_FONT}:weight=600",
    "mathtext.sf": PLOT_FONT,
    "mathtext.fallback": "stix",
}
plt.rcParams.update(PLOT_FONT_SETTINGS)


def apply_plot_typography(fig):
    """Use Inter for plot lettering, with Semibold for existing emphasized text.

    Scientific values, labels, sizes, and layout are deliberately preserved.
    Mathematical symbols unavailable in Inter retain the STIX fallback.
    """
    for text in fig.findobj(match=Text):
        text.set_fontfamily(PLOT_FONT)
        weight = text.get_fontweight()
        emphasized = weight in ("bold", "heavy", "extra bold", "black", "semibold", "demibold")
        if emphasized or (isinstance(weight, (int, float)) and weight >= 600):
            text.set_fontweight(600)
        text.set_math_fontfamily("custom")

# Figures are sized for their actual LaTeX inclusion, not their source canvas.
# Scientific redraws have their own print checks and deliberately chosen layout.
SCIENTIFIC_PRINT_FIGURES = {'ECE_reliability', 'CBPE_estimation', 'PAPE_reweighting',
                          'DLE_nanny', 'RCD_decomposition'}


def prepare_book_figure(fig, name, width_fraction=None):
    """Set ordinary lettering to 8pt and panel titles to 8.5pt in print.

    Dense diagrams use 7.2pt lettering and 7.5pt panel titles.

    Line breaks and a few documented layout treatments change; plotted data,
    scales and scientific computations are untouched. Heatmap text contrast
    follows the rendered cell colour. BERTScore retains only its matching panel.
    """
    import re, textwrap
    name = Path(name).stem
    apply_plot_typography(fig)
    if width_fraction is None:
        width_fraction = 1.0
        for source in (FIGURES_DIR.parent).glob('*.tex'):
            pattern = r'width=([\d.]*)\\textwidth\]\{figures/' + re.escape(name) + r'\.png\}'
            match = re.search(pattern, source.read_text())
            if match:
                width_fraction = float(match.group(1) or 1)
                break
    from print_layout import arrange_for_print, dense_figure
    if name not in SCIENTIFIC_PRINT_FIGURES:
        arrange_for_print(fig, name)
    # Leave a real gap between 3D y-axis lettering and its colour bar.
    if any(hasattr(ax, 'get_zlim') for ax in fig.axes):
        for ax in fig.axes:
            if not hasattr(ax, 'get_zlim') and ax.get_position().width < .12:
                box=ax.get_position()
                ax.set_position([box.x0+.09,box.y0,box.width,box.height])
    base_pt = 7.2 if dense_figure(name) or name in SCIENTIFIC_PRINT_FIGURES else 8.0
    # The 3D metric name is already printed on its colour bar. Keeping a second
    # copy beside the z ticks creates collisions once labels are readable.
    for ax in fig.axes:
        if hasattr(ax, 'get_zlim'):
            ax.set_zlabel('')
            ax.xaxis.labelpad = 14
            ax.yaxis.labelpad = 14
            ax.tick_params(pad=2)
        legend = ax.get_legend()
        if legend and name in {'Dice_overlap_examples', 'Demographic_Parity', 'Equality_of_Opportunity'}:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend.remove()
                ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.08),ncol=1,frameon=False)
    for text in fig.findobj(match=Text):
        label = text.get_text()
        if len(label)>42 and '$' not in label:
            text.set_text('\n'.join(textwrap.fill(line,34,break_long_words=False,break_on_hyphens=False)
                                    for line in label.split('\n')))
    titles = {id(ax.title) for ax in fig.axes}
    # A nominal 7pt label can hide a 5pt mathematical subscript. Measure the
    # actual math glyph sizes and keep the smallest at least 7.1pt in print.
    from matplotlib.mathtext import MathTextParser
    math_parser = MathTextParser('path')
    math_ratios = {}
    for text in fig.findobj(match=Text):
        label = text.get_text()
        if '$' in label and text.get_visible():
            props = text.get_fontproperties().copy()
            props.set_size(10)
            glyphs = [g for line in label.split('\n')
                      for g in math_parser.parse(line, dpi=72, prop=props).glyphs]
            if glyphs: math_ratios[id(text)] = min(g[1] for g in glyphs) / 10

    print_width = PRINT_WIDTH_IN * width_fraction
    # Iterate because a tight crop includes labels and therefore changes scale.
    for _ in range(30):
        fig.canvas.draw()
        crop = fig.get_tightbbox(fig.canvas.get_renderer())
        if crop.width > 2 * fig.get_figwidth():
            raise ValueError(f'Unbounded label layout in {name}')
        scale = print_width/crop.width
        texts = [t for t in fig.findobj(match=Text) if t.get_visible() and t.get_text()]
        change = 0
        for text in texts:
            printed_target = base_pt + .5 if id(text) in titles else base_pt
            if id(text) in math_ratios:
                printed_target = max(printed_target, 7.12 / math_ratios[id(text)])
            target = printed_target / scale
            change = max(change,abs(target-text.get_fontsize()))
            text.set_fontsize(target)
        if change < .04:
            break
    fig.canvas.draw()
    scale = print_width / fig.get_tightbbox(fig.canvas.get_renderer()).width
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.collections import Collection
    for artist in fig.findobj():
        if isinstance(artist, (Line2D, Patch)):
            if artist.get_linewidth() > 0:
                artist.set_linewidth(max(artist.get_linewidth(), .8 / scale))
        elif isinstance(artist, Collection):
            widths = artist.get_linewidths()
            if len(widths): artist.set_linewidths([max(w, .8 / scale) if w else 0 for w in widths])
    fig.canvas.draw()


# ---------------------------------------------------------------------------
# Standard figure sizes
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE = (6.4 * 1.5, 4.8 * 1.5)   # (9.6, 7.2) — cross-sections, 3D
FIGSIZE_HEATMAP = (12, 6)                    # heatmaps / imshow
FIGSIZE_COMPARISON = (14, 6)                 # side-by-side subplots
FIGSIZE_LARGE = (12, 10)                     # classification 3D / 2D multi-line
FIGSIZE_SMALL = (7, 3)                       # compact inline plots (e.g. Pinball)

# ---------------------------------------------------------------------------
# Print-size rule. The approved compact layout has a 109.7 mm text area
# (4.3189 inches). A figure's lettering scales with its actual inclusion width.
# The final profiles target 8 pt ordinary / 7.2 pt dense lettering.
# BOOK_FONT/BOOK_FONT_MIN are legacy source-canvas defaults; save_figure adjusts
# them using the actual crop and inclusion width. Check with printed_pt().
# A tight crop can alter the final scale: use its saved width for precise checks.
# ---------------------------------------------------------------------------
PRINT_WIDTH_IN = 109.7 / 25.4
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
    prepare_book_figure(fig, name)
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
