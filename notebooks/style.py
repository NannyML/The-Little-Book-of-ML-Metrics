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


def save_figure(fig, name: str, *, dpi: int = 300, crop: bool = True):
    """Save a figure to book/figures/<name>.png with book-standard settings."""
    if getattr(fig, '_book_print', None):
        return save_print_figure(fig, name, crop=crop)
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


# ===========================================================================
# Print-size design
#
# Book figures are drawn at the size they print, with their final type sizes,
# so the layout checked on screen is the layout on the page. A figure made with
# book_figure() is saved by save_figure() without any rescaling, wrapping or
# per-figure layout patches.
#
# House conventions (see .claude/skills/book-plot/SKILL.md):
#   - lettering 8 pt (panel titles 8.5 pt); in-cell numbers of dense matrices
#     and token diagrams 7 pt; nothing smaller.
#   - labels in lower case, except acronyms, symbols and metric names written
#     that way in the prose ("nDCG", "SNR", "Y").
#   - left and bottom spines only, offset slightly so tick labels never meet at
#     the origin; no gridlines; direct labels instead of legends when they fit.
#   - 0-1 score axes print one decimal (0.0, 0.5, 1.0); other axes use whole
#     "nice" steps with thousands separators.
#   - negative numbers use the true minus sign; exact zeros carry no sign.
#   - data lines 1.4 pt, reference lines 0.7 pt gray, markers 3.5 pt.
#   - color meaning: cyan = good / correct / matched, red = bad / wrong,
#     purple = second series or middle, grays = reference and context.
# ===========================================================================
from matplotlib import ticker as _ticker

TEXTWIDTH_IN = PRINT_WIDTH_IN          # 109.7 mm
TEXT_PT = 8.0
TITLE_PT = 8.5
SMALL_PT = 7.0
LW = 1.4
LW_THIN = 0.7
MS = 3.5

INK = '#2a2a2a'        # data drawn in neutral ink, emphasized labels
MUTED = '#6f6f6f'      # annotations and secondary text
REF = '#9a9a9a'        # reference lines, leaders, neutral markers
LIGHT = '#d6d6d6'      # neutral cells and empty markers
FILL = '#efefef'       # light context shading
CYAN_TINT = '#e2f3f9'
RED_TINT = '#fbe9e9'
PURPLE_TINT = '#ebe5f2'

PRINT_RC = {
    **PLOT_FONT_SETTINGS,
    'font.size': TEXT_PT,
    'axes.titlesize': TITLE_PT,
    'axes.titleweight': 'normal',
    'axes.titlepad': 4,
    'axes.labelsize': TEXT_PT,
    'axes.labelpad': 3,
    'axes.labelcolor': 'black',
    'axes.linewidth': 0.6,
    'axes.edgecolor': 'black',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.unicode_minus': True,
    'xtick.labelsize': TEXT_PT,
    'ytick.labelsize': TEXT_PT,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.pad': 2,
    'ytick.major.pad': 2,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'lines.linewidth': LW,
    'lines.markersize': MS,
    'lines.solid_capstyle': 'round',
    'patch.linewidth': 0.6,
    'legend.fontsize': TEXT_PT,
    'legend.frameon': False,
    'legend.handlelength': 1.4,
    'legend.handletextpad': 0.4,
    'legend.borderaxespad': 0.2,
    'legend.labelspacing': 0.3,
    'legend.columnspacing': 1.2,
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'figure.constrained_layout.h_pad': 0.02,
    'figure.constrained_layout.w_pad': 0.02,
    'figure.constrained_layout.hspace': 0.04,
    'figure.constrained_layout.wspace': 0.04,
}


def use_print_style():
    """Switch matplotlib to the print-size defaults (call once per generator)."""
    plt.rcParams.update(PRINT_RC)


def book_figure(width=1.0, height=2.4, nrows=1, ncols=1, *, layout='constrained',
                subplot_kw=None, gridspec_kw=None, **kwargs):
    """Figure at its printed size: width is the fraction of the text width used
    in the \\includegraphics call, height is in inches."""
    use_print_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXTWIDTH_IN * width, height),
                             layout=layout, subplot_kw=subplot_kw,
                             gridspec_kw=gridspec_kw, **kwargs)
    fig._book_print = {'width': width}
    return fig, axes


def book_canvas(width=1.0, height=2.4, layout=None):
    """Empty print-size figure for hand-placed axes (3D surfaces, diagrams)."""
    use_print_style()
    fig = plt.figure(figsize=(TEXTWIDTH_IN * width, height), layout=layout)
    fig._book_print = {'width': width}
    return fig


def tidy_axes(ax, offset=3, left=True, bottom=True):
    """Open left/bottom spines, offset from the data so zero labels never meet."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    if left:
        ax.spines['left'].set_position(('outward', offset))
    if bottom:
        ax.spines['bottom'].set_position(('outward', offset))
    if not left:
        ax.tick_params(axis='y', left=False, labelleft=False)
    if not bottom:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    return ax


def bare_axes(ax):
    """No spines or ticks: for diagrams, images and token figures."""
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def num(x, nd=2, sign=False, pct=False, thousands=False):
    """Format a number for figure text: true minus sign, no sign on zero."""
    value = round(float(x), nd)
    if value == 0:
        value = 0.0
    body = f'{abs(value):,.{nd}f}' if thousands else f'{abs(value):.{nd}f}'
    if value < 0:
        body = '−' + body
    elif sign and value > 0:
        body = '+' + body
    return body + ('%' if pct else '')


def unit_ticks(ax, axis='y', step=0.5, lo=0.0, hi=1.0):
    """0-1 score axis printed with one decimal (0.0, 0.5, 1.0)."""
    ticks = np.round(np.arange(lo, hi + step / 2, step), 6)
    labels = [num(t, 1) for t in ticks]
    if axis == 'y':
        ax.set_yticks(ticks, labels=labels)
    else:
        ax.set_xticks(ticks, labels=labels)
    return ticks


def thousands_axis(ax, axis='y'):
    fmt = _ticker.FuncFormatter(lambda v, _: num(v, 0, thousands=True))
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(fmt)


def minus_axis(ax, axis='y', nd=None):
    """Tick labels with the true minus sign and an explicit number of decimals."""
    def f(v, _):
        d = nd if nd is not None else (0 if float(v).is_integer() else 1)
        return num(v, d)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(_ticker.FuncFormatter(f))


def label_end(ax, x, y, text, color, dx=3, dy=0, ha='left', va='center', **kw):
    """Direct label next to a line end, offset in points."""
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points',
                       ha=ha, va=va, color=color, annotation_clip=False, **kw)


def note(ax, x, y, text, xy=None, color=None, ha='left', va='center', **kw):
    """Gray annotation, optionally with a thin leader to xy (data coords)."""
    color = color or MUTED
    if xy is None:
        return ax.text(x, y, text, color=color, ha=ha, va=va, **kw)
    return ax.annotate(text, xy=xy, xytext=(x, y), color=color, ha=ha, va=va,
                       arrowprops=dict(arrowstyle='-', color=REF, lw=LW_THIN,
                                       shrinkA=1.5, shrinkB=2),
                       annotation_clip=False, **kw)


def _flatten(path):
    from PIL import Image
    with Image.open(path) as im:
        if im.mode == 'RGB':
            return
        rgba = im.convert('RGBA')
        flat = Image.new('RGB', rgba.size, 'white')
        flat.paste(rgba, mask=rgba.getchannel('A'))
        flat.save(path)


def _check_print_text(fig, name):
    problems = []
    for text in fig.findobj(match=Text):
        label = text.get_text()
        if not text.get_visible() or not label.strip():
            continue
        if text.get_fontsize() < SMALL_PT - 0.01:
            problems.append(f'{label[:30]!r} at {text.get_fontsize():.1f} pt')
    if problems:
        raise ValueError(f'{name}: lettering below {SMALL_PT} pt: {problems}')


def save_print_figure(fig, name, *, crop=True, dpi=600):
    """Save a print-size figure. With crop=True the canvas width is adjusted so
    the tight crop equals the printed width exactly (no scaling at inclusion)."""
    name = Path(name).stem
    apply_plot_typography(fig)
    target = TEXTWIDTH_IN * fig._book_print['width']
    if crop:
        fig.set_dpi(dpi)          # measure text at the resolution it is saved at
        for _ in range(8):
            fig.canvas.draw()
            box = fig.get_tightbbox(fig.canvas.get_renderer())
            error = target - box.width
            if abs(error) < 0.002:
                break
            if fig.get_figwidth() + error < 0.5 * target:
                raise ValueError(f'{name}: content is {box.width:.2f} in wide, cannot fit {target:.2f} in')
            fig.set_figwidth(fig.get_figwidth() + error)
        fig.canvas.draw()
        # Freeze the final layout so savefig renders exactly what was measured.
        fig.set_layout_engine('none')
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        if box.width > target + 0.01:
            raise ValueError(f'{name}: crop width {box.width:.3f} in, expected {target:.3f} in')
        # Fixed-aspect content cannot grow with the canvas: center it on a
        # white strip of exactly the printed width instead.
        from matplotlib.transforms import Bbox
        pad = max(0.0, (target - box.width) / 2)
        box = Bbox([[box.x0 - pad, box.y0 - 0.01], [box.x1 + pad, box.y1 + 0.01]])
    _check_print_text(fig, name)
    path = FIGURES_DIR / f'{name}.png'
    if crop:
        fig.savefig(path, dpi=dpi, bbox_inches=box, pad_inches=0)
    else:
        fig.savefig(path, dpi=dpi)
    _flatten(path)
    print(f'Saved: {path} ({fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in canvas)')

# Set-overlap figures (Jaccard, Dice): the prediction is cyan and the ground
# truth purple, as in the formulas; their overlap is the blend of the two.
PRED_ONLY = '#8fd3ea'
TRUTH_ONLY = '#b3a0d0'
OVERLAP = '#2254aa'
