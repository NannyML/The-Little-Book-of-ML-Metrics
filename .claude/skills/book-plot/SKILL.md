---
name: book-plot
description: Create or improve authorized book figures using computed data, NannyML styling, and verification in the compiled PDF. Use for plot work, not prose-only feedback.
---

# Book figures

Read [AGENTS.md](../../../AGENTS.md) for scope and [tufte-viz](../tufte-viz/SKILL.md) for graphical judgment. During writing-only work, report visual defects separately and leave plots and generators unchanged.

## Decide what needs improvement

Inspect the existing figure in the compiled PDF and identify the comparison or mechanism the reader needs to understand. Preserve a useful plot. A chart with a few bars is not defective because it has little ink, and a continuous sweep is not inherently better. Explain what a redesign helps the reader see before making it.

Keep the book's established 3D surfaces and cross-sections unless a specific redesign is authorized. Assess them for honest scales and readable labels; the brand does not excuse a misleading plot. Benchmarks intentionally has no figures.

## Generate from data

Use `notebooks/style.py` for the current palette, helpers, and save behavior. Import from it in the generator; inspect the module rather than relying on copied constants. Helpers include `create_line_plot`, `create_3d_surface`, `create_heatmap`, `create_comparison`, `add_colorbar`, and `save_figure`.

Compute plotted scores, counts, annotations, and caption values from the same underlying data. Format those results into labels. Use a reproducible seed for simulated examples. Check units and numerical types; cast image arrays to floating point before subtracting or squaring to avoid integer overflow.

Save descriptive PNG names under `book/figures/` using `save_figure(fig, name)` (300 dpi by default). Preserve reproducible generation code. Inspect both the generated image and its book rendering.

## Style at printed size

- Use the NannyML palette consistently; make color meanings explicit. Use a reversed colormap only when it matches the intended score direction, not because every low value is inherently bad.
- Prefer direct labels when they remain legible; keep a legend when it makes identification easier. Place annotations near their data without covering it.
- Give panels enough separation for labels and comparisons. Use common scales where direct comparison requires them. Bar lengths need an honest baseline; range frames are optional for suitable line/scatter plots.
- Keep reference lines visible but subordinate. Adjust stroke widths to the actual rendering; a fixed linewidth is not a quality test.
- Check text at the final inclusion width with `printed_pt(fontsize, fig_width_in, width_frac)` from `style.py`. The project's 6.5 pt printed minimum is a floor, not a readability target. Increase size or simplify the layout when labels strain the reader.
- Prefer regular-weight labels and a clear hierarchy. Do not remove explanatory text merely to increase data density.

## Choosing a visual explanation

Use a plot form suited to the question: a continuous sweep for behavior across a parameter; overlap geometry for set overlap; a threshold versus a smooth curve for how decisions differ; or small multiples holding one score fixed while another varies. These are options, not requirements. Existing generators such as `notebooks/d2_log_loss_plot.py` and `notebooks/cv_plots.py` provide examples, not automatic replacements for current figures.

## Verify and deliver

Check displayed values against the computed data, axes and scales, caption agreement, color interpretation, clipping, and legibility in the compiled PDF. Compare with the original at the same scale. If a tool cannot display the image, use an available rendering path or report the verification gap; do not claim visual inspection based on source code alone.

Report the actual improvement and any limitation. Following a Tufte checklist does not by itself demonstrate that a figure is clearer or more attractive.
