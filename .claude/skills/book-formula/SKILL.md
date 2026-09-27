---
name: book-formula
description: Add or correct annotated LaTeX equations and tune their rendered layout in The Little Book of ML Metrics. Use only when formula work is in scope.
---

# Book formulas

Preserve existing formulas and manual arrow adjustments during writing-only work. Before an authorized formula edit, inspect the current source and Git diff. Verify the mathematical definition, variant, indices, units, and assumptions against its source; follow [AGENTS.md](../../../AGENTS.md) for the Bibliography.

## Conventions

Use the existing `nml*` colors defined in `book/main.tex`, not copied hex values: `nmlcyan` for actual values, `nmlpurple` for predictions, `nmlred` for counts/scaling, and `nmlgreen`/`nmlyellow` for special functions and parameters. Follow established semantics within the metric; labels must explain what each term actually means. Formula cyan and plotting cyan have different current definitions; do not silently normalize them.

Use `\displaystyle`, the existing TikZ node structure, and outward arrow annotations rather than introducing underbraces. Recent formulas use `\draw[overlay,-latex,...]` plus an invisible `\path[room for labels]` to reserve space. Copy the nearest suitable current formula and tune it; no single set of coordinates works for every equation.

For example, the Perplexity block in `book/8-genai.tex` shows one node with reserved label space; MAUVE shows a multi-node layout. A plain equation can be appropriate where the chapter already uses one. Preserve section labels and the local page style. This skill does not prescribe prose length or a full metric template.

## Tools

From the repository root, preview a single formula first:

```bash
uv run python tools/formula_layout.py 8-genai.tex --only "Perplexity"
```

The tool produces a comparison without changing the chapter. Inspect it before adding `--write`. `--dump` helps inspect symbol positions; `--skip` excludes selected sections. `tools/formula_tuner.py` supports manual adjustment.

Do not run a blanket layout pass over tuned formulas. Historical automatic-layout failures include RCD, Expected Range, Row Count, Sum, and Standard Deviation; check current user changes before touching those or other hand-tuned sections.

## Verify visually

Compile and render the affected pages. Check each arrow's term, direction, label, margins, and collision with surrounding prose. Automated symbol matching can confuse similar colors, target another line, or fail to reserve enough room. Check the equation itself independently of the layout tool.

Preserve the original-versus-revised evidence. Do not describe a formula as verified merely because LaTeX compiled or the layout tool exited successfully.
