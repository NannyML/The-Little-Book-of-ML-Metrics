# Project guidance

The Little Book of ML Metrics is a LaTeX book in its final editorial stages. Most content exists. Improve the requested material while preserving explanations, examples, historical facts, and visuals that already work. Current user instructions take precedence over older session notes or remembered templates.

## Where things live

- `book/main.tex`: preamble, fonts, page styles, custom commands, and chapter includes.
- `book/1-*.tex` through `book/11-*.tex`: introduction and subject chapters; Benchmarks is a section in `book/8-genai.tex`.
- `book/12-sources.tex`: chapter-grouped Bibliography.
- `book/figures/`: rendered figures; `notebooks/`: their generators and shared `style.py`.
- `BOOK_STATUS.md`: navigation and current work pointers, not a second feedback database.
- Supabase's **book-reviewer** project: review comments and the published book version. Inspect the actual schema and relevant rows rather than guessing table or column names.

## Scope and review workflow

1. Read the requested comments, current source, and rendered pages. Review-only requests produce feedback; applying feedback permits scoped edits. Do not rewrite whole sections just to match a template.
2. During a writing-only pass, leave plots, plot generators, displayed formulas, and formula arrows unchanged. Improve explanations without endorsing a known visual error; record any blocked correction or visual issue separately. A request to fix a specific formula or visual overrides that default for the named item.
3. Preserve a stable **original** revision/PDF before editing. A before/after comparison uses that original throughout the chapter's review, not the previous assistant draft. Record the baseline commit, page mapping, and output paths alongside the comparison.
4. After book edits, compile, render, and inspect the affected pages. Check text flow, formulas, captions, page breaks, blank pages, and clipping. Passing compilation alone is insufficient.
5. Make local commits at useful checkpoints using the configured human identity. Never add an assistant author, co-author, or signature. Keep unrelated user work out of the commit and never reset it. Push and publish when requested; do not watch GitHub CI unless asked.
6. When asked to update Supabase, resolve only comments whose requested changes were applied and verified. Leave visual comments open during writing passes. For mixed comments, create or preserve an open comment covering the unresolved visual part before resolving the original. Avoid duplicates; preserve existing feedback and attribution. Update the book version only to the requested, verified artifact, and confirm the saved result.

## Editorial skills

Six skills live in `.agents/skills/<name>/SKILL.md`, mirrored in `.claude/skills/<name>/SKILL.md`:

- **santiago-writing**: voice, teaching order, and approved/proposed examples. Owns the writing preferences.
- **book-content-upgrade**: apply chapter feedback, verify source-backed changes, and deliver a rendered comparison.
- **book-review**: one review standard for both existing content and revised drafts. Owns the checks and feedback format.
- **book-formula**: equation correctness, annotation conventions, and formula layout tools.
- **book-plot**: computed figures, NannyML styling, and verification at printed size.
- **tufte-viz**: visual comparison and graphical integrity; principles applied with judgment.

Read the relevant skill, not the entire catalog. Keep both skill directories identical when updating them. Store a rule with its owner and link to it from other skills instead of copying it. `CLAUDE.md` points here so shared project rules have one maintained source.

## Sources and bibliography

When adding or substantively revising a factual, scientific, or historical claim using a paper, standard, or documentation page, add or verify its reference in `book/12-sources.tex` in the same change. Verify source existence, support for the precise claim, authors, title, year, and a direct URL or DOI. Prefer primary sources; do not infer support from a title or search snippet.

The rendered chapter is **Bibliography**. References begin immediately, with no introductory summary, and are grouped by chapter rather than metric. Avoid duplicates. The book does not require parenthetical author-year citations in body prose; useful researcher names and dates may remain naturally in explanations or callouts. Cover sources used for the current changes; a complete bibliography audit is a separate task.

## Layout and LaTeX conventions

Most metric entries use two pages: explanation, annotated formula, interpretation and strengths/weaknesses, followed by the figure and supporting discussion. Preserve the local layout. Observability monitors use one page. Benchmarks intentionally has multiple entries per page and no figures. Chapter openers and decision maps have their own formats; do not force missing template blocks into them.

`\coloredboxes{strengths}{weaknesses}` produces the paired boxes; `\orangebox{title}{content}` produces a callout. Page styles and colors live in `book/main.tex`. Fonts are Rubik, Lato, and PlayfairDisplay from `book/fonts/`.

Use math mode for measured values, percentages, indices, and equations (`$0.6$`, `$80\%$`); use `--` for numeric ranges. Spell out small natural-language counts when appropriate. Dates, model names, and identifiers do not automatically belong in math mode. Preserve section labels and cross-references.

For benchmarks, explain influential historical contributions. Verify dates, named models, benchmark versions, and evaluation conditions when describing progress or saturation. Avoid declaring a benchmark universally "defeated" based on one score.

## Build and tools

From `book/`:

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```

The output is `book/main.pdf`. The build needs XeLaTeX and the bundled fonts. Use `uv run python ...` for project Python tools; `uv sync` installs the declared dependencies when needed.

`tools/formula_layout.py` and `tools/formula_tuner.py` help with scoped formula work; read **book-formula** before using them. New/edited plots use `notebooks/style.py`; read **book-plot**. Do not regenerate either as part of a prose-only edit.
