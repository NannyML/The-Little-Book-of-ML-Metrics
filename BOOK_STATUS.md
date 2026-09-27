# Book work pointers

All subject chapters have content. The book is in editorial review, not initial chapter construction. Presence of a formula, plot, or callout is not evidence that a page meets the quality bar.

Use the current conversation and the **book-reviewer** Supabase comments for outstanding work. Read [AGENTS.md](AGENTS.md) for the workflow. Do not infer current approval or edit restrictions from old completion counts, lock lists, or session summaries.

## Finding work

- Read the requested chapter's current feedback before choosing edits; do not treat every page as needing a comment.
- Check Git status and recent commits for local revisions and user edits. Existing before/after PDFs are under `review/` and `output/pdf/`; preserve their original baseline.
- `book/main.tex` lists chapter files. Benchmarks belongs to GenAI; the Bibliography is in `book/12-sources.tex`.
- Formula tools live in `tools/`; the formula skill documents their current use. `FORMULA_FIXES.md`, where present, is historical, not an automatic queue of authorized changes.
- The hosted reviewer is the normal feedback location. `tools/review_recorder.py` remains available for explicitly requested local recording; its notes belong in `review/`.

## Current follow-up

The GenAI chapter opener was revised on September 27, 2026 to establish the subject and purpose before introducing the comparisons. The revised text is pending Santiago's review; see page one of `output/pdf/genai-before-after.pdf`, against the original at `447b323`.

Do not maintain a second per-comment checklist here. Resolve comments in Supabase only after the corresponding correction has been applied and verified.
