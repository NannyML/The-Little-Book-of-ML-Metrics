---
name: book-review
description: Review existing or revised book chapters for scientific accuracy, clear teaching, natural writing, and rendered-page quality. Produce actionable feedback or check applied corrections without rewriting unless asked.
---

# Review the book

Use this one review standard for an existing chapter and for a revised draft. Follow [project guidance](../../../AGENTS.md) and [Santiago writing](../santiago-writing/SKILL.md). A review-only request produces comments, not manuscript edits.

Read the requested source and rendered pages. For a revised draft, read the original too: a more technical explanation can still be a worse explanation. Review the whole passage, including the context before it, rather than isolated sentences.

## Reader comprehension

- Does the opening say plainly what the metric measures, before an example, analogy, formula interpretation, or limitation?
- Does a chapter opener establish what the subject is and why it matters before listing methods? Are terms such as "reference dataset" introduced before the prose assumes the reader knows them?
- Can a reader follow what happens, what is compared, and what the result means? Check the population, denominator, aggregation level, and unit where they change the meaning. "Sum per daily partition" is clearer than an unexplained "sum."
- Does each sentence build on the previous one? Flag both overloaded prose and disconnected glossary-like statements. Active voice should improve the flow, not force repetitive "we" sentences.
- Are examples concrete and useful? Has the rewrite preserved a clearer original example? Are true historical details and names kept where interesting?
- Is precision placed where it helps: essential meaning and necessary qualifications in the opening, derivation and secondary details where the reader is ready for them? Never hide a qualification needed to make the opening true.

Use the [book calibration examples](../santiago-writing/references/book-examples.md). No list of banned words can substitute for reading and understanding the paragraph.

## Scientific accuracy

Check definitions, formula variants, averaging, ranges, units, direction, chance baselines, edge cases, and stated assumptions against primary papers or authoritative implementation documentation. Verify concrete numbers and historical claims. Distinguish predictions from outcomes, calibration from overall predictive quality, and a conditional rate from a whole-population rate.

Flag unsupported thresholds, universal superiority claims, outdated benchmark claims, and causal explanations that a score cannot establish. Name the relevant version and evaluation conditions when they matter. Do not make text harder to read merely to list every possible caveat.

For changed source-backed claims, verify the Bibliography entry and its actual support. Review scope determines the sources to audit; it does not require completing the entire book's bibliography first.

## Rendered pages and scope

Inspect formulas, labels, and captions for agreement with the text. Check clipping, overlap, readability at printed size, and unexpected page breaks. Follow the existing chapter format: observability, benchmarks, openers, and decision maps have legitimate exceptions to the usual metric layout.

When plots or formulas are outside the editing scope, record their issues separately. Reading a figure to check a caption does not authorize changing it. Use **book-plot**, **book-formula**, or **tufte-viz** only for the relevant technical assessment; a legend or non-range-frame axis is not automatically a defect.

## Useful comments

Each comment should identify a location or quoted passage, the specific problem, why it matters to the reader, and a feasible direction for correction. Include a verified source for a scientific correction. Distinguish confirmed errors from questions needing verification. Avoid duplicate comments and arbitrary quotas; a page with no meaningful issue needs no comment. Priority labels are unnecessary unless requested.

Example:

> Equality of opportunity, opening: "80% of men and women" leaves the denominator unclear and can sound like demographic parity. State that we compare only applicants who would repay the loan, then give the approval rate within each of those groups. Keep the conditional distinction without adding a catalog of fairness definitions here.

For a technically dense comment, add an implementation note when helpful: **Explain the essential correction simply; do not pack every qualification in this comment into the book's paragraph.**

When asked to save feedback in Supabase, follow **AGENTS.md**, inspect the live schema, attach it to the correct chapter/page/version, and verify the saved rows. Keep writing and visual issues separate when feasible. Existing good content is not a reason to invent a comment.

## Report honestly

For revision checks, state what improved relative to the original, any remaining actionable issues, and what was not verified. Passing a formatting checklist is not evidence of a good explanation. Do not declare a chapter perfect, user-approved, or fully fact-checked beyond the work actually performed.
