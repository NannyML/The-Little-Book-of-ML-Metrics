---
name: book-content-upgrade
description: Apply scoped feedback to The Little Book of ML Metrics, verify revised claims, inspect the compiled PDF, and prepare an original-versus-revised comparison. Use for book edits, not review-only requests.
---

# Apply book feedback

This skill owns the editing workflow. [Santiago writing](../santiago-writing/SKILL.md) owns prose style and teaching order; [book-review](../book-review/SKILL.md) owns the quality check. Follow the shared [project guidance](../../../AGENTS.md), including scope, Git, bibliography, and Supabase rules.

## Establish the change

Read the feedback and the whole affected explanation, including the existing figure and caption. Identify what the reader cannot understand or what is factually wrong. Preserve clear original examples and wording where possible. An accurate paragraph does not need rewriting merely because it starts with a definition or lacks a surprising fact.

Confirm the original comparison baseline from prior review artifacts or Git before editing. Keep it stable across revisions. Work from the current files and protect unrelated user edits.

When interpreting dense review comments, identify the correction the reader needs. A technical review may discuss several caveats; that does not mean all of them belong in the opening paragraph.

## Revise and verify

- Start metric openings with plain meaning; introduce an example after the reader knows what it illustrates. Chapter openers explain the subject and purpose before naming its methods. Read the writing skill's [book examples](../santiago-writing/references/book-examples.md) when drafting either.
- Check source-backed changes against primary papers, standards, or official documentation. Add or verify the corresponding Bibliography entries in the same change. Check calculated examples using the stated definitions and populations.
- Explain range, direction, baseline, or averaging when relevant. Do not invent universal "good" cutoffs, force a fixed number of use cases, or append an alternative to every recommendation.
- Keep useful researcher names, origins, and historical context. A factual "Did you know" box need not be replaced with a mathematical connection. Change it when it is wrong, unclear, irrelevant, or when a demonstrably better fact serves the reader.
- Explain related metrics where a comparison helps. Distinguish a possible explanation for differing scores from a conclusion those scores establish.
- Keep visual and formula changes within the authorized scope. If an existing figure contradicts a correction, state the unresolved issue rather than writing a misleading caption or quietly changing the plot.

## Check the result in context

Use **book-review** on the revised passages and compare them with the original. Ask whether the correction is both accurate and easier to follow. Keep the stronger original wording where a rewrite loses clarity. A separate review skill or mandatory agent loop is unnecessary; use an independent reviewer only when authorized and useful.

Compile and inspect the affected PDF pages at readable size. Preserve the local page format rather than forcing every entry into the same template. Check neighboring page breaks as well as the changed paragraph.

If text overflows, remove repetition before shrinking fonts or crowding the page. Check which box or float moved. `figure*` with `[H]` has caused disappearing figures here; use an appropriate existing float pattern or a nonfloating centered image when layout work is in scope. `\enlargethispage` can help a borderline page, but inspect the footer and avoid repeated spacing hacks. Report a remaining layout issue honestly.

## Deliver

Prepare a labeled original/revised PDF comparison using the stable original baseline and corresponding pages. Record the commit or source snapshot for each side and inspect the comparison itself. Make the requested local checkpoint and summarize substantive changes and unresolved feedback.

Publishing and Supabase updates follow the user's request and **AGENTS.md**. A revision awaiting human review is not automatically an approved or published book version.
