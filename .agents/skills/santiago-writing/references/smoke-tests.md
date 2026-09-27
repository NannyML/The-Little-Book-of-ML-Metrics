# Manual smoke tests

Use these prompts to evaluate the installed skill. These are proposed behavioral tests, not a claim of successful execution in Codex. Judge meaning, scope, and edit depth rather than requiring an exact string match.

## Grammar only

Prompt: "$santiago-writing Check grammar: nope it is not registered anywhere. my idea is to tomorrow create the landing page, deploy on vercel and send you the DNS records"

Pass: Preserve "Nope" and the plan, fix syntax and capitalization, add no new facts, and return only the edited text.

## Preserve facts and uncertainty

Prompt: "$santiago-writing Make this simpler: The integration works. Slack alerts are not ready. We might test it internally next week. The agent will suggest a fix, not apply it."

Pass: Preserve all four facts, including "might," the missing alerts, and the distinction between suggesting and applying a fix.

## Natural community post

Prompt: "$santiago-writing Draft a short Reddit post asking how people notice when a Supabase table stops updating or gets fewer new rows. I have had a cron job fail without noticing. Do not promote a product."

Pass: Preserve the supplied experience, cover both failure modes, ask a genuine question, and add no product pitch or invented experience.

## Keywords and capability boundary

Prompt: "$santiago-writing Write one landing-page sentence for Supabase builders. Include cron job and edge function. The product detects missing table updates and alerts me; it does not fix the issue."

Pass: Include both exact phrases naturally. Do not promise a fix or replace the plain problem with vague observability claims.

## Required technical vocabulary

Prompt: "$santiago-writing Write two sentences for data governance teams at Soda. Explain that a shared data quality policy can apply to several data contracts. Keep the terms data quality and data contracts."

Pass: Keep the required terms. Do not apply the Supabase-builder vocabulary restrictions to this audience or invent implementation details.

## Override the default voice

Prompt: "Use a formal institutional tone, not my usual casual voice. Draft a two-sentence acknowledgement that we received the application and will reply by Friday."

Pass: Follow the explicit formal-tone instruction and the supplied deadline. Do not force Santiago's personal voice.

## Protect code placeholders

Prompt: "$santiago-writing Improve only this UI string: `Your dataset {datasetName} has encountered {failedCount} unsuccessful validations.` Keep both placeholders exactly."

Pass: Return a plainer string while preserving `{datasetName}` and `{failedCount}` exactly. Do not rename code or add behavior.

## Do not over-trigger

Prompt: "Fix this Python function's sorting bug without changing its output messages."

Pass: Do not rewrite prose or invoke a personal writing workflow solely because Santiago owns the project.

## Scope a one-off exclusion

Prompt: "$santiago-writing Shorten this paragraph. Keep the existing bullets and the term observability because the audience uses it."

Pass: Respect the requested bullets and term instead of treating earlier campaign choices as universal bans.

## Connected technical explanation

Prompt: "$santiago-writing Make this explanation natural without fluff: The setting z controls the width of the range. Lower values flag smaller changes. Higher values allow more variation. A common choice for this type of rule is z = 3. Settings depend on the tool."

Pass: Connect the relationship between lower and higher settings, vary rhythm naturally, preserve the tool-dependent qualification, and add no story, metaphor, or unsupported claim. Short sentences are allowed; a sequence of isolated facts or one overloaded sentence is not the goal.

## Metric opening

Prompt: "Improve this book opening: For two summaries that use different words, BERTScore can give a high match. It uses contextual embeddings and greedy token matching."

Pass: Explain what BERTScore compares in plain language first, introduce how it compares meaning, and use an example only if it clarifies. Do not imply factual correctness or agreement with human judgment is guaranteed.

## Chapter setup

Prompt: "Write an opening for the GenAI metrics chapter. It covers text, images, speech, human ratings, and influential benchmarks. Don't start with an example."

Pass: Establish the subject and purpose before explaining comparisons. Avoid an acronym catalog, manufactured intrigue, and a list of examples without context. Do not invent coverage beyond the supplied topics.

## Preserve and correct

Prompt: "Make this conditional-rate explanation easier to understand, keeping its useful loan example. A reviewer supplied six technical caveats."

Pass: Keep the concrete example, make its denominator explicit, and place necessary qualifications where they help. Do not mechanically insert every caveat into the introduction or trade precision for a simpler false statement.
