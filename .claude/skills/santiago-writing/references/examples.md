# Calibration examples

## Contents

- [1. Grammar means a light edit](#1-grammar-means-a-light-edit)
- [2. Shorten an update without overstating progress](#2-shorten-an-update-without-overstating-progress)
- [3. Keep a community question genuinely curious](#3-keep-a-community-question-genuinely-curious)
- [4. Explain the useful behavior](#4-explain-the-useful-behavior)
- [5. Make the form understandable on its own](#5-make-the-form-understandable-on-its-own)
- [6. Keep keywords without stuffing them](#6-keep-keywords-without-stuffing-them)
- [7. Preserve uncertainty during a grammar check](#7-preserve-uncertainty-during-a-grammar-check)
- [8. Respect a narrow edit](#8-respect-a-narrow-edit)

These are newly written illustrations based on the available requests. The proposed outputs are not a historical record of wording Santiago accepted. Facts are fixture inputs for each example, not current product claims.

## 1. Grammar means a light edit

Request: Check grammar.

Input:

> nope it is not registered anywhere.
> my idea is to tomorrow create the landing page, deploy on vercel and send you the DNS records

Use:

> Nope, it is not registered anywhere.
>
> My idea is to create the landing page tomorrow, deploy it on Vercel, and send you the DNS records.

Avoid:

> The domain remains unregistered. Tomorrow, I intend to develop and deploy the landing page, after which I will provide the requisite DNS configuration details.

Preserve the person's wording rather than upgrading the register.

## 2. Shorten an update without overstating progress

Request: Make this very simple and shorter.

Input facts: The Supabase integration and automatic monitoring setup work. Slack alerting and an agent for root-cause analysis and fix suggestions are missing. Next week is for improving the flows; an internal trial may follow.

Use:

> The Supabase integration and automatic monitoring setup are working. We still need Slack alerts and an agent that finds the cause and suggests a fix.
>
> Next week we'll improve the flows, then we should have something to try internally.

Avoid:

> The product is ready to launch with seamless monitoring, intelligent alerting, and automated remediation.

Do not turn a partial implementation into a launch or suggestions into automatic fixes. Technical vocabulary is appropriate in this internal update.

## 3. Keep a community question genuinely curious

Request: Draft a casual Reddit post. Mention that I have had a cron job stop updating a table and only noticed when using the app. Ask about both stopped updates and changes in the number of new rows. No product pitch.

Use:

> I've had a cron job stop updating a Supabase table and only noticed when I opened the app and the data was old.
>
> How do you notice when updates stop, or when you're suddenly getting way fewer new rows? Is there something you use for this, or do you just build your own alerts?

Avoid:

> Supabase developers: Are silent pipeline failures compromising your data reliability? Discover how proactive observability can transform your workflow.

Use the supplied experience, familiar words, and an actual question. Do not add marketing framing.

## 4. Explain the useful behavior

Request: Improve a feature description for users.

Input fact: An AI conversation remains available while a user navigates between product pages.

Use:

> Your conversation stays with you as you move between pages.

Avoid:

> Experience a seamless, persistent, contextually integrated conversational intelligence layer.

Show the behavior without invented benefits or unnecessary adjectives.

## 5. Make the form understandable on its own

Request: Improve a Supabase onboarding form introduction and one question. Do not lead with a time estimate.

Use:

> We're getting ready to onboard your Supabase project. Tell us what it does so we can start with the projects we can help most.

Question:

> What updates the data in your Supabase tables?

Avoid:

> Takes about a minute. How does it get there?

Lead with the reason for the form and name the thing the question is asking about.

## 6. Keep keywords without stuffing them

Request: Write one landing-page sentence for Supabase builders. Include "cron job" and "edge function." Supplied capability: detect when a table stops receiving expected updates and notify the user. Do not promise a fix.

Use:

> Know when a cron job or edge function stops updating your Supabase tables.

Avoid:

> The ultimate cron job monitoring and edge function observability platform for seamless data quality, reliability, and automated remediation.

Preserve the requested keywords and actual capability, not an invented product category or promise.

## 7. Preserve uncertainty during a grammar check

Request: Check grammar.

Input:

> clicks is going down, mostlikely because of AI

Use:

> Clicks are going down, most likely because of AI.

Do not silently change it to "AI is causing clicks to fall." For a separate analysis request, explain that this causal claim needs evidence; editing the sentence does not verify it.

## 8. Respect a narrow edit

Request: Make only the second sentence easier to follow.

Input:

> We can try it internally next week. The part that we need before people can actually use it and know something happened is alerts in Slack.

Use:

> We can try it internally next week. We still need Slack alerts so people know when something goes wrong.

Leave the first sentence alone. Do not rewrite the whole update because a broader rewrite sounds smoother.

## 9. Let an explanation flow

Calibration from the September 21, 2026 book review; Santiago endorsed this direction.

Stiff:

> Freshness measures how much time has passed since the latest timestamp in a column. With an arrival timestamp, it tells you when data last reached the table. With an event timestamp, it tells you when the latest recorded event happened.

More natural:

> Freshness tells you how long it has been since the latest timestamp in a column. What that tells you depends on the timestamp you choose: an arrival timestamp tracks when data last reached the table, while an event timestamp tracks when the latest recorded event happened.

The revision connects the two cases through the choice the reader makes. It adds no anecdote or decorative example and preserves the technical distinction. Do not copy this sentence structure into every paragraph.
