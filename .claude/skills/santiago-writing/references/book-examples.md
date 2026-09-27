# Book writing examples

Use these to understand the editorial choices, not as a sentence template. The Perplexity "after" is wording Santiago explicitly endorsed on September 27, 2026. Other replacements below are illustrative drafts unless stated otherwise; they are not recorded approvals or changes to the manuscript.

## 1. A definition prepares the reader: Perplexity

Before, supplied by Santiago:

> A language model that scores held-out text at a perplexity of 20 is, at each token, as uncertain as if it were choosing among 20 equally likely words. Perplexity is the exponential of the average negative log-likelihood of the actual next tokens, and it needs no labels: only the text and the model’s probabilities.

After, explicitly preferred by Santiago:

> Perplexity measures how well a language model predicts a piece of text. At each step, the model assigns probabilities to possible next tokens, which can be words or word pieces. We check the probabilities it gives to the tokens that actually appear in the text: the higher those probabilities, the lower the perplexity.

Why it works: the first sentence tells us what we are measuring. Each following sentence explains one part of that process, introduces tokens, and connects assigned probabilities to the score. It gives the reader a basis for understanding the formula later. It does not require an analogy or a numerical example to be useful.

The old "at each token" interpretation also overstates what an aggregate score tells us. Do not preserve that error for the sake of simplicity. Keep the exact definition and evaluation caveats in the appropriate later explanation; the opening does not replace them.

## 2. Chapter introductions need the same preparation

Current GenAI opening, criticized by Santiago for starting with examples:

> An image can look convincing while missing something the prompt requested. A passage can be fluent but unlike human writing, and speech can sound clean while remaining hard to understand.

Illustrative replacement for the chapter introduction, not yet applied or approved:

> Generative AI models produce text, images and speech. Evaluating these outputs helps us understand how well they perform the task we asked for. Their quality depends on several things, including whether they follow the request, resemble relevant examples, and make sense to the people using them.
>
> This chapter introduces ways to assess those qualities. Some metrics compare an output with a reference or a prompt; others compare collections of generated and real examples. We also look at ratings from people and several influential benchmarks used to compare models.

Why this direction is better: it establishes the subject and the purpose of evaluation before describing comparisons. It prepares the reader for the metrics without dropping a list of unfamiliar names into the introduction. A final chapter edit still needs a check against the exact coverage and surrounding pages.

## 3. Introduce a term through its meaning

Abrupt:

> The reference dataset contains earlier examples that the model did not use for training, along with its predictions and their actual outcomes.

Illustrative revision, following a paragraph about waiting for new outcomes:

> To estimate performance while we wait, we start with earlier predictions whose outcomes we already know. These examples form our reference dataset.

The reader learns why we need the data before receiving its name. Keep necessary training/validation separation in the following explanation; this revision is not a complete account of estimator fitting. Do not introduce "a chunk in the figures" merely because the figures use that label.

## 4. Make the denominator concrete

Ambiguous:

> Equality of opportunity means approving men and women at the same rate.

Illustrative revision:

> Equality of opportunity requires equal approval rates among applicants who would repay the loan. If the model approves 80% of those men, it should also approve 80% of those women. Here we compare only people who would repay; demographic parity compares approval rates across all applicants in each group.

The distinction comes from who enters the comparison. Keep it explicit instead of adding increasingly abstract terminology. This explains the criterion, not a claim that repayment outcomes are observable for every rejected applicant or that meeting one metric establishes overall fairness.

## 5. Preserve a historical fact when it earns its place

Do not replace a verified account of who introduced a metric, why they needed it, or how it was adopted solely because the callout lacks a mathematical surprise. If the original fact is clear, interesting, and supported, keep it. Correct misleading implications and verify its bibliography entry. Researcher names and dates are legitimate content, including in benchmark history outside a callout.

## 6. A review comment is not the finished paragraph

A comment may discuss the dependence of comparisons on the dataset, tokenization, evaluation window, and implementation. Extract the relevant correction and explain it where the reader needs it. Do not pack all of those details into the opening just to demonstrate that every caveat was considered.

When judging a revision, read both versions in context. If the original example teaches the idea more clearly, preserve it and correct the specific error. "More accurate but harder to follow" is unfinished work, not an acceptable tradeoff by default.
