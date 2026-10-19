# Limitations — known gaps we don't paper over

This file is the honest one. Use it when proposing fixes that might hide a weakness.

## The model ceiling is real

Jane's guards are strong; the models under them are the loose end.

- The shipped default is a 7B coder. It is good, but it is not 32B-good.  
- The eval history is full of cases where the *SQL was right and the prose lied*, or the *prose was right and the model burned retries on itself*. Those are model-quality events, not logic gaps.
- A wrong query cannot always be guarded, because the verifier sees what the query returns, not what the table would return under a correct query. That is a known structural limit, not a regression.

Don't promise "perfect on sales data" as a code outcome. Jane can be far better than it is, but model flakiness is a real floor under some questions.

## The receipt is honest about a small guard set

The verification receipt is now measured, not assembled, and that is the important win. It is still only as broad as the guards that actually run.

- `truth` and `contradiction` are the workhorse guards and they run on every answer path.
- Several other guards are question-conditioned and may be silent when they are clean. Silence is not the same as a passed check.
- A skipped-only report is intentionally unverified. If there is nothing to trace a number against, the answer is not certified.

The current engineering TODO calls this out directly: the receipt should eventually say more than "truth and contradiction ran". That is a fidelity problem, not a correctness lie yet.

## Multi-table honesty is still incomplete

Jane can load several tables, and prep makes multi-table the default flow, but the prompt still describes the accessible tables in a way that can blur which one is active.

- The schema block announces the tables the agent may query.
- Verification is anchored to the active table.
- Those two things can disagree once more than one table is loaded.

This is a latent mismatch. It existed before prep, and prep exposed it more often. It is not fixed yet.

## Thin margins and tiny groups can still be overstated

A small lead can look decisive when it is not.

- A 1.5% gap can be presented like a winner.
- A tiny group can make an outlier look like a pattern.
- The answer template once had a "catch" slot that could quietly say "no catch" when there was one.

There is an open idea to make the catch slot respond to thin margins, small groups, and outlier-driven conclusions with something real. Until then, treat "clear winner" language with caution on close calls.

## Some analysis shapes are still flaky

These are the recurring weak spots from live runs:

- Breakdowns that the model trims itself, usually by writing `LIMIT 1` or `LIMIT 2` when the real answer needs all groups.
- Percentage attribution, where a number matches a real pair but the sentence names the wrong pair.
- Comparison conclusions that cite both sides correctly and still pick the wrong winner.
- Conditional percentages computed over the wrong denominator.
- Correlation answers that either parrot the prompt or invent a coefficient.

None of these are exotic. They are the everyday mistakes of a small local model under retry pressure.

## A few UX and platform gaps are still open

These are not model problems, but they are still limitations.

- Session history is not yet surfaced in the React app the way Streamlit reads it.
- There is no export or download path for a result, table, or receipt in the frontend yet.
- A running ask cannot be canceled; the bridge holds one global lock for the run.
- Conversation memory is short and slice-based; summarization is an open idea, not a shipped feature.
- CSV title rows that contain a delimiter can still look like a header row. That is an inherent CSV ambiguity and is documented, not hidden.

## What we will not do

- We will not let the verifier invent numbers to make an answer look certified.
- We will not claim a guard ran when it did not.
- We will not turn a model-quality weakness into a feature claim.
- We will not quietly regress the receipt or the sandbox to make a metric look better.

If a fix would hide one of the above, it has to fight for its life.
