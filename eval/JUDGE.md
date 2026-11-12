# Judge instructions for the evaluation set

The same instructions score the current generator (baseline) and a changed one, so results compare. Run the judge on a model other than the writer.

## 0. Comparing two versions (the decision protocol)

A change of a default (an engine, a prompt rule, a check) is decided this way (ADR 0002):

1. Items: the `confirm` items of `eval/sets/baseline.yaml` (never generated before they joined the set). `dev` items may guide work but do not decide.
2. Generation: each version writes every item; both versions share the reading cache, so they differ in what they write, not in what they read.
3. Blinding: for each item the two posts are copied to `X.md` and `Y.md` in a random order recorded in `blind-mapping.json`, kept from the judges.
4. Judging: three independent judge sessions per pair (separate agents, no shared context, none sees the mapping or the other judgments). Each writes the result JSON below plus a pairwise `preference`.
5. Decision: `python -m eval.compare <run dir>` unblinds and counts. A version wins an item when at least two of its three sessions prefer it; it wins the comparison when it wins more items than it loses and no item is lost on criterion F (no hallucination). Anything else is "no measurable difference": the default stays.

Record every run under `eval/runs/<name>/` (judge JSON and `blind-mapping.json`; posts are not committed) and its conclusion in `eval/RESULTS.md`, with the generation's code commit, the judge model and the cost.

Inputs:
- the post (Markdown);
- the item's entry in `eval/sets/baseline.yaml` (its sources);
- the item's checklist in `eval/sets/human_fixes.yaml`.

## 1. Rubric scores

Score each criterion A–F of `eval/ARTIFACT_RUBRIC.md` from 0 to 10, using its anchors. Every deduction quotes the post with a line number.

Also check the writing principles of `docs/redesign.md` §5.1 and record each as met, partly met or not met, with evidence:
- claim headings;
- a running example;
- mechanism → design choice and reason → formula or code → worked example;
- facts said once;
- numbers with variant, scope and unit;
- nothing invented;
- the writer's own assessment at the end;
- length fitted to difficulty;
- type-specific shape, with the open-the-guide-with-TL;DR/architecture-first and survey-numbers-in-one-table rules;
- Korean style.

## 2. Claim check

Pick 15 factual claims from across the post: numbers, method details, attributions and comparisons, with at least 8 of them numbers.

Check each against the primary source:
- for a paper, the arXiv HTML at `https://arxiv.org/html/<id>` or the abstract page;
- for a guide, the official docs page.

Record each claim as supported, unsupported, wrong or unverifiable, with the source passage. Report how many are unsupported or wrong.

## 3. Human-fix checklist

For each `check` in the item's checklist, record whether the post repeats the defect (`repeated`), avoids it (`avoided`) or does not cover it (`n/a`). Quote the evidence.

## 4. Survey and guide extras

- Survey: plan papers the post never cites, sections that are only a heading or one paragraph, venue strings that are wrong (index names, publisher series, lowercase or garbled names), and comparison-table cells that are prose, empty or "-".
- Guide: where architecture and internals first appear (line), the share of non-official sources, and whether it opens with the problem and closes with when to choose it and its failure modes.

## Output

Return a JSON object followed by a short summary. A pairwise judgment holds one such object per post under `X` and `Y` and a preference:

```
{"item": ..., "judge_model": ..., "session": ...,
 "X": {"scores": {"A": {"score": n, "reason": ...}, ...}, "principles": {...},
       "claims": {"checked": n, "unsupported": n, "wrong": n, "error_rate": r, "items": [...]},
       "checklist": {"repeated": n, "avoided": n, "na": n, "repeated_ids": [...]},
       "structure_defects": n, "top_issues": ["...", "..."]},
 "Y": {...},
 "preference": {"overall": "X" | "Y" | "tie", "per_criterion": {"A": "X", ...}, "reason": ...}}
```
