# Judge instructions for the evaluation set

The same instructions score the current generator (baseline) and the redesigned one, so results compare. Run the judge on a model other than the writer, and for comparisons run it three times and take the median.

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

Return a JSON object followed by a short summary:

```
{"item": ..., "scores": {"A": n, ..., "F": n}, "principles": {...},
 "claims": {"checked": 15, "unsupported": n, "wrong": n, "examples": [...]},
 "checklist": {"repeated": n, "avoided": n, "na": n, "repeated_ids": [...]},
 "extras": {...}, "top_issues": ["...", "..."]}
```
