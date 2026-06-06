# Scholar-Lens Artifact-Quality Rubric

Rubric for judging the **content quality of generated artifacts** — paper
**reviews**, paper **summaries**, and **technical guides/blog posts** — as
opposed to code quality (see `RUBRIC.md`). Used by an LLM-judge subagent.

Every deduction MUST cite concrete evidence from the artifact (a quoted phrase
or a line/section reference). Score each criterion 0–10.

**Scoring anchors**
- **10** — Exemplary; publish as-is to a senior technical audience.
- **9** — Excellent; only cosmetic nits.
- **8** — Good; 1–2 substantive but non-blocking issues.
- **6–7** — Acceptable; several real issues a reader would notice.
- **≤5** — Needs rework; a reader would be misled, lost, or bored.

**Gate target:** every criterion ≥ 8.5; none < 8.0; `G. No hallucination` must
be ≥ 9.0 (factual integrity is non-negotiable).

---

## Shared criteria (all artifact types)

- **A. Structure** — logical section ordering; headings reflect a coherent
  arc; no orphaned/duplicate/empty sections; correct Markdown (renders cleanly
  on a Jekyll blog: valid headings, tables, fenced code, `$…$`/`$$…$$` math).
- **B. Expression** — clear, precise technical prose; consistent terminology;
  natural target-language writing (Korean by default) that keeps established
  English technical terms; no awkward machine-translation artifacts.
- **C. Smooth connection (flow)** — sections and paragraphs transition
  logically; concepts are introduced before they are used; cross-references are
  accurate (no "see section 17" when there are 12); reads as one coherent
  document, not stitched fragments.
- **D. Timely kindness (적재적소의 친절함)** — genuinely hard parts get extra
  help (intuition, analogy, worked example, a defined term) at the moment they
  appear; easy parts are NOT padded. Calibrated to a competent-but-non-expert
  reader. Over-explaining the trivial is a deduction just as under-explaining
  the hard is.
- **E. Visual/illustrative elements** — figures, tables, equations, and code
  are used where they aid understanding (comparison → table; algorithm → code;
  relationship → math), are correctly formatted, captioned/referenced in prose,
  and never duplicated. Absence where one would clearly help is a deduction;
  decorative or redundant visuals are also a deduction.
- **G. No hallucination** — every API, flag, default, number, equation, claim,
  and citation is supported by the source (paper / fetched docs). No invented
  methods, fabricated results, mis-attributed work, or broken/empty links. This
  is the highest-stakes criterion.

## Review-specific

- **R1. Depth & rigor** — explains the paper section-by-section with real
  technical depth: the core method, derivations/intuition, implementation
  details, experimental setup and results, limitations. Not a surface skim.
- **R2. Pedagogical layering** — for hard contributions, builds understanding
  in layers (intuition → analogy → formalism → code/usage) so a reader can
  follow without already knowing the result.

## Summary-specific

- **U1. Sufficiency** — the five-question arc (motivation / novel solution /
  implementation / results / significance) is fully covered; a reader finishes
  knowing what the paper did, how, and why it matters — without the full text.
- **U2. Concision** — appropriately compressed (≈2 pages); detailed on the
  novel solution + implementation, brief on background/results/future; no
  filler.

## Tech-guide-specific

- **T1. Detail & runnability** — teaches by doing: concrete, runnable code,
  real config/flags/options, end-to-end so a reader can actually use the tool;
  not a vague overview.
- **T2. Grounded coverage** — covers the genuinely important topics present in
  the sources, in a sensible learning order; does not pad with sections the
  sources do not support.
