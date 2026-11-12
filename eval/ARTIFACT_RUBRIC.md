# Scholar-Lens Artifact-Quality Rubric

Rubric for judging the **content quality of generated artifacts** — paper
**reviews**, paper **summaries**, and **technical guides/blog posts** — as
opposed to code quality. Used by an LLM-judge subagent.

Every deduction MUST cite concrete evidence from the artifact (a quoted phrase
or a line/section reference). Score each criterion 0–10.

**Scoring anchors**
- **10** — Exemplary; publish as-is to a senior technical audience.
- **9** — Excellent; only cosmetic nits.
- **8** — Good; 1–2 substantive but non-blocking issues.
- **6–7** — Acceptable; several real issues a reader would notice.
- **≤5** — Needs rework; a reader would be misled, lost, or bored.

**Gate target:** every criterion ≥ 8.5; none < 8.0; `F. No hallucination` must
be ≥ 9.0 (factual integrity is non-negotiable).

---

## Shared criteria (all artifact types)

- **A. Structure & blog conventions** — logical section ordering; headings
  reflect a coherent arc; no orphaned/duplicate/empty sections; Markdown that
  renders on the Jekyll blog: no body H1 and no skipped heading levels, real
  tables, fenced code, and math only as `$$…$$` (inline and display; display
  blocks surrounded by blank lines, also inside blockquotes; no `\( \)` or
  `\[ \]`, no single `$` or TeX next to one, no unpaired `$$`, no `\_` inside
  math outside `\text{}`). Front matter: `subtitle` and `paper_author`
  (publishing organization) on paper posts, plus `paper_doi` when the paper
  has a DOI; no `author` key, two-level `categories`, `use_math` only when
  math is present.
- **B. Expression** — clear, precise technical prose; consistent terminology;
  natural target-language writing (Korean by default, "입니다" register in
  reviews and summaries) that keeps established English technical terms; no
  machine-translation artifacts, hype words, or "이 절에서는 …" framing.
  Naturalness (Korean): reads as a Korean engineer writing for colleagues,
  not as a translation, judged against the blog's hand-edited Insights posts.
  Quote each unnatural sentence with its rewrite; deduct for noun frames
  ("~다는 점입니다", "~음을 보였습니다"), particle calques ("~와의 비교에서",
  "정책에 대한 손실", "~에 있어서"), passives that hide who acts and double
  passives ("되어진", "보여지는"), English default verbs (가지다, 수행하다,
  역할을 하다), formulaic sentences ("직관은 다음과 같습니다.", "결국 …라는
  것입니다", "할 수 있습니다" for what simply happens), and a register break
  (a "~한다" sentence or table cell in a "입니다" post, quotations aside),
  and a sentence ending in 다 and a colon before a list ("셋입니다:"; the
  blog lint fails it).
  A meaning-obscuring calque or a register break in a table caps B at 7.
  Em-dashes and colon headings are fine in moderation (the hand-edited posts
  use both).
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
  relationship → math), placed where discussed, referred to descriptively (never
  by the paper's "Figure 3"/"Table 2" numbering), and never duplicated. Tables
  copy the source cells verbatim with one header row and equal cell counts;
  long tables are shortened by whole rows. Absence where one would clearly help
  is a deduction; decorative or redundant visuals are also a deduction.
- **F. No hallucination** — every API, flag, default, number, equation, claim,
  and citation is supported by the source (paper / fetched docs). No invented
  methods, fabricated results, mis-attributed work, or broken/empty links. This
  is the highest-stakes criterion. Check explicitly (each found case is a
  deduction of at least 2):
  - table and result numbers match the source table cell for cell;
  - no future work the paper already does (an appendix derivation counts as
    done); future work comes only from the paper's Limitations/Future Work/
    Conclusion text;
  - formulas from outside the paper are exactly right;
  - derivation steps agree in signs and variable definitions (the gradient's
    final form matches its intermediate variables);
  - no adjacent duplicate table rows;
  - "all"/"모든" about a setting holds in every experiment;
  - math symbols: \mid for conditionals, \lvert\rvert for lengths, \|/\Vert
    only for norms and KL divergence.
  - each number is attributed to the exact model, variant and setting of its
    row (not a neighbouring row's value, not the SFT score for the RL model,
    not the pre-trained score for the instruct model: Llama 3 405B MMLU 87.3,
    not 85.2);
  - unit conversions are exact: 1B = 10억, 1M = 100만, 1T = 1조 (Mistral 7B is
    70억 parameters, not 7억; Qwen3's 0.6B is 6억, not 0.6억);
  - an exponent or coefficient stays one (DeepSeek LLM's M_opt ∝ C^0.5243 is
    not "52.43% of the budget");
  - a setting used only in an ablation is not the overall setup (Gemma 2's
    7B teacher);
  - an acronym is expanded as the paper expands it (RAPTOR is Recursive
    Abstractive Processing for Tree-Organized Retrieval);
  - relative gains stay relative (55.4→63.8 is +15.2%, not +15.2 points);
  - "best"/"most"/"almost all" keep the paper's scope ("among open models")
    and agree with the post's own tables;
  - a quantity keeps its definition (a share of memorised outputs is not an
    emission rate; a formula's input is not its result);
  - a claim credited to a cited work or method is that source's own (the
    15T-token point is Llama 3's, not Chinchilla's).
- **G. Blog markup** (deterministic; the validator and the blog lint must
  both pass): `use_math: true` when the body has `$$`; no `|` or `\|` inside
  `$$` on a table row; a plain-text `description` of 120-160 characters,
  longer than the page title, unique on the site, opening with the model or
  method name (not "대규모 언어 모델의 발전…"), with numbers only from the
  TL;DR (a guide or survey: from its body); a guide carries one too; every TL;DR number stated in the paper; Liquid (`{{ }}`, `{% %}`)
  only inside `{% raw %}`; prompt tokens like `<think>`/`<s>`/`<input>` in
  code or escaped; a fence info string of one language word; a
  `YYYY-MM-DD-` file name not before `date:`; tags spelled as on the blog.

## Review-specific

- **R1. Depth & rigor** — explains the paper section-by-section with real
  technical depth: the core method, derivations/intuition, implementation
  details, experimental setup and results, limitations. Not a surface skim.
- **R2. Pedagogical layering** — for hard contributions, builds understanding
  in layers (intuition → formalism → worked example → figure/code) so a reader
  can follow without already knowing the result; length tracks difficulty, so
  a simple paper yields a shorter review.
- **R3. Framing** — the `## TL;DR` answers its four questions briefly and
  concretely (headline numbers with their setting); the closing assessment
  names specific limitations and practical implications, without follow-up
  work, numbers or links the review does not support. Required items and
  limitations are claims the paper's text makes, never values read off a
  figure or critiques of a caption.

## Summary-specific

- **U1. Sufficiency** — the five-question arc (motivation / novel solution /
  implementation / results / significance) is fully covered; a reader finishes
  knowing what the paper did, how, and why it matters — without the full text.
- **U2. Concision & emphasis** — readable in about five minutes; opens with a
  `## TL;DR` section holding a two-to-three-line plain-prose summary (no
  math); depth follows the paper (method papers weight solution and
  implementation, empirical papers weight results, with one distilled results
  table); states at least one grounded limitation; no filler.

## Tech-guide-specific

- **T1. Detail & runnability** — teaches by doing: concrete, runnable code,
  real config/flags/options, end-to-end so a reader can actually use the tool;
  not a vague overview.
- **T2. Grounded coverage** — covers the genuinely important topics present in
  the sources, in a sensible learning order; does not pad with sections the
  sources do not support.
- **T3. Depth and breadth** — beyond installation and usage, explains how the
  key mechanisms work internally and why they are designed that way, what each
  major feature means (problem, mechanism, trade-off, when to use it), and how
  to fit the tool into a larger system (where it sits, interfaces, scaling,
  observability, failure modes), as far as the sources support it;
  installation does not dominate. Commands, flags and defaults follow the
  official docs; concepts and architecture rest on explanatory sources
  (papers, design docs, reputable write-ups), and design advice reads as a
  recommendation, not as documented behaviour. A command, flag or default
  that only an older-docs-version page gives is absent; an older page may
  explain how a mechanism used to work, named as that version's.

## Survey-specific

- **S1. Cards and comparison table** — a paper's limitations are the paper's,
  never the input's ("초록에는 실험 수치가 없어" is a false limitation); a
  table cell's unit words are in the post's language and its setting is a
  noun phrase; nothing an earlier section explained is explained again.

## Taxonomy (paper posts)

- **X1. Category and tags** — the topic category is one of the blog's topics;
  tags are short and reusable: one controlled tag (a second only when the
  method or experiments are about it) and a few specific tags in the
  paper's own terms (an existing tag's spelling only when it names this
  paper's contribution), never a restatement of the title or a technique the
  paper does not use.
