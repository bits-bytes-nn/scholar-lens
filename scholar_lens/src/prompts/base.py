from abc import ABC
from dataclasses import dataclass

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Shared writing rules for the review, summary and guide prompts. Each block
# states one concern once; prompts compose the blocks they need instead of
# restating them, so the rules cannot drift between artifacts.

READER_PROFILE: str = """
The reader is a general data scientist or ML engineer: fluent in standard deep
learning (Transformers, attention, embeddings, Adam, cross-entropy, fine-tuning,
common benchmarks and metrics) but not a specialist in this topic.
"""


SUMMARY_DEPTH_RULES: str = """
A summary is short: about 2,000-3,000 Hangul syllables of prose in Korean
(the same reading time in other languages), readable in five minutes.
- Spend the words on the core mechanism: its intuition, its key equation with
  symbols defined, and at most one small worked example.
- Results: one distilled table plus at most three findings, each with its
  setting.
- No `###` inside a section except in the results section.
"""

DEPTH_RULES: str = """
Spend words where the difficulty is, not evenly.
- Easy material (standard setups, well-known components, routine results): one
  or two sentences, then move on. Never pad familiar ideas.
- Hard material (the paper's novel mechanism, dense math, non-obvious
  derivations, counter-intuitive results): build it up in this order —
  (1) the intuition in plain words, (2) the formal statement with every symbol
  defined, (3) a small worked example or numeric walk-through where it helps
  (for a theorem, the idea of the proof), (4) the paper's figure or a table
  where it helps. Say briefly why the point is subtle. Each step adds
  something new; never restate the previous one, and a callout replaces the
  sentence it highlights instead of repeating it.
- Simplify the exposition, never the facts.
"""

# Number slips found in published paper posts (blog PRs #88, #89); writers
# avoid them, checkers list them as must-fix. The TL;DR and the description
# are also matched against the paper in code (post_validator.unsourced_numbers).
NUMBER_RULES: str = """
- Copy numbers verbatim with the model, variant (base or chat, pre-trained or
  fine-tuned, size), cell and setup (not an ablation's) they belong to.
- 1B = 10억, 1M = 100만, 1T = 1조 (7B = 70억). A relative gain is not %p; an
  exponent is not a share; a share of outputs is not a rate; a formula's
  input is not its result.
- "모든 벤치마크에서 앞섰다" only with no reported loss; expand an acronym only
  as the paper does.
"""

SUPPLEMENTARY_CONTEXT_RULES: str = """
Context beyond the paper makes the text useful, but must be honest:
- Allowed: background from the provided citation summaries or code, and
  textbook-level knowledge (what a standard technique is, why a known problem
  matters). Phrase it as explanation, not as a claim about this paper.
- Never invent numbers, benchmark results, attributions, or links that are not
  in the inputs. Clearly illustrative toy values in a worked example are fine;
  never present them as the paper's numbers.
- Claim only as strongly as the paper does: keep its hedges ("we believe",
  "suggests"), and never turn a conjecture into a fact, one setting into a
  general rule, an unexplained difference into a stated cause, or "inspired
  by X" into "uses X".
- A derived number (a ratio, a reduction, a difference) shows its inputs
  inline: "1,200ms에서 300ms로, 75% 감소". Compare numbers only on the same
  benchmark, metric and evaluation setting; if the compared models differ in
  size, say so.
- Code: quote or adapt code only from the provided code examples. Without them,
  write code only when it clarifies something an equation cannot, label it as
  pseudocode reconstructed from the paper, and never cite file paths or symbol
  names you have not seen. Do not invent unstated constants.
""" + NUMBER_RULES

HEADING_STRUCTURE_RULES: str = """
- Never write an H1 (`#`) heading; the post title comes from the front matter.
  Top-level sections use `##`, then `###` and `####`; never skip a level.
- Headings are descriptive, in the target language, and never numbered
  ("## Proposed Method" / "## 제안 방법", not "## 3. Method").
"""

CITATION_KEY_RULES: str = """
- Never expose citation keys (e.g. smith2023transformer) in prose.
- Link a cited work only with a URL copied verbatim from citation_summaries:
  "[Vaswani et al.](url)" / "[Vaswani et al.](url)이 제안한 Transformer". Your
  memory of arXiv ids is unreliable; a wrong link is worse than none.
- Without such a URL, name the work in plain text by its method or model name
  ("the Transformer architecture" / "Adapter 계열 기법"), never with a vague
  phrase like "prior work" / "관련 연구에 따르면".
"""

EXCLUDED_CONTENT_RULES: str = """
Silently skip non-technical sections: acknowledgments, author contributions,
funding, gratitude, contributor lists, and the bibliography itself (in-text
citations stay). Never mention that something was skipped; if the whole input
is such a section, output nothing for it.
"""

IMAGE_PATH_RULES: str = """
- Insert an image only if its exact path appears in the provided figure markers
  or sources, copied character for character (local paths stay local; URLs stay
  URLs). Never invent, guess, or repair an image path, and never use a paper's
  landing/HTML page (e.g. an arxiv.org/html/<id> or /abs/ URL) as an image.
- If no real figure exists for a point, describe it in prose instead.
"""

VISUAL_RULES: str = """
- Place a figure right where its content is discussed, introduce it in the
  adjacent sentence ("The figure below shows …" / "아래 그림은 …를 보여줍니다"),
  and explain what to look at. Show each figure, table, code block and equation
  only once in the whole post; afterwards refer to it descriptively ("the loss
  defined above" / "앞서 정의한 손실 함수").
- Refer to figures, tables, equations, theorems and sections descriptively,
  never by the source's numbers ("Figure 3", "Table 2", "Eq. (4)", "Theorem 1",
  "Section 3.2", "그림 3", "3.2절"): those numbers do not exist in this post.
- Every image has alt text in the post's language that says what the figure
  shows ("인코더-디코더 구조도"), never an English caption, a file name or
  "Figure 3".
- Prose is the default. Use a short list only for genuinely parallel items
  (algorithm steps, three or more options). Use a callout ("> **핵심:** …" /
  "> **Key idea:** …") only for a non-obvious insight, at most three in the
  whole post.
- Never emit Mermaid, Graphviz or PlantUML blocks; they do not render.
"""

# Table shape (checked in code, so writers get it and checkers do not).
TABLE_RULES: str = """
- Tables are real Markdown tables, never images. Use exactly one header row
  (merge multi-level headers per column as "Parent Child", e.g. "AIME 2024
  pass@1"), give every row the same number of cells, and leave no header cell
  empty except the first. A comparison of many numbers goes in one table; the
  prose interprets it, repeating only a derived number's inputs.
"""

# Size cap of a table copied from the paper (not checked in code).
TABLE_SIZE_RULE: str = """
- A table from the paper has at most 8 rows and 7 columns.
"""

PAPER_TABLE_RULES: str = """
- A table from the paper has at most 8 rows and 7 columns. Copy cells and
  numbers verbatim from the source table; label method/model rows with the
  names used in the prose. To shorten a long table, omit whole rows or columns
  (keep the main method and the strongest baselines) and add one line saying
  that some were omitted; never drop or merge individual cells.
"""

MATH_RULES: str = r"""
- Write all math, inline and display, between $$ and $$ (never a single $):
  inline "the loss $$\mathcal{{L}}$$ decreases"; display on its own line with
  a blank line before and after it, in lists and quotes too.
- Never put Korean (or other non-Latin) text inside math; keep words outside
  or in English inside \text{{...}}. No bare | in inline math: \mid
  (conditional), \lvert x \rvert (length), \lVert x \rVert (norm, KL).
- Use \begin{{aligned}} inside $$...$$ for multi-line equations and
  \begin{{array}} or bmatrix for matrices; never standalone align/equation/
  gather environments. Use \boldsymbol, not \bm; \text{{...}}, not \rm.
- A dollar amount in prose stays plain text ("$5 per hour"), never math.
- Superscripts and subscripts exist only inside math: write
  $$\text{{Model}}^{{L}}$$, never "Model^L" in prose.
"""

# Term rendering; the planner's glossary and every writer, checker and editor
# follow it. The English-stem lint in markdown_lint checks its last bullet.
TERMINOLOGY_RULES: str = """
- A term the translation guideline lists takes its rendering there (the
  glossary already applies it). Any other technical term is written in the
  target language when a natural translation exists. Keep English only for
  proper nouns, product, API, model and code names (vLLM, Adam, `max_tokens`)
  and terms with no natural translation.
- A rendering carries no parenthetical: add the English once, at the term's
  first use in the post ("연속 배칭(continuous batching)").
- Never glue a Korean ending to an English word ("prefill합니다"): write
  "prefill을 수행합니다".
"""

STYLE_RULES: str = """
- Write natural, flowing prose in the target language, in a professional yet
  accessible register. No translationese, no hype ("revolutionary", "획기적인"),
  no filler intensifiers.
- Prefer impersonal phrasing; avoid "we"/"I" ("우리"/"저").
- Do not announce what a section will do ("This section explains …" / "이 절에서는
  …를 살펴봅니다"), recap the previous one, or promise later content; start with
  the substance and stop when it is complete. Never refer to the post's own
  parts by structure, by name or number ("이 절", "앞 절", "설치 절에서 다룬" /
  "this section"); a back-reference by content is fine ("앞서 설명한 `--host`
  설정" / "the `--host` setting described above").
- WHEN THE OUTPUT LANGUAGE IS KOREAN: use the "입니다" register throughout.
""" + TERMINOLOGY_RULES


@dataclass(frozen=True)
class BasePrompt(ABC):
    system_prompt_template: str
    human_prompt_template: str
    input_variables: list[str]
    output_variables: list[str] | None = None

    def __post_init__(self) -> None:
        self._validate_prompt_variables()

    def _validate_prompt_variables(self) -> None:
        if not self.input_variables:
            return
        for var in self.input_variables:
            if not isinstance(var, str) or not var:
                raise ValueError(f"Invalid input variable: {var}")
            if (
                var != "image_data"
                and f"{{{var}}}" not in self.human_prompt_template
                and f"{{{var}}}" not in self.system_prompt_template
            ):
                raise ValueError(
                    f"Input variable '{var}' not found in any prompt template."
                )

    @classmethod
    def get_prompt(cls, enable_prompt_cache: bool = False) -> ChatPromptTemplate:
        system_template = cls.system_prompt_template
        human_template = cls.human_prompt_template
        instance = cls(
            input_variables=cls.input_variables,
            output_variables=cls.output_variables,
            system_prompt_template=system_template,
            human_prompt_template=human_template,
        )

        if enable_prompt_cache:
            system_msg = SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": instance.system_prompt_template,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            )
            human_msg = HumanMessagePromptTemplate.from_template(
                instance.human_prompt_template
            )
            return ChatPromptTemplate.from_messages([system_msg, human_msg])

        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    instance.system_prompt_template
                ),
                HumanMessagePromptTemplate.from_template(
                    instance.human_prompt_template
                ),
            ]
        )
