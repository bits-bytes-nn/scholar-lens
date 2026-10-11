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

SUMMARY_DEPTH_RULES: str = """
A summary is short: about 2,000-3,000 Korean characters in total (more only
for a dense paper), readable in five minutes.
- Spend the words on the core mechanism: its intuition, its key equation with
  symbols defined, and at most one small worked example.
- Results: one distilled table plus at most three findings, each with its
  setting.
- Prose is the default; no `###` inside a section except in the results
  section; lists only for genuinely parallel items.
"""

DEPTH_RULES: str = """
Spend words where the difficulty is, not evenly.
- Easy material (standard setups, well-known components, routine results): one
  or two sentences, then move on. Never pad familiar ideas.
- Hard material (the paper's novel mechanism, dense math, non-obvious
  derivations, counter-intuitive results): build it up in this order —
  (1) the intuition in plain words, (2) the formal statement with every symbol
  defined, (3) a small worked example or numeric walk-through, (4) the paper's
  figure or a table where it helps. Say briefly why the point is subtle.
- Simplify the exposition, never the facts.
"""

SECTION_SCOPE_RULES: str = """
Section metadata sets scope, not length: STANDARD = cover the core ideas only;
DETAILED = mention every technical element of the section. ADVANCED depth =
explain derivations and design trade-offs; INTERMEDIATE = add background and
an example; BASIC = follow the paper's own explanation. Within any scope, the
length per element still follows its difficulty.
"""

SUPPLEMENTARY_CONTEXT_RULES: str = """
Context beyond the paper makes the text useful, but must be honest:
- Allowed: background from the provided citation summaries or code, and
  textbook-level knowledge (what a standard technique is, why a known problem
  matters). Phrase it as explanation, not as a claim about this paper.
- Never invent numbers, benchmark results, attributions, or links that are not
  in the inputs. Mark your own derivations or estimates as such (e.g. "a rough
  calculation gives …" / "대략 계산하면 …"). Toy values in a worked example
  are fine when clearly illustrative ("예를 들어 $$d=4$$, $$r=1$$이라면 …");
  never present them as the paper's numbers.
- Code: quote or adapt code only from the provided code examples. Without them,
  write code only when it clarifies something an equation cannot, label it as
  pseudocode reconstructed from the paper, and never cite file paths or symbol
  names you have not seen. Do not invent unstated constants.
"""

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
- Refer to figures, tables, equations and sections descriptively, never by the
  paper's numbers ("Figure 3", "Table 2", "Eq. (4)", "Section 3.2", "그림 3",
  "3.2절"): those numbers do not exist in this post.
- Tables are real Markdown tables, never images. Copy header labels, cells
  and numbers verbatim from the source table. Use exactly one header row
  (merge multi-level headers per column as "Parent Child", e.g. "AIME 2024
  pass@1"), give every row the same number of cells, and leave no header cell
  empty except the first. To shorten a long table, omit whole rows (keep the
  main method and the strongest baselines) and add one line saying that some
  rows were omitted — name an omitted row only if you are certain it was
  omitted; never drop or merge individual cells.
- Prose is the default. Use a short list only for genuinely parallel items
  (algorithm steps, three or more options), and at most one callout per hard
  concept, e.g. "> **핵심:** …" / "> **Key idea:** …".
- Never emit Mermaid, Graphviz or PlantUML blocks; they do not render.
"""

MATH_RULES: str = r"""
- Write ALL math, inline and display, between $$ and $$ — the blog renders
  nothing else (no \( \), no \[ \], no single $). Inline: "the weight
  $$W_0$$ stays frozen". Display: $$ ... $$ on its own line with a blank line
  before and after.
- Do not escape underscores inside math ($$W_0$$, not $$W\_0$$); only inside
  \text{{...}} or \texttt{{...}} write \_. Never put Korean (or other non-Latin)
  text inside math; keep words outside or in English inside \text{{...}}.
- Inside inline math never use a bare | (it breaks tables): use
  \lvert x \rvert, \lVert x \rVert or \mid.
- Use \begin{{aligned}} inside $$...$$ for multi-line equations and
  \begin{{array}} or bmatrix for matrices; never standalone align/equation/
  gather environments. Use \boldsymbol, not \bm; \text{{...}}, not \rm.
- A dollar amount in prose stays plain text ("$5 per hour"), never math.
- Superscripts and subscripts exist only inside math: write
  $$\text{{Adapter}}^{{L}}$$, never "Adapter^L" in prose.
"""

STYLE_RULES: str = """
- Write natural, flowing prose in the target language, in a professional yet
  accessible register. No translationese, no hype ("revolutionary", "획기적인"),
  no filler intensifiers.
- Prefer impersonal phrasing; avoid "we"/"I" ("우리"/"저").
- Do not announce what a section will do ("This section explains …" / "이 절에서는
  …를 살펴봅니다"), recap the previous one, or promise later content; start with
  the substance and stop when it is complete.
- Keep established English technical terms when the target language has no
  clear equivalent (e.g. BERT, Transformer); otherwise use the target
  language's standard terms.
- WHEN THE OUTPUT LANGUAGE IS KOREAN: use the "입니다" register throughout and
  standard translated terms ("손실 함수", "경사 하강법", "사전 학습").
"""


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
