from abc import ABC
from dataclasses import dataclass

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

GRANULARITY_RULES: str = """
**STANDARD**: Concise summary of key ideas and main concepts
- Focus on brevity and clarity with essential technical details
- Include key equations, figures, and core methodologies
- Suitable for Introduction, Related Work, Evaluation, and Conclusion sections
- Avoid exhaustive step-by-step breakdowns while maintaining technical accuracy

**DETAILED**: Comprehensive explanation with no omissions
- Include ALL technical content, equations, algorithms, methodological details
- Step-by-step explanation of all concepts, equations, algorithms
- Maximum depth and coverage of all technical content
- Nothing from original paper should be omitted or oversimplified
"""

TECHNICAL_DEPTH_RULES: str = """
**BASIC**: Direct explanation following the paper
- Use original expressions/terms/examples/equations from paper
- Explain core concepts clearly but concisely
- Focus on fundamental understanding and accurate representation

**INTERMEDIATE**: Enhanced explanation with supplementary materials
- Detailed concept and methodology explanations
- Include practical examples, use cases, background knowledge
- Properly attribute and integrate explanations from referenced papers
- Include expanded mathematical derivations with step-by-step walkthroughs
- Provide code implementations illustrating key concepts
- **REQUIRED: Moderate use of supplementary materials (citations OR code)**

**ADVANCED**: Comprehensive expert-level explanation
- Most detailed and thorough explanation possible
- In-depth theoretical foundations and mathematical proofs
- Extensive practical examples and implementations
- Full mathematical derivations with insights into each step
- Complete code implementations with detailed explanations
- **REQUIRED: Extensive use of supplementary materials (citations AND code)**
"""

GRANULARITY_DEPTH_CONFLICT_RULES: str = """
**STANDARD + ADVANCED**: Select only core concepts (~60% coverage), explain selected concepts in depth (100% depth)
**STANDARD + INTERMEDIATE**: Enhanced explanations while maintaining brevity (~70% coverage with 80% depth)
**DETAILED + BASIC**: Cover all content comprehensively (100% coverage) at fundamental level (60% depth)
**DETAILED + INTERMEDIATE/ADVANCED**: Maximum thoroughness (100% coverage with 80-100% depth)
"""

DIFFICULTY_ADAPTIVE_RULES: str = """
Beyond the configured granularity/depth, calibrate effort to the INTRINSIC
DIFFICULTY of each concept so the review reads as "easy where it can be, patient
where it must be":

**EASY concepts** (standard setups, well-known components, routine results):
- State them concisely and move on. Do NOT pad familiar ideas with filler.
- A single clear sentence + the relevant equation/figure reference is enough.

**HARD concepts** (novel mechanisms, dense math, non-obvious derivations,
counter-intuitive results, anything a capable student would stumble on):
- Slow down and scaffold understanding before formalism:
  1. Give the intuition first (a plain-language "what this really means").
  2. Add a concrete analogy or worked micro-example.
  3. Then present the formal statement / derivation step by step.
  4. Reinforce with a visual aid where one exists — reference the most relevant
     figure, table, or code snippet from the source materials (never invent one).
  5. Pull in citation_summaries for background the paper assumes but doesn't teach.
- Briefly flag WHY it is subtle ("the non-obvious part is ...") so the reader
  knows where to focus.

**Calibration discipline**:
- Spend the explanation budget where difficulty is highest; do not distribute it
  uniformly. Skimming the easy 80% buys depth for the hard 20%.
- Visual/auxiliary materials (figures, tables, code, citations) are tools for
  HARD parts first; do not decorate easy parts with them.
- Never sacrifice accuracy for accessibility — simplify the exposition, not the
  facts.
"""

HEADING_STRUCTURE_RULES: str = """
- Main title: # Paper Title (single # only)
- Section: ## Section Title (double ##)
- Subsection: ### Subsection Title (triple ###)
- Subsubsection: #### Subsubsection Title (quadruple ####)
- Detailed points: ##### Detailed Title (five #####)

CRITICAL REQUIREMENTS (headings are written in the target output language):
- NEVER include section numbers in headings
- CORRECT: "## Introduction" / "## 서론", "### Proposed Method" / "### 제안 방법"
- INCORRECT: "## 1. Introduction", "### 2.1 제안 방법", "## Section 3"
- Never skip heading levels (e.g., going from ## directly to ####)
"""

CITATION_KEY_RULES: str = """
**NEVER expose citation keys** (e.g., smith2023transformer, brown2024attention) in prose.

**Hyperlink URLs may ONLY be copied verbatim from a URL that appears in
citation_summaries.** NEVER write a URL (arxiv.org/abs, arxiv.org/pdf, doi.org,
…) from your own memory — your recall of arXiv IDs is unreliable and produces
links to the WRONG paper (e.g. labelling a GPT-2 link with T5's id). If the
referenced paper has no URL in citation_summaries, mention it as plain text with
NO hyperlink. A wrong link is far worse than no link.

Reference prose is written in the target output language; the examples below show
English and Korean forms — follow the same pattern in whatever language you write.

When author information is clear in citation_summaries AND a URL is provided there:
- Use proper format with the url copied verbatim, e.g.
  "the Transformer architecture proposed by [Vaswani et al.](url)" /
  "[Vaswani et al.](url)에서 제안된 Transformer 아키텍처"

When author info is unclear, only a citation key is available, or NO url is provided:
- Use specific identifiers as PLAIN TEXT (no hyperlink): paper title, model name,
  algorithm name, or method name
- CORRECT (no url available): "the Transformer architecture", "BERT 모델"
- CORRECT (url in citation_summaries): "[the Transformer architecture](url)"
- INCORRECT: "[vaswani2017attention](url)", "in prior work" / "이전 연구에서",
  "according to related work" / "관련 연구에 따르면", or any [text](url) whose url
  you recalled rather than copied from citation_summaries
"""

EXCLUDED_CONTENT_RULES: str = """
ABSOLUTELY EXCLUDE (non-technical administrative content):
- Acknowledgments sections (감사의 말, Acknowledgments)
- Author contributions (저자 기여도, Author Contributions, CRediT)
- Funding information (연구비 지원, Funding, Grant information)
- Gratitude expressions (감사 표현, Thanks to...)
- Contributor lists (기여자 목록)
- References/Bibliography sections (the list itself, not in-text citations)

These sections have ZERO technical value and must be completely ignored.

**Silently** omit them: do NOT announce, describe, or comment on the exclusion.
If the current content to explain is one of these administrative sections (e.g.
a References list), output NOTHING for it — never write meta-text such as "이
부분은 참고문헌 목록이라 분석을 생략합니다" or "이 섹션은 행정적 정보입니다". The
reader must never see that a section was skipped.
"""

IMAGE_PATH_RULES: str = """
**Image paths MUST be copied EXACTLY character-by-character from the source - NO modifications whatsoever**

Path preservation rules:
- Source has local path → Use local path: ![설명](/exact/local/path.png)
- Source has HTTPS URL → Use HTTPS URL: ![설명](https://exact.url.com/path.png)
- Source has relative path → Use relative path: ![설명](./exact/relative/path.png)

FORBIDDEN modifications:
- Converting local to URL or URL to local
- Adding/removing https:// or http://
- Changing / to \\ or vice versa
- Adding/removing file extensions
- ANY character-level changes

**NEVER invent or guess an image URL.** Only insert an image whose EXACT path
appears in the provided figures/analysis. In particular, NEVER use a paper
landing/HTML page (e.g. `https://arxiv.org/html/2106.09685`) or an `/abs/` URL
as an image `src` — those are web pages, not images, and render as broken
images. If no real figure asset exists for a point, describe it in prose instead
of inserting a fabricated image link.
"""

VISUAL_DUPLICATION_RULES: str = """
**Each visual element (figure, table, code block, equation) MUST appear ONLY ONCE in the ENTIRE document**

Before inserting ANY visual element:
1. Search through ALL of previous_explanation for this image path
2. Look for any figure/table/code with similar description
3. If found → DO NOT insert again, use descriptive reference only

After first insertion, use natural descriptive references:
- CORRECT: "위 그림에서 보듯이", "앞서 보여준 표와 같이"
- INCORRECT: "그림 3에서", "표 2처럼", "Figure 1에서"

**Every inserted figure MUST be referenced in the surrounding prose.** Never drop
an image as a caption-only orphan: the sentence immediately before or after the
image must point to it, in the target output language (e.g. "The figure below
shows …" / "아래 그림은 …를 보여줍니다", "As the next figure shows …" / "다음 그림에서 보듯…").
If you cannot naturally tie a figure into the narrative, do not insert it.
"""

TABLE_RENDERING_RULES: str = """
**Tables MUST be rendered as actual markdown tables, NOT as image links**

CORRECT:
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| GPT-4 | 95.2% | 0.94 |

INCORRECT:
![표 1: 성능 비교](https://arxiv.org/html/2401.12345v1#S1.T1)
"""

# Output-style rules. Written language-agnostically so they hold for ANY target
# output language; the Korean specifics are explicitly scoped to "when the output
# language is Korean" so non-Korean runs are not given conflicting guidance.
STYLE_RULES: str = """
- Write in natural, flowing prose in the target output language, in a
  professional yet accessible academic tone.
- Prefer impersonal phrasing; avoid first-person pronouns (e.g. "we"/"우리").
  * "We applied this method" → "This method was applied"
- Never use explicit section-number references in prose.
  * INCORRECT: "in Section 3.2", "as described in Chapter 2"
  * CORRECT: "as described earlier", "the method introduced above"
- Use the target language's established technical vocabulary; keep English only
  for proper nouns or terms with no clear translation (e.g. BERT, GPT).
- WHEN THE OUTPUT LANGUAGE IS KOREAN: use the "입니다" style; avoid "우리"/"저";
  prefer translated terms ("손실 함수", "경사 하강법", "사전 학습", "역전파") and
  keep English only when a translation is unclear or for proper nouns.
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
