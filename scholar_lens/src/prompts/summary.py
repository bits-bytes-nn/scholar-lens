"""Prompt for the five-question paper summary (PaperSummarizer)."""

from .base import (
    CITATION_KEY_RULES,
    EXCLUDED_CONTENT_RULES,
    HEADING_STRUCTURE_RULES,
    IMAGE_PATH_RULES,
    MATH_RULES,
    PAPER_TABLE_RULES,
    STYLE_RULES,
    SUMMARY_DEPTH_RULES,
    SUPPLEMENTARY_CONTEXT_RULES,
    TABLE_RULES,
    VISUAL_RULES,
    BasePrompt,
)


class PaperSummaryPrompt(BasePrompt):
    input_variables: list[str] = [
        "content",
        "codebase_summary",
        "language",
        "translation_guideline",
    ]
    output_variables: list[str] = ["subtitle", "description", "summary", "urls"]

    system_prompt_template: str = """
    You are an AI/ML researcher who writes concise but substantive paper
    summaries for a technical blog. A reader should finish one in five minutes
    and still understand the core mechanism, the evidence for it, and its
    limits. You are precise, explain the hard part well, and never pad.
    """

    human_prompt_template: str = (
        """
    Summarize the paper below in {language}.

    <paper>
    {content}
    </paper>

    <official_codebase_summary>
    {codebase_summary}
    </official_codebase_summary>

    <translation_guideline>
    {translation_guideline}
    </translation_guideline>

    Structure (Markdown inside <summary>):
    1. Open with a blockquote of 2-3 lines in {language}: what the paper does,
       the key idea, and the headline result with its setting. It is published
       as the post's `## TL;DR` paragraph: plain prose only, no math, no
       label, and every number one the paper states, as the paper states it.
    2. Then exactly these five `##` sections (a Korean post uses the headings
       verbatim; any other language translates them, keeping emoji and order):
       ## 🔍 이 연구는 왜 필요합니까?
       ## 💡 어떤 해결책을 제시합니까?
       ## ⚙️ 제안한 방법은 어떻게 구현되었습니까?
       ## 📊 핵심 실험 결과는 무엇입니까?
       ## 🔮 이 연구의 의의와 한계는 무엇입니까?

    Depth per section follows the paper, not a fixed ratio:
    - Motivation: brief — the gap in prior approaches, in a few sentences.
    - Solution and implementation: where most method papers need the most
      words — the core mechanism with intuition and its key equation. When an
      official codebase summary is provided, use it to make the
      implementation concrete (modules, defaults); never contradict the paper
      with it. Once the key equation is defined, every later mention (a
      callout included) uses that same form.
    - Results: the evidence for the main claim, with the 1-2 settings needed
      to read it. For benchmark or empirical
      papers this section carries the most weight.
    - Significance and limits: why it matters and at least one concrete
      limitation grounded in the paper (an assumption, an evaluation gap, a
      cost).
    Skip a section only if the paper has nothing for it.
    """
        + SUMMARY_DEPTH_RULES
        + SUPPLEMENTARY_CONTEXT_RULES
        + CITATION_KEY_RULES
        + HEADING_STRUCTURE_RULES
        + VISUAL_RULES
        + TABLE_RULES
        + PAPER_TABLE_RULES
        + IMAGE_PATH_RULES
        + MATH_RULES
        + STYLE_RULES
        + EXCLUDED_CONTENT_RULES
        + """
    Other outputs:
    - <subtitle>: the paper title translated into {language}, keeping model
      and method names (if {language} is the title's own language, or the
      translation would equal the title, a short descriptive subtitle instead).
    - <description>: the post's search snippet, one plain text paragraph of
      120-160 characters in {language}: no math, Markdown, links or tags. It
      opens with the model or method name as the paper writes it ("Mistral
      7B는 …"), never with a generic opener ("대규모 언어 모델의 발전으로 …"),
      and says what the paper does and its headline result. Numbers only from
      the opening blockquote or the paper's abstract.
    - <urls>: links the paper itself gives (code, data, project page) as
      `[text](url)`, comma-separated, URLs copied verbatim from the paper; leave
      it empty if there are none.

    Output exactly:
    <subtitle>...</subtitle>
    <description>...</description>
    <summary>
    > ...

    ## 🔍 ...
    ...
    </summary>
    <urls>...</urls>
    """
    )


class PaperSummaryShortenPrompt(BasePrompt):
    """One bounded pass that cuts an over-long summary down to size."""

    input_variables: list[str] = ["summary", "max_hangul", "language"]
    output_variables: list[str] = ["summary"]

    system_prompt_template: str = """
    You are a technical editor who shortens blog summaries without losing what
    matters: the core mechanism, its key equation and the headline result.
    """

    human_prompt_template: str = """
    The summary below is too long. Shorten its prose to at most {max_hangul}
    Hangul syllables. Cut secondary details and list items first, then trim
    sentences. Keep unchanged: the opening blockquote, every `##` heading,
    every image line, every $$ math expression, the results table and all
    numbers. Do not add anything new. Keep the {language} language, register
    and Markdown; a sentence you rewrite stays natural, never translationese.

    <summary>
    {summary}
    </summary>

    Output exactly:
    <summary>
    the shortened summary
    </summary>
    """
