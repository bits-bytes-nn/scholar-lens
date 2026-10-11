"""Prompt for the five-question paper summary (PaperSummarizer)."""

from .base import (
    CITATION_KEY_RULES,
    EXCLUDED_CONTENT_RULES,
    HEADING_STRUCTURE_RULES,
    IMAGE_PATH_RULES,
    MATH_RULES,
    STYLE_RULES,
    SUMMARY_DEPTH_RULES,
    SUPPLEMENTARY_CONTEXT_RULES,
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
    output_variables: list[str] = ["subtitle", "summary", "urls"]

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
       the key idea, and the headline result with its setting
       (Korean: start with "> **요약:** "). Plain prose only — no math in this
       blockquote; the blog uses it as the post's search description.
    2. Then exactly these five `##` sections. Use the Korean headings verbatim
       when {language} is Korean; otherwise translate them, keeping emoji and
       order:
       ## 🔍 이 연구는 왜 필요합니까?
       ## 💡 어떤 해결책을 제시합니까?
       ## ⚙️ 제안한 방법은 어떻게 구현되었습니까?
       ## 📊 핵심 실험 결과는 무엇입니까?
       ## 🔮 이 연구의 의의와 한계는 무엇입니까?

    Depth per section follows the paper, not a fixed ratio:
    - Motivation: brief — the gap in prior approaches, in a few sentences.
    - Solution and implementation: where most method papers need the most
      words — the core mechanism with intuition, its key equation, the main
      architecture figure if one exists, and concrete specifics
      (hyper-parameters, algorithm steps, data). When an official codebase
      summary is provided, use it to make the implementation concrete
      (modules, defaults); never contradict the paper with it.
    - Results: the evidence for the main claim — one distilled Markdown table
      of the headline numbers when the paper has comparative results, the
      setting that produced them, and the ablation that supports the core
      claim. For benchmark or empirical papers this section carries the most
      weight.
    - Significance and limits: why it matters, at least one concrete
      limitation grounded in the paper (an assumption, an evaluation gap, a
      cost), and directions the authors or the evidence suggest.
    Skip a section only if the paper has nothing for it.
    """
        + SUMMARY_DEPTH_RULES
        + SUPPLEMENTARY_CONTEXT_RULES
        + CITATION_KEY_RULES
        + HEADING_STRUCTURE_RULES
        + VISUAL_RULES
        + IMAGE_PATH_RULES
        + MATH_RULES
        + STYLE_RULES
        + EXCLUDED_CONTENT_RULES
        + """
    Other outputs:
    - <subtitle>: the paper title rendered naturally in {language}, keeping
      model and method names (for English, a short descriptive subtitle that
      differs from the title).
    - <urls>: links the paper itself gives (code, data, project page) as
      `[text](url)`, comma-separated, URLs copied verbatim from the paper; leave
      it empty if there are none.

    Output exactly:
    <subtitle>...</subtitle>
    <summary>
    > ...

    ## 🔍 ...
    ...
    </summary>
    <urls>...</urls>
    """
    )
