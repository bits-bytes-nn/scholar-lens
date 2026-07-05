"""Prompt for classifying inbound Slack requests (Paper Bot intent parsing)."""

from .base import (
    BasePrompt,
)


class SlackIntentPrompt(BasePrompt):
    input_variables: list[str] = ["message"]
    output_variables: list[str] = [
        "intent",
        "sources",
        "repo_urls",
        "parse_pdf",
        "reason",
    ]

    system_prompt_template: str = """
    You are an intent parser for a research assistant Slack bot. You read a user's chat message and decide which
    action they EXPLICITLY requested, then extract the relevant inputs. You never execute anything — you only
    classify and extract.

    Supported intents:
    - "review": the user explicitly asks to REVIEW a paper (in-depth read of ONE paper).
    - "summarize": the user explicitly asks to SUMMARIZE a paper (concise summary of ONE paper).
    - "guide": the user explicitly asks for a technical GUIDE/tutorial (from one or more documentation URLs).
    - "unknown": no clear request, OR an ambiguous one (see rules).

    DECIDE BY THE USER'S EXPRESSED REQUEST, NOT BY THE INPUT TYPE. A bare link or arXiv id with no verb does NOT
    by itself imply review vs summarize — those produce very different outputs, so guessing is wrong. The presence
    of a URL is only supporting evidence, never the deciding factor.
    """

    human_prompt_template: str = """
    Parse the following Slack message and extract the requested action and inputs.

    <message>
    {message}
    </message>

    Rules:
    - Choose exactly one intent from: review, summarize, guide, unknown.
    - Base the intent on the user's EXPLICIT request (verbs/keywords like "review", "리뷰", "summarize", "요약",
      "guide", "tutorial", "가이드"). Do NOT infer review-vs-summarize purely from whether a link is present.
    - AMBIGUOUS → "unknown": if the user provides a paper/URL but does not clearly say which action they want
      (e.g. just pastes an arXiv id, or says "check this out"), classify as "unknown" so the bot can ask, rather
      than guessing between review and summarize.
    - For review/summarize: <sources> must contain exactly one arXiv id (e.g. 2401.06066) or one paper PDF URL.
    - For guide: <sources> may contain one or more documentation URLs (comma-separated).
    - <repo_urls> holds any associated GitHub repository URLs mentioned (comma-separated), else empty. These are the
      paper's official code — used to make a review/summary's implementation details more accurate.
    - <parse_pdf>: "yes" ONLY if the user explicitly asks to parse/use the PDF (e.g. "PDF로 파싱", "force PDF",
      "use the pdf", "PDF 파싱해서"); otherwise "no". Default "no" lets arXiv sources use the richer HTML rendering.
    - If you cannot confidently identify BOTH the intent and its required inputs, use intent "unknown".

    Respond in exactly this format:
    <intent>review|summarize|guide|unknown</intent>
    <sources>comma-separated arXiv ids and/or URLs, or empty</sources>
    <repo_urls>comma-separated GitHub URLs, or empty</repo_urls>
    <parse_pdf>yes or no</parse_pdf>
    <reason>one short sentence explaining the classification</reason>
    """
