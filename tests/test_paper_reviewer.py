"""Orchestration of the review workflow with fake LLM chains (no AWS/network).

``PaperReviewer.__init__`` builds Bedrock chains and loads NLTK data, so tests
construct it with ``__new__`` and set the attributes the workflow reads. The
fakes record every payload so the tests can check what each stage was given.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from scholar_lens.src.citation_summarizer import CitationSummarizer
from scholar_lens.src.content_extractor import Attributes, Citation
from scholar_lens.src.explainer import Paper, PaperReviewer, ReviewText
from scholar_lens.src.metrics import TokenBudgetExceeded, TokenUsageTracker
from scholar_lens.src.parser import Content
from scholar_lens.src.post_validator import body_errors
from scholar_lens.src.review_plan import (
    PlanError,
    SourceText,
    draft_rank,
    publish_blockers,
    rule_issues,
)

MARKER = '[Image: alt="arch", src="/p/figures/0.png", caption="The architecture."]'
SENTENCES = [
    "Full fine-tuning is costly.",
    "Adapters add latency.",
    f"We propose LoRA {MARKER}.",
    "The update is W0 + BA.",
    "B starts at zero.",
    "Results on GLUE match full fine-tuning.",
    "Rank 4 suffices.",
    "We thank our sponsors.",
]


class FakeChain:
    """Returns queued responses (or a function of the payload) and records
    each payload."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> Any:
        self.calls.append(payload)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return item(payload) if callable(item) else item


class FakeFactory:
    def fit_text(self, model_id: Any, text: str, **_: Any) -> str:
        return text


def _plan(**overrides: Any) -> str:
    plan = {
        "glossary": {"low-rank adaptation": "저랭크 적응"},
        "excluded_ranges": [[7, 7]],
        "sections": [
            {
                "title": "문제와 동기",
                "sentence_ranges": [[0, 1]],
                "must_cover": ["full fine-tuning cost", "adapter latency"],
                "difficulty": "easy",
                "importance": "context",
                "owned_concepts": ["adapter latency"],
            },
            {
                "title": "LoRA의 핵심 아이디어",
                "sentence_ranges": [[2, 4]],
                "must_cover": ["update rule W0 + BA", "zero init of B"],
                "difficulty": "hard",
                "importance": "core",
                "owned_concepts": ["low-rank update"],
                "figures": ["F1"],
                "prerequisites": [{"concept": "matrix rank", "reference": "R1"}],
                "needs_code": True,
            },
            {
                "title": "실험 결과",
                "sentence_ranges": [[5, 6]],
                "must_cover": ["GLUE parity", "rank 4"],
                "difficulty": "medium",
                "importance": "supporting",
            },
        ],
    }
    plan.update(overrides)
    return f"<plan>{json.dumps(plan, ensure_ascii=False)}</plan>"


def _writer(payload: dict[str, Any]) -> str:
    title = payload["section_brief"].splitlines()[0].split(": ", 1)[1]
    return f"<explanation>## {title}\n\n{title} 본문입니다.</explanation><has_more>n</has_more>"


def _pass(_: dict[str, Any]) -> dict[str, str]:
    return {"quality_score": "88", "missing_items": "none", "improvement_feedback": ""}


TLDR = "\n\n".join(f"### 질문 {i}\n답입니다." for i in range(1, 5))


def _final(_: dict[str, Any]) -> dict[str, str]:
    return {
        "subtitle": "LoRA: 저랭크 적응",
        "key_takeaways": TLDR,
        "closing": "## 한계와 시사점\n\n한계입니다.",
    }


def _no_edits(_: dict[str, Any]) -> str:
    return "<missing_coverage>none</missing_coverage>"


def _reviewer(
    *,
    planner: FakeChain | None = None,
    writer: FakeChain | None = None,
    checker: FakeChain | None = None,
    editor: FakeChain | None = None,
    finalizer: FakeChain | None = None,
    citations: list[Citation] | None = None,
    code_retriever: Any = None,
    max_edit_rounds: int = 2,
    max_synthesis_attempts: int = 3,
) -> PaperReviewer:
    r = PaperReviewer.__new__(PaperReviewer)
    r.paper = Paper(
        arxiv_id="2106.09685",
        title="LoRA: Low-Rank Adaptation of Large Language Models",
        published=datetime(2021, 6, 17),
        pdf_url="https://arxiv.org/pdf/2106.09685",
        content=Content(text=" ".join(SENTENCES)),
        attributes=Attributes(affiliation="Microsoft", category="Finetuning"),
        citations=citations
        or [Citation(authors="Eckart", title="Matrix approximation", arxiv_id=None)],
    )
    r.language = "Korean"
    r.callbacks = []
    r._token_tracker = None
    r.max_total_tokens = None
    r.notes = []
    r.open_defects = []
    r.open_gaps = []
    r.citation_summarizer = AsyncMock(spec=CitationSummarizer)
    r.citation_summarizer.summarize.return_value = ["[R] Rank summary."]
    r.code_retriever = code_retriever
    r.translation_guideline = []
    r.min_quality_score = 70
    r.max_synthesis_attempts = max_synthesis_attempts
    r.max_sections = 20
    r.max_edit_rounds = max_edit_rounds
    r.max_code_results = 4
    r.planning_model_id = "planner"  # type: ignore[assignment]
    r.llm_factory = FakeFactory()  # type: ignore[assignment]
    r.planner = planner or FakeChain(_plan())  # type: ignore[assignment]
    r.writer = writer or FakeChain(_writer)  # type: ignore[assignment]
    r.checker = checker or FakeChain(_pass)  # type: ignore[assignment]
    r.editor = editor or FakeChain(_no_edits)  # type: ignore[assignment]
    r.finalizer = finalizer or FakeChain(_final)  # type: ignore[assignment]
    r._source = lambda: SourceText.from_sentences(SENTENCES)  # type: ignore[method-assign]
    return r


async def test_end_to_end_produces_review_text() -> None:
    reviewer = _reviewer()
    review = await reviewer.run()
    assert isinstance(review, ReviewText)
    assert review.body.count("\n## ") == 2 and review.body.startswith("## 문제와 동기")
    assert review.subtitle == "LoRA: 저랭크 적응"
    assert review.closing.startswith("## 한계와 시사점")
    # plan, 3 sections × (write + check), one edit round, finalize.
    assert len(reviewer.planner.calls) == 1  # type: ignore[attr-defined]
    assert len(reviewer.writer.calls) == 3  # type: ignore[attr-defined]
    assert len(reviewer.editor.calls) == 1  # type: ignore[attr-defined]


async def test_writer_gets_brief_ledger_glossary_and_only_its_figures() -> None:
    reviewer = _reviewer()
    await reviewer.run()
    calls = reviewer.writer.calls  # type: ignore[attr-defined]
    first, second, third = calls
    assert "full fine-tuning cost" in first["section_brief"]
    assert "저랭크 적응" in second["glossary"]
    assert "earlier, '문제와 동기': adapter latency" in second["owned_elsewhere"]
    assert "later, 'LoRA의 핵심 아이디어': low-rank update" in first["owned_elsewhere"]
    assert MARKER in second["source_text"] and MARKER in second["section_brief"]
    assert MARKER not in first["source_text"]
    # The article so far flows into the next section.
    assert "문제와 동기 본문입니다." in second["previous_sections"]
    assert "LoRA의 핵심 아이디어 본문입니다." in third["previous_sections"]
    # Excluded acknowledgments never reach a writer.
    assert all("sponsors" not in c["source_text"] for c in calls)


async def test_enrichment_follows_the_plan() -> None:
    code = AsyncMock()
    code.search_similar_code.return_value = [
        {"content": "def lora(): ...", "source": "a.py"}
    ]
    reviewer = _reviewer(code_retriever=code)
    await reviewer.run()
    reviewer.citation_summarizer.summarize.assert_awaited_once()  # type: ignore[attr-defined]
    identifiers = reviewer.citation_summarizer.summarize.await_args.args[0]  # type: ignore[attr-defined]
    assert identifiers == ["Matrix approximation"]
    code.search_similar_code.assert_awaited_once()  # only the needs_code section
    second = reviewer.writer.calls[1]  # type: ignore[attr-defined]
    assert "Rank summary" in second["citation_summaries"]
    assert "def lora" in second["code"]


async def test_failed_citation_summaries_are_dropped() -> None:
    reviewer = _reviewer()
    reviewer.citation_summarizer.summarize.return_value = [  # type: ignore[attr-defined]
        f"[R1]\n{CitationSummarizer.FAILURE_STRING}"
    ]
    await reviewer.run()
    assert reviewer.writer.calls[1]["citation_summaries"] == "- none"  # type: ignore[attr-defined]


async def test_missing_items_trigger_revision_from_best_draft() -> None:
    def check(payload: dict[str, Any]) -> dict[str, str]:
        if (
            "LoRA의 핵심" in payload["section_brief"].splitlines()[0]
            and "개정" not in payload["draft"]
        ):
            return {
                "quality_score": "85",
                "missing_items": "- zero init of B",
                "improvement_feedback": "B의 초기화를 설명하세요.",
            }
        return _pass(payload)

    def write(payload: dict[str, Any]) -> str:
        text = _writer(payload)
        return (
            text.replace("본문입니다.", "개정 본문입니다.")
            if payload["draft_to_revise"]
            else text
        )

    reviewer = _reviewer(writer=FakeChain(write), checker=FakeChain(check))
    review = await reviewer.run()
    revision = [c for c in reviewer.writer.calls if c["draft_to_revise"]]  # type: ignore[attr-defined]
    assert len(revision) == 1
    assert "zero init of B" in revision[0]["improvement_feedback"]
    assert "LoRA의 핵심 아이디어 본문입니다." in revision[0]["draft_to_revise"]
    assert "LoRA의 핵심 아이디어 개정 본문입니다." in review.body


async def test_best_draft_kept_when_attempts_run_out() -> None:
    scores = iter(["60", "40", "50"] + ["90"] * 10)

    def check(_: dict[str, Any]) -> dict[str, str]:
        return {
            "quality_score": next(scores),
            "missing_items": "none",
            "improvement_feedback": "x",
        }

    drafts = iter(["A", "B", "C"] + ["D"] * 10)

    def write(payload: dict[str, Any]) -> str:
        return f"<explanation>## t\n\n초안 {next(drafts)}</explanation><has_more>n</has_more>"

    reviewer = _reviewer(writer=FakeChain(write), checker=FakeChain(check))
    review = await reviewer.run()
    assert review.body.startswith("## t\n\n초안 A")  # 60 beat 40 and 50


async def test_a_rule_slip_never_loses_to_an_invented_low_score_draft() -> None:
    # 85 with one numbered-reference slip beats a 40 with no blocking defect.
    replies = iter(
        ["충실한 설명입니다. Table 2와 같습니다.", "지어낸 결과 99.9%입니다."]
    )

    def write(payload: dict[str, Any]) -> str:
        text = _writer(payload)
        if "LoRA의 핵심" not in payload["section_brief"]:
            return text
        return text.replace("본문입니다.", next(replies, "지어낸 결과 2입니다."))

    def check(payload: dict[str, Any]) -> dict[str, str]:
        if "충실한" in payload["draft"]:
            return {"quality_score": "85", "missing_items": "none"}
        if "지어낸" in payload["draft"]:
            return {"quality_score": "40", "missing_items": "none"}
        return _pass(payload)

    reviewer = _reviewer(writer=FakeChain(write), checker=FakeChain(check))
    review = await reviewer.run()
    assert "충실한 설명입니다." in review.body
    assert "99.9%" not in review.body


async def test_continuations_are_joined() -> None:
    def write(payload: dict[str, Any]) -> str:
        if not payload["current_explanation"]:
            return "<explanation>## a\n\n첫 부분</explanation><has_more>y</has_more>"
        return "<explanation>둘째 부분</explanation><has_more>n</has_more>"

    reviewer = _reviewer(writer=FakeChain(write))
    review = await reviewer.run()
    assert "첫 부분\n\n둘째 부분" in review.body


async def test_unbalanced_writer_tags_are_stripped() -> None:
    reviewer = _reviewer(writer=FakeChain("<explanation>## x\n\n잘린 본문"))
    review = await reviewer.run()
    assert "<explanation>" not in review.body and "잘린 본문" in review.body


def test_an_unclosed_has_more_tag_does_not_leak_its_value() -> None:
    from scholar_lens.src.explainer import _split_writer_output

    assert _split_writer_output("충분합니다.<has_more>n") == ("충분합니다.", False)


async def test_empty_draft_keeps_the_checker_feedback() -> None:
    drafts = iter(["초안 A", "", "초안 B"] + ["초안 C"] * 10)
    scores = iter(["60"] + ["90"] * 10)

    def write(payload: dict[str, Any]) -> str:
        draft = next(drafts)
        return (
            f"<explanation>## t\n\n{draft}</explanation><has_more>n</has_more>"
            if draft
            else ""
        )

    def check(_: dict[str, Any]) -> dict[str, str]:
        return {
            "quality_score": next(scores),
            "missing_items": "none",
            "improvement_feedback": "예시를 더합니다.",
        }

    reviewer = _reviewer(writer=FakeChain(write), checker=FakeChain(check))
    await reviewer.run()
    after_empty = reviewer.writer.calls[2]  # type: ignore[attr-defined]
    assert "예시를 더합니다." in after_empty["improvement_feedback"]
    assert "empty" in after_empty["improvement_feedback"]


async def test_plan_repair_then_failure() -> None:
    reviewer = _reviewer(planner=FakeChain("not a plan", _plan()))
    await reviewer.run()
    second = reviewer.planner.calls[1]  # type: ignore[attr-defined]
    assert "could not be used" in second["repair_note"]

    broken = _reviewer(planner=FakeChain("not a plan"))
    with pytest.raises(PlanError):
        await broken.run()


async def test_edit_rounds_apply_edits_and_stop_without_gaps() -> None:
    def edit(payload: dict[str, Any]) -> str:
        if not payload["round_note"]:
            return (
                "<edit><find>실험 결과 본문입니다.</find>"
                "<replace>GLUE 결과 본문입니다.</replace></edit>"
                "<missing_coverage>- rank 4</missing_coverage>"
            )
        assert "rank 4" in payload["round_note"]
        return (
            "<edit><find>GLUE 결과 본문입니다.</find>"
            "<replace>GLUE 결과와 랭크 4 본문입니다.</replace></edit>"
            "<missing_coverage>none</missing_coverage>"
        )

    reviewer = _reviewer(editor=FakeChain(edit))
    review = await reviewer.run()
    assert len(reviewer.editor.calls) == 2  # type: ignore[attr-defined]
    assert "GLUE 결과와 랭크 4 본문입니다." in review.body
    assert reviewer.finalizer.calls[0]["explanation"] == review.body  # type: ignore[attr-defined]


async def test_skipped_edits_are_fed_into_the_next_round() -> None:
    def edit(payload: dict[str, Any]) -> str:
        if not payload["round_note"]:
            return (
                "<edit><find>없는 문장</find><replace>x</replace></edit>"
                "<missing_coverage>none</missing_coverage>"
            )
        assert "could not be applied" in payload["round_note"]
        return "<missing_coverage>none</missing_coverage>"

    reviewer = _reviewer(editor=FakeChain(edit))
    await reviewer.run()
    assert len(reviewer.editor.calls) == 2  # type: ignore[attr-defined]


async def test_edit_disabled() -> None:
    reviewer = _reviewer(max_edit_rounds=0)
    await reviewer.run()
    assert reviewer.editor.calls == []  # type: ignore[attr-defined]


async def test_budget_exhaustion_stops_before_the_next_call() -> None:
    reviewer = _reviewer()
    reviewer._token_tracker = TokenUsageTracker(input_tokens=900, output_tokens=200)
    reviewer.max_total_tokens = 1000
    with pytest.raises(TokenBudgetExceeded):
        await reviewer.run()
    assert reviewer.planner.calls == []  # type: ignore[attr-defined]


async def test_coverage_plan_lists_every_must_cover_item() -> None:
    reviewer = _reviewer()
    await reviewer.run()
    coverage = reviewer.editor.calls[0]["coverage_plan"]  # type: ignore[attr-defined]
    for item in ("full fine-tuning cost", "zero init of B", "rank 4"):
        assert item in coverage


async def test_writer_and_checker_see_what_the_article_already_shows() -> None:
    def write(payload: dict[str, Any]) -> str:
        title = payload["section_brief"].splitlines()[0].split(": ", 1)[1]
        body = f"## {title}\n\n본문입니다.\n\n$$h = W_0 x + BAx$$\n"
        return f"<explanation>{body}</explanation><has_more>n</has_more>"

    checker = FakeChain(_pass)
    reviewer = _reviewer(writer=FakeChain(write), checker=checker)
    await reviewer.run()
    second_writer = reviewer.writer.calls[1]  # type: ignore[attr-defined]
    assert "equation: $$h = W_0 x + BAx$$" in second_writer["already_in_article"]
    assert "heading: ## 문제와 동기" in checker.calls[1]["already_in_article"]


async def test_missing_source_items_do_not_force_revision() -> None:
    def write(payload: dict[str, Any]) -> str:
        return _writer(payload) + "<missing_source>rank 4</missing_source>"

    def check(payload: dict[str, Any]) -> dict[str, str]:
        # A checker following its prompt: an item the writer reported as absent
        # from the source is not missing and does not cap the score.
        if "실험 결과" in payload["section_brief"].splitlines()[0]:
            assert "rank 4" in payload["unsupported_items"]
            return {
                "quality_score": "88",
                "missing_items": "",
                "improvement_feedback": "",
            }
        return _pass(payload)

    reviewer = _reviewer(writer=FakeChain(write), checker=FakeChain(check))
    await reviewer.run()
    assert not any(c["draft_to_revise"] for c in reviewer.writer.calls)  # type: ignore[attr-defined]


async def test_a_writer_claim_cannot_excuse_a_checker_missing_item() -> None:
    def write(payload: dict[str, Any]) -> str:
        return _writer(payload) + "<missing_source>실험</missing_source>"

    def check(payload: dict[str, Any]) -> dict[str, str]:
        if "실험 결과" in payload["section_brief"].splitlines()[0]:
            return {
                "quality_score": "88",
                "missing_items": "- 실험 결과 표",
                "improvement_feedback": "",
            }
        return _pass(payload)

    reviewer = _reviewer(writer=FakeChain(write), checker=FakeChain(check))
    await reviewer.run()
    assert any(c["draft_to_revise"] for c in reviewer.writer.calls)  # type: ignore[attr-defined]


async def test_continuation_does_not_repeat_the_heading() -> None:
    def write(payload: dict[str, Any]) -> str:
        title = payload["section_brief"].splitlines()[0].split(": ", 1)[1]
        if not payload["current_explanation"]:
            return (
                f"<explanation>## {title}\n\n앞부분</explanation><has_more>y</has_more>"
            )
        return f"<explanation>## {title}\n\n뒷부분</explanation><has_more>n</has_more>"

    reviewer = _reviewer(writer=FakeChain(write))
    review = await reviewer.run()
    assert review.body.count("## 문제와 동기") == 1
    assert "앞부분\n\n뒷부분" in review.body


async def test_writer_output_entities_are_unescaped_and_code_is_fenced() -> None:
    code = AsyncMock()
    code.search_similar_code.return_value = [
        {
            "content": "def lora():\n    return 1",
            "source": "loralib/layers.py",
            "similarity": 0.1,
        }
    ]

    def write(payload: dict[str, Any]) -> str:
        return _writer(payload).replace("본문입니다.", "A &amp; B 본문입니다.")

    reviewer = _reviewer(writer=FakeChain(write), code_retriever=code)
    review = await reviewer.run()
    assert "A & B" in review.body and "&amp;" not in review.body
    second = reviewer.writer.calls[1]  # type: ignore[attr-defined]
    assert "# source: loralib/layers.py\n```python\ndef lora():" in second["code"]


async def test_finalizer_gets_glossary_and_closing_points_and_is_repaired() -> None:
    answers = iter(
        [
            {"subtitle": "s", "key_takeaways": "## TL;DR\n### 하나\n답", "closing": ""},
            {
                "subtitle": "s",
                "key_takeaways": "## TL;DR\n" + TLDR,
                "closing": "## 한계와 시사점\n\nx",
            },
        ]
    )
    finalizer = FakeChain(lambda _: next(answers))
    reviewer = _reviewer(
        planner=FakeChain(_plan(closing_points=["no batching across tasks"])),
        finalizer=finalizer,
    )
    review = await reviewer.run()
    first, second = finalizer.calls
    assert "저랭크 적응" in first["glossary"]
    assert "no batching across tasks" in first["closing_points"]
    assert "four ### questions" in second["repair_note"]
    assert review.key_takeaways.startswith("### 질문 1")


async def test_must_fix_blocks_a_passing_score_until_fixed() -> None:
    def check(payload: dict[str, Any]) -> dict[str, str]:
        result = _pass(payload)
        if "fine-tune합니다" in payload["draft"]:
            result["must_fix"] = "- 'fine-tune합니다' → '미세 조정합니다'"
        return result

    def write(payload: dict[str, Any]) -> str:
        text = _writer(payload)
        if "LoRA의 핵심" in payload["section_brief"] and not payload["draft_to_revise"]:
            return text.replace("본문입니다.", "모델을 fine-tune합니다.")
        return text

    reviewer = _reviewer(writer=FakeChain(write), checker=FakeChain(check))
    review = await reviewer.run()
    revision = next(
        c for c in reviewer.writer.calls if c["draft_to_revise"]  # type: ignore[attr-defined]
    )
    assert "Blocking defects" in revision["improvement_feedback"]
    assert "fine-tune합니다" not in review.body
    assert review.notes == ()


async def test_a_passing_revision_beats_a_higher_scoring_blocked_draft() -> None:
    def check(payload: dict[str, Any]) -> dict[str, str]:
        if "fine-tune합니다" in payload["draft"]:
            return {
                "quality_score": "95",
                "missing_items": "none",
                "must_fix": "- 'fine-tune합니다' → '파인튜닝합니다'",
            }
        return {"quality_score": "80", "missing_items": "none"}

    def write(payload: dict[str, Any]) -> str:
        text = _writer(payload)
        if "LoRA의 핵심" in payload["section_brief"] and not payload["draft_to_revise"]:
            return text.replace("본문입니다.", "모델을 fine-tune합니다.")
        return text

    reviewer = _reviewer(writer=FakeChain(write), checker=FakeChain(check))
    review = await reviewer.run()
    assert "fine-tune합니다" not in review.body
    assert review.notes == ()


async def test_rule_slips_block_the_section_and_degraded_output_is_noted() -> None:
    def write(payload: dict[str, Any]) -> str:
        return _writer(payload).replace("본문입니다.", "이 절에서는 본문을 봅니다.")

    reviewer = _reviewer(
        writer=FakeChain(write),
        finalizer=FakeChain({"subtitle": "s", "key_takeaways": "", "closing": ""}),
        max_synthesis_attempts=2,
    )
    review = await reviewer.run()
    revision = reviewer.writer.calls[1]  # type: ignore[attr-defined]
    assert "self-reference '이 절'" in revision["improvement_feedback"]
    assert sum("open must-fix/missing item" in n for n in review.notes) == 3
    assert any("finalization still malformed" in n for n in review.notes)


async def test_a_worse_revision_is_named_in_the_next_feedback() -> None:
    scores = iter(["60", "40", "75"])

    def check(_: dict[str, Any]) -> dict[str, str]:
        return {
            "quality_score": next(scores, "75"),
            "missing_items": "none",
            "improvement_feedback": "고칠 점",
        }

    reviewer = _reviewer(checker=FakeChain(check))
    await reviewer.run()
    third = reviewer.writer.calls[2]  # type: ignore[attr-defined]
    assert "rejected (score 40" in third["improvement_feedback"]


async def test_finalization_repair_keeps_parts_that_were_already_valid() -> None:
    answers = iter(
        [
            {"subtitle": "좋은 부제", "key_takeaways": TLDR, "closing": "no heading"},
            {
                "subtitle": "",
                "key_takeaways": "### 하나",
                "closing": "## 한계와 시사점\n\n평가입니다.",
            },
        ]
    )
    reviewer = _reviewer(finalizer=FakeChain(lambda _: next(answers)))
    review = await reviewer.run()
    assert review.subtitle == "좋은 부제"
    assert review.key_takeaways == TLDR
    assert review.closing.startswith("## 한계와 시사점")
    assert review.notes == ()


@pytest.mark.parametrize(
    "first,second,part",
    [
        # A usable TL;DR with too few questions is not replaced by an empty one.
        (
            {
                "subtitle": "부제",
                "key_takeaways": "### 하나\n답",
                "closing": "## 한계와 시사점\n\n평가",
            },
            {
                "subtitle": "부제",
                "key_takeaways": "",
                "closing": "## 한계와 시사점\n\n평가",
            },
            "key_takeaways",
        ),
        # A closing missing its heading is not replaced by an empty one.
        (
            {"subtitle": "부제", "key_takeaways": TLDR, "closing": "한계입니다."},
            {"subtitle": "부제", "key_takeaways": TLDR, "closing": ""},
            "closing",
        ),
    ],
)
async def test_finalization_repair_never_swaps_in_an_empty_part(
    first: dict[str, str], second: dict[str, str], part: str
) -> None:
    answers = iter([first, second])
    reviewer = _reviewer(finalizer=FakeChain(lambda _: next(answers)))
    review = await reviewer.run()
    assert getattr(review, part) == first[part].strip()


def test_nltk_data_is_used_when_present_and_a_failed_download_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scholar_lens.src import explainer

    download = MagicMock(return_value=True)
    monkeypatch.setattr(explainer.nltk, "download", download)
    monkeypatch.setattr(explainer.nltk.data, "find", lambda _: "found")
    explainer._ensure_nltk_data()
    download.assert_not_called()

    def missing(_: str) -> None:
        raise LookupError

    monkeypatch.setattr(explainer.nltk.data, "find", missing)
    download.return_value = False
    with pytest.raises(RuntimeError, match="punkt_tab"):
        explainer._ensure_nltk_data()


def test_finalization_problems_are_structural_only() -> None:
    from scholar_lens.src.explainer import _finalization_problems

    long_closing = "## 한계와 시사점\n\n" + "한계를 설명합니다. " * 120
    final = {"subtitle": "s", "key_takeaways": TLDR, "closing": long_closing}
    assert _finalization_problems(final, "Title") == {}
    final["closing"] = "한계입니다."
    assert set(_finalization_problems(final, "Title")) == {"closing"}


def test_subtitle_that_repeats_the_title_is_a_problem() -> None:
    # A bare model name ("Pixtral 12B") "translates" to itself; the repair
    # round must ask for a descriptive subtitle instead.
    from scholar_lens.src.explainer import _finalization_problems

    final = {
        "subtitle": "pixtral  12B",
        "key_takeaways": TLDR,
        "closing": "## 한계\n\n끝.",
    }
    assert set(_finalization_problems(final, "Pixtral 12B")) == {"subtitle"}


async def test_revisions_stop_when_must_fix_items_do_not_shrink() -> None:
    # Logged DPO pattern: over the score bar, nothing missing, the same number
    # of subjective must-fix items on every attempt (2 → 2 → 2).
    def check(_: dict[str, Any]) -> dict[str, str]:
        return {
            "quality_score": "80",
            "missing_items": "none",
            "must_fix": "- 'A' 표현을 바꿉니다\n- 'B' 근거를 보강합니다",
        }

    reviewer = _reviewer(checker=FakeChain(check), max_synthesis_attempts=3)
    review = await reviewer.run()
    sections = len(json.loads(_plan()[6:-7])["sections"])
    assert len(reviewer.checker.calls) == 2 * sections  # type: ignore[attr-defined]
    note = next(n for n in review.notes if "open must-fix/missing item" in n)
    assert "Open: 'A' 표현을 바꿉니다" in note


async def test_shrinking_must_fix_items_keep_revising() -> None:
    counts = iter([3, 2, 1, 0] * 10)

    def check(_: dict[str, Any]) -> dict[str, str]:
        n = next(counts)
        items = "\n".join(f"- 항목 {i}" for i in range(n)) or "none"
        return {"quality_score": "80", "missing_items": "none", "must_fix": items}

    reviewer = _reviewer(checker=FakeChain(check), max_synthesis_attempts=3)
    await reviewer.run()
    # 3 → 2 → 1: every revision made progress, so all three attempts run.
    assert len(reviewer.checker.calls) >= 3  # type: ignore[attr-defined]


BAD_TABLE = "\n\n| 방법 | | 정확도 |\n| --- | --- | --- |\n| LoRA | 1 | 90.1 |\n"
GOOD_TABLE = "\n\n| 방법 | 랭크 | 정확도 |\n| --- | --- | --- |\n| LoRA | 1 | 90.1 |\n"
TARGET = "LoRA의 핵심"


def _with(payload: dict[str, Any], extra: str) -> str:
    return _writer(payload).replace("</explanation>", extra + "</explanation>")


class TestRuleIssuesAfterPublisherFixes:
    @pytest.mark.parametrize(
        "text",
        [
            "손실은 다음과 같습니다.\n$$\nh = W_0 x\n$$\n이 식입니다.",
            "갱신 \\\\(x\\_t\\\\)를 씁니다.",
            "변환은 다음과 같습니다.\n\\\\[ y = f(x) \\\\]\n끝입니다.",
        ],
    )
    def test_math_the_publisher_fixes_is_not_an_issue(self, text: str) -> None:
        assert body_errors(text)  # the raw draft would fail ...
        assert rule_issues(text) == []  # ... but not after normalization

    def test_real_table_error_still_counts(self) -> None:
        assert publish_blockers(BAD_TABLE.strip())

    async def test_autofixable_math_spends_no_revision(self) -> None:
        calls = {"n": 0}

        def write(payload: dict[str, Any]) -> str:
            if TARGET in payload["section_brief"]:
                calls["n"] += 1
                return _with(payload, "\n손실입니다.\n$$\nh = W_0 x\n$$\n핵심입니다.")
            return _writer(payload)

        review = await _reviewer(
            writer=FakeChain(write), checker=FakeChain(_pass)
        ).run()
        assert calls["n"] == 1
        assert not any("quality bar" in note for note in review.notes)


class TestPublishableDraftWins:
    def test_rank_puts_publishable_before_score(self) -> None:
        broken = draft_rank(passes=False, publishable=False, blocking=1, score=88)
        clean = draft_rank(passes=False, publishable=True, blocking=1, score=85)
        assert clean > broken

    async def test_lower_scoring_publishable_revision_is_kept(self) -> None:
        calls = {"n": 0}

        def write(payload: dict[str, Any]) -> str:
            if TARGET in payload["section_brief"]:
                calls["n"] += 1
                return _with(payload, BAD_TABLE if calls["n"] == 1 else GOOD_TABLE)
            return _writer(payload)

        def check(payload: dict[str, Any]) -> dict[str, str]:
            if "| 방법 |" not in payload["draft"]:
                return _pass(payload)
            score = "88" if "| 방법 | |" in payload["draft"] else "85"
            return {
                "quality_score": score,
                "missing_items": "none",
                "must_fix": "- '본문입니다' → 더 구체적으로",
            }

        review = await _reviewer(
            writer=FakeChain(write), checker=FakeChain(check)
        ).run()
        assert "| 방법 | | 정확도 |" not in review.body
        assert body_errors(review.body) == []

    async def test_no_stall_while_a_publish_blocker_remains(self) -> None:
        calls = {"n": 0}

        def write(payload: dict[str, Any]) -> str:
            if TARGET in payload["section_brief"]:
                calls["n"] += 1
                return _with(payload, GOOD_TABLE if calls["n"] >= 3 else BAD_TABLE)
            return _writer(payload)

        review = await _reviewer(
            writer=FakeChain(write), checker=FakeChain(_pass)
        ).run()
        assert calls["n"] == 3
        assert "| 방법 | | 정확도 |" not in review.body


def test_finalization_part_that_the_validator_would_reject_is_a_problem() -> None:
    from scholar_lens.src.explainer import _finalization_problems

    closing = "## 한계와 시사점\n\n| 방법 | |\n| --- | --- |\n| A | 1 |\n"
    final = {"subtitle": "s", "key_takeaways": TLDR, "closing": closing}
    assert set(_finalization_problems(final, "Title")) == {"closing"}


async def test_open_must_fix_items_reach_the_editor() -> None:
    def check(_: dict[str, Any]) -> dict[str, str]:
        return {
            "score": "85",
            "missing_items": "none",
            "must_fix": "- '틀린 문장' 고칩니다",
        }

    reviewer = _reviewer(checker=FakeChain(check), max_synthesis_attempts=2)
    await reviewer.run()
    first_edit = reviewer.editor.calls[0]  # type: ignore[attr-defined]
    assert "'틀린 문장' 고칩니다" in first_edit["round_note"]


async def test_open_missing_items_reach_the_editor_as_conditional_additions() -> None:
    # Regression: missing items were listed under "the quoted text is from the
    # article", though they quote nothing and may not belong in the review.
    def check(_: dict[str, Any]) -> dict[str, str]:
        return {"score": "85", "missing_items": "- 순위 4 실험", "must_fix": "none"}

    reviewer = _reviewer(checker=FakeChain(check), max_synthesis_attempts=2)
    await reviewer.run()
    note = reviewer.editor.calls[0]["round_note"]  # type: ignore[attr-defined]
    assert "순위 4 실험" in note
    assert "quoted text" not in note and "paper supports" in note
