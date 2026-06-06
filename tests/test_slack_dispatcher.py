"""Tests for the Slack Paper Bot dispatcher core (no Slack, no AWS)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scholar_lens.slack.dispatcher import DispatchResult, JobDispatcher
from scholar_lens.slack.intent import (
    IntentParser,
    ParsedIntent,
    SlackIntent,
    _coerce_intent,
    _split_csv,
    _unwrap_slack_links,
)


class TestIntentHelpers:
    def test_unwrap_plain_link(self) -> None:
        assert _unwrap_slack_links("<https://x.com/a>") == "https://x.com/a"

    def test_unwrap_labelled_link(self) -> None:
        assert (
            _unwrap_slack_links("see <https://arxiv.org/abs/2401.06066|paper>")
            == "see https://arxiv.org/abs/2401.06066"
        )

    def test_split_csv_drops_empty_sentinel(self) -> None:
        assert _split_csv("a, b ,empty,  c") == ["a", "b", "c"]
        assert _split_csv("") == []

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("review", SlackIntent.REVIEW),
            ("  SUMMARIZE ", SlackIntent.SUMMARIZE),
            ("guide", SlackIntent.GUIDE),
            ("nonsense", SlackIntent.UNKNOWN),
        ],
    )
    def test_coerce_intent(self, raw: str, expected: SlackIntent) -> None:
        assert _coerce_intent(raw) == expected


class TestIntentParserFromRaw:
    def test_actionable_review(self) -> None:
        parsed = IntentParser.from_raw(
            {
                "intent": "review",
                "sources": "2401.06066",
                "repo_urls": "https://github.com/x/y",
                "reason": "arxiv id",
            }
        )
        assert parsed.intent is SlackIntent.REVIEW
        assert parsed.sources == ["2401.06066"]
        assert parsed.repo_urls == ["https://github.com/x/y"]
        assert parsed.is_actionable

    def test_unknown_not_actionable(self) -> None:
        parsed = IntentParser.from_raw(
            {"intent": "unknown", "sources": "", "repo_urls": "", "reason": "chit-chat"}
        )
        assert not parsed.is_actionable

    def test_intent_without_sources_not_actionable(self) -> None:
        parsed = IntentParser.from_raw(
            {"intent": "review", "sources": "", "repo_urls": "", "reason": "no id"}
        )
        assert not parsed.is_actionable

    def test_parse_pdf_parsed_from_raw(self) -> None:
        yes = IntentParser.from_raw(
            {"intent": "summarize", "sources": "2401.06066", "parse_pdf": "yes"}
        )
        assert yes.parse_pdf is True
        no = IntentParser.from_raw(
            {"intent": "summarize", "sources": "2401.06066", "parse_pdf": "no"}
        )
        assert no.parse_pdf is False
        default = IntentParser.from_raw(
            {"intent": "summarize", "sources": "2401.06066"}
        )
        assert default.parse_pdf is False


def _dispatcher(submit_mock: MagicMock, *, with_guide: bool = True) -> JobDispatcher:
    return JobDispatcher(
        MagicMock(),
        project_name="scholar-lens",
        stage="dev",
        review_job_queue="rq",
        review_job_definition="rd",
        guide_job_queue="gq" if with_guide else None,
        guide_job_definition="gd" if with_guide else None,
    )


class TestJobDispatcher:
    def test_review_submits_with_mode_and_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        submitted = {}

        def fake_submit(session, name, queue, definition, parameters=None):  # type: ignore[no-untyped-def]
            submitted.update(
                name=name, queue=queue, definition=definition, parameters=parameters
            )
            return "job-123"

        monkeypatch.setattr(
            "scholar_lens.slack.dispatcher.submit_batch_job", fake_submit
        )
        d = _dispatcher(MagicMock())
        parsed = ParsedIntent(intent=SlackIntent.REVIEW, sources=["2401.06066"])
        result = d.dispatch(parsed, timestamp="20260603000000")
        assert isinstance(result, DispatchResult)
        assert result.job_id == "job-123"
        assert submitted["parameters"]["mode"] == "review"
        assert submitted["parameters"]["source"] == "2401.06066"
        # Review must route to the REVIEW queue + definition, not the guide one.
        assert submitted["queue"] == "rq"
        assert submitted["definition"] == "rd"
        assert "review" in submitted["name"]
        # No Slack context -> NULL sentinels (so Batch Ref:: substitution works).
        assert submitted["parameters"]["slack_channel"] == "null"
        assert submitted["parameters"]["slack_thread_ts"] == "null"

    def test_slack_context_threaded_to_batch_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scholar_lens.slack.dispatcher import SlackContext

        captured = {}
        monkeypatch.setattr(
            "scholar_lens.slack.dispatcher.submit_batch_job",
            lambda s, name, queue, definition, parameters=None: captured.update(
                parameters or {}
            )
            or "id",
        )
        d = _dispatcher(MagicMock())
        parsed = ParsedIntent(intent=SlackIntent.REVIEW, sources=["2401.06066"])
        ctx = SlackContext(channel="C123", thread_ts="1700000000.0001", user="U1")
        d.dispatch(parsed, timestamp="t", slack_context=ctx)
        assert captured["slack_channel"] == "C123"
        assert captured["slack_thread_ts"] == "1700000000.0001"

    def test_guide_routes_to_guide_definition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression: guides MUST run the dedicated guide job definition, never
        # the paper-review one (different container entrypoint).
        submitted = {}
        monkeypatch.setattr(
            "scholar_lens.slack.dispatcher.submit_batch_job",
            lambda s, name, queue, definition, parameters=None: submitted.update(
                queue=queue, definition=definition
            )
            or "id",
        )
        d = _dispatcher(MagicMock())
        parsed = ParsedIntent(intent=SlackIntent.GUIDE, sources=["https://docs.x.io"])
        d.dispatch(parsed, timestamp="t")
        assert submitted["definition"] == "gd"
        assert submitted["queue"] == "gq"

    def test_guide_without_definition_is_rejected(self) -> None:
        d = _dispatcher(MagicMock(), with_guide=False)
        parsed = ParsedIntent(intent=SlackIntent.GUIDE, sources=["https://docs.x.io"])
        with pytest.raises(ValueError, match="guide job definition"):
            d.dispatch(parsed, timestamp="t")

    def test_summarize_sets_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}
        monkeypatch.setattr(
            "scholar_lens.slack.dispatcher.submit_batch_job",
            lambda *a, **k: captured.update(k) or "id",
        )
        d = _dispatcher(MagicMock())
        parsed = ParsedIntent(intent=SlackIntent.SUMMARIZE, sources=["2401.06066"])
        d.dispatch(parsed, timestamp="t")
        assert captured["parameters"]["mode"] == "summarize"

    def test_guide_passes_all_urls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}
        monkeypatch.setattr(
            "scholar_lens.slack.dispatcher.submit_batch_job",
            lambda *a, **k: captured.update(k) or "id",
        )
        d = _dispatcher(MagicMock())
        parsed = ParsedIntent(
            intent=SlackIntent.GUIDE,
            sources=["https://docs.x.io/a", "https://docs.x.io/b"],
        )
        d.dispatch(parsed, timestamp="t")
        assert (
            captured["parameters"]["urls"] == "https://docs.x.io/a https://docs.x.io/b"
        )

    def test_non_actionable_rejected(self) -> None:
        d = _dispatcher(MagicMock())
        parsed = ParsedIntent(intent=SlackIntent.UNKNOWN, sources=[])
        with pytest.raises(ValueError):
            d.dispatch(parsed, timestamp="t")
