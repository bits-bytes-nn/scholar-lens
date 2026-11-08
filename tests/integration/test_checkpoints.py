"""Stage checkpoints and resume (docs/engine.md, 4.9): a rerun with the same
inputs makes no model call for a saved stage, a changed input recomputes
only the stages after it, a resumed run's notes, blocking problems and
verdicts match an uninterrupted run's, an interrupted run resumes after its
last saved section, and a fresh run recomputes."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from scholar_lens.adapters.scholarly.survey_documents import EvidenceTextStore
from scholar_lens.application.checkpoints import (
    CHECKPOINT_VERSION,
    Checkpoint,
    Checkpoints,
    FreshStore,
    input_hash,
)
from scholar_lens.application.evidence_writer import (
    ClaimCheck,
    PostFinding,
    PostReview,
    SentenceVerdict,
)
from scholar_lens.application.paper_engine import EvidencePaperEngine, PaperSettings
from tests.unit.test_paper_engine import ScriptedModel, _paper

METHOD_TITLE = "보상 모델 없이 푸는 선호 최적화"
# The whole-post finding's marker (its fix adds a sentence to the method).
WHOLE_POST = "WHOLE-POST"


class StatefulModel(ScriptedModel):
    """The paper script with writer state to restore: the claim check
    objects to the problem section's one sentence (deleting it would cut
    the whole section, so the post is blocked), and the whole-post review
    names the method section (its fix adds a sentence, checked against the
    cached verdicts of the rest) and a section the post does not have (a
    note). ``fail_on``: the draft of that section kind raises."""

    def __init__(self, post: str = "review", *, fail_on: str = "") -> None:
        super().__init__(post)
        self.fail_on = fail_on
        self.claim_prompts: list[str] = []

    async def structured(self, *, role, system, prompt, schema, context=""):  # type: ignore[no-untyped-def]
        if schema is ClaimCheck:
            self.calls.append("ClaimCheck")
            self.claim_prompts.append(prompt)
            return ClaimCheck(
                verdicts=[
                    SentenceVerdict(n=int(n), verdict="unsupported", reason="넓음")
                    for n, sentence in re.findall(r"^(\d+)\. (.*)$", prompt, re.M)
                    if "RLHF는" in sentence
                ]
            )
        if schema is PostReview:
            self.calls.append("PostReview")
            if "repetition only" in system:
                return PostReview()
            return PostReview(
                issues=[
                    PostFinding(section=METHOD_TITLE, problem=WHOLE_POST, fix="더함"),
                    PostFinding(section="없는 절", problem="x", fix="y"),
                ]
            )
        return await super().structured(
            role=role, system=system, prompt=prompt, schema=schema, context=context
        )

    async def text(self, *, role, system, prompt, context="") -> str:  # type: ignore[no-untyped-def]
        if "<problems>" in prompt:
            self.calls.append("fix")
            old = prompt.split("<section>\n", 1)[1].split("\n</section>", 1)[0]
            if WHOLE_POST in prompt:
                return old.strip() + "\n\nDPO는 보상 모델이 필요 없습니다 [[E2]].\n"
            return old
        if self.fail_on and f"kind: {self.fail_on}\n" in prompt:
            raise RuntimeError("the model went away")
        return await super().text(
            role=role, system=system, prompt=prompt, context=context
        )


def _engine(model: Any, store: Any, **settings: Any) -> EvidencePaperEngine:
    return EvidencePaperEngine(
        model,
        PaperSettings(
            language="Korean",
            post="review",
            reader_tag="sonnet",
            min_total_chars=1_000,
            max_total_chars=2_000,
            **settings,
        ),
        store=store,
        checkpoints=store,
    )


def _write(engine: EvidencePaperEngine) -> Any:
    return asyncio.run(engine.write("DPO", _paper(), abstract="We propose DPO."))


@pytest.fixture
def store(tmp_path: Path) -> EvidenceTextStore:
    return EvidenceTextStore(tmp_path / "evidence", None, "")


def _saved(tmp_path: Path, stage: str) -> list[Path]:
    return sorted((tmp_path / "evidence" / "checkpoints" / "review" / stage).glob("*"))


def test_a_rerun_with_the_same_inputs_calls_no_model(
    store: EvidenceTextStore, tmp_path: Path
) -> None:
    first = StatefulModel()
    done = _write(_engine(first, store))
    assert {"ExtractedTargets", "PaperOutline", "ClaimCheck", "fix"} <= set(first.calls)
    for stage in ("targets", "outline", "section", "review", "tldr", "final"):
        assert _saved(tmp_path, stage), stage
    # Four sections (the TL;DR is its own stage).
    assert len(_saved(tmp_path, "section")) == 4

    again = StatefulModel()
    engine = _engine(again, store)
    resumed = _write(engine)
    assert again.calls == []
    assert resumed.sections == done.sections
    assert resumed.subtitle == done.subtitle
    assert resumed.notes == done.notes and resumed.blocking == done.blocking


def test_a_changed_section_setting_recomputes_only_the_later_stages(
    store: EvidenceTextStore,
) -> None:
    _write(_engine(StatefulModel(), store))
    changed = StatefulModel()
    _write(_engine(changed, store, max_fix_rounds=1))
    # The targets, the reading and the outline (with its evidence check)
    # are loaded; the sections onward run again.
    assert not {"ExtractedTargets", "read", "PaperOutline", "EvidenceCheck"} & set(
        changed.calls
    )
    assert {"text", "ClaimCheck", "PostReview", "FinalText"} <= set(changed.calls)


def test_a_changed_prompt_invalidates_its_stage(store: EvidenceTextStore) -> None:
    _write(_engine(StatefulModel(), store))
    engine = _engine(StatefulModel(), store)
    engine.post_prompts = engine.post_prompts.__class__(
        **{**engine.post_prompts.__dict__, "outline": "A new outline prompt."}
    )
    _write(engine)
    calls = engine.model.calls  # type: ignore[attr-defined]
    assert "ExtractedTargets" not in calls and "PaperOutline" in calls


def test_a_resumed_run_restores_the_notes_blocking_and_verdicts(
    tmp_path: Path,
) -> None:
    whole = _engine(StatefulModel(), EvidenceTextStore(tmp_path / "a", None, ""))
    uninterrupted = _write(whole)
    assert uninterrupted.blocking and any(
        "naming no section" in n for n in uninterrupted.notes
    )

    store = EvidenceTextStore(tmp_path / "b", None, "")
    with pytest.raises(RuntimeError, match="went away"):
        _write(_engine(StatefulModel(fail_on="experiments"), store))
    model = StatefulModel()
    resumed_engine = _engine(model, store)
    resumed = _write(resumed_engine)

    assert resumed.sections == uninterrupted.sections
    assert resumed.notes == uninterrupted.notes
    assert resumed.blocking == uninterrupted.blocking
    assert resumed_engine.verdicts.entries() == whole.verdicts.entries()
    # The whole-post fix of the loaded method section ran its registered
    # checks and sent only the sentence it added to the claim check.
    review_checks = [p for p in model.claim_prompts if "필요 없습니다" in p]
    sent = [s for p in review_checks for _, s in re.findall(r"^(\d+)\. (.*)$", p, re.M)]
    assert sent and not any("선호를 최적화합니다" in s for s in sent)


def test_an_interrupted_run_resumes_after_its_last_saved_section(
    store: EvidenceTextStore, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError):
        _write(_engine(StatefulModel(fail_on="experiments"), store))
    # The problem and method sections were accepted before the failure.
    assert len(_saved(tmp_path, "section")) == 2
    model = StatefulModel()
    prompts: list[str] = []
    original = model.text

    async def text(**call: Any) -> str:
        prompts.append(call["prompt"])
        return await original(**call)

    model.text = text  # type: ignore[method-assign]
    _write(_engine(model, store))
    assert not {"ExtractedTargets", "read", "PaperOutline"} & set(model.calls)
    drafts = [p for p in prompts if "<problems>" not in p]
    assert "kind: experiments\n" in drafts[0]
    assert not any("kind: problem\n" in p or "kind: method\n" in p for p in drafts)


def test_a_fresh_run_recomputes_and_still_saves(store: EvidenceTextStore) -> None:
    _write(_engine(StatefulModel(), store))
    fresh = StatefulModel()
    _write(_engine(fresh, FreshStore(store)))
    assert {"ExtractedTargets", "read", "PaperOutline", "text"} <= set(fresh.calls)
    after = StatefulModel()
    _write(_engine(after, store))
    assert after.calls == []


def test_without_a_store_every_stage_runs() -> None:
    model = StatefulModel()
    engine = EvidencePaperEngine(
        model,
        PaperSettings(
            language="Korean",
            post="review",
            reader_tag="sonnet",
            min_total_chars=1_000,
            max_total_chars=2_000,
        ),
    )
    _write(engine)
    assert engine.checkpoints is None and "PaperOutline" in model.calls


# ---- the checkpoint format ------------------------------------------------------


def test_the_input_hash_is_stable_and_refuses_unstable_parts() -> None:
    settings = PaperSettings(language="Korean", post="review", reader_tag="x")
    assert input_hash(settings, {"b": 1, "a": frozenset({"y", "x"})}) == input_hash(
        settings, {"a": frozenset({"x", "y"}), "b": 1}
    )
    assert input_hash("a") != input_hash("b")
    with pytest.raises(TypeError):
        input_hash(lambda: None)


def test_an_unreadable_or_older_checkpoint_is_run_again(
    store: EvidenceTextStore,
) -> None:
    checkpoints = Checkpoints(store, "review")
    key = checkpoints.key("outline", input_hash("x"))
    assert key.startswith("checkpoints/review/outline/")
    assert checkpoints.load(key) is None
    store.put(key, "{not json")
    assert checkpoints.load(key) is None
    store.put(
        key,
        Checkpoint(
            version=CHECKPOINT_VERSION + 1, stage="outline", value={}
        ).model_dump_json(),
    )
    assert checkpoints.load(key) is None
    saved = Checkpoint(version=CHECKPOINT_VERSION, stage="outline", value={"a": 1})
    checkpoints.save(key, saved)
    assert checkpoints.load(key) == saved


def test_a_checkpoint_that_does_not_fit_its_stage_is_run_again(
    store: EvidenceTextStore, tmp_path: Path
) -> None:
    _write(_engine(StatefulModel(), store))
    for path in _saved(tmp_path, "outline"):
        data = Checkpoint.model_validate_json(path.read_text())
        path.write_text(
            data.model_copy(update={"value": {"sections": "broken"}}).model_dump_json()
        )
    model = StatefulModel()
    _write(_engine(model, store))
    assert "PaperOutline" in model.calls and "ExtractedTargets" not in model.calls


def test_a_fresh_runs_store_is_written_but_not_read(tmp_path: Path) -> None:
    from scholar_lens.bootstrap import evidence_store

    kept = evidence_store(tmp_path, None, "")
    kept.put("k", "v")
    assert kept.get("k") == "v"
    fresh = evidence_store(tmp_path, None, "", fresh=True)
    assert isinstance(fresh, FreshStore) and fresh.get("k") is None
    fresh.put("k2", "w")
    assert kept.get("k2") == "w"


# ---- every kind resumes -----------------------------------------------------------


def test_a_guide_rerun_calls_no_model(store: EvidenceTextStore) -> None:
    from scholar_lens.application.guide_engine import (
        EvidenceGuideEngine,
        GuideSettings,
    )
    from tests.unit.test_guide_engine import ScriptedModel as GuideModel
    from tests.unit.test_guide_engine import _blog, _official

    def run() -> tuple[Any, Any]:
        model = GuideModel()
        engine = EvidenceGuideEngine(
            model,
            GuideSettings(
                language="Korean",
                title_max_length=70,
                reader_tag="sonnet",
                min_total_chars=5_000,
            ),
            store=store,
            checkpoints=store,
            identifier_check=lambda md: [],
        )
        guide = asyncio.run(engine.write("Ray Serve batching", [_blog(), _official()]))
        return model, guide

    first_model, first = run()
    assert first_model.calls
    again_model, again = run()
    assert again_model.calls == []
    assert again.body == first.body and again.notes == first.notes


def test_a_survey_rerun_calls_no_model(store: EvidenceTextStore) -> None:
    from scholar_lens.application.survey_engine import (
        EngineSettings,
        EvidenceSurveyEngine,
    )
    from tests.unit.test_evidence_engine import FakeDocuments, _plan
    from tests.unit.test_evidence_engine import ScriptedModel as SurveyModel

    def run() -> tuple[Any, Any]:
        model = SurveyModel()
        engine = EvidenceSurveyEngine(
            model,
            FakeDocuments(),
            EngineSettings(language="Korean", title_max_length=70, reader_tag="sonnet"),
            store,
            checkpoints=store,
        )
        return model, asyncio.run(engine.write(_plan()))

    first_model, first = run()
    assert first_model.calls
    again_model, again = run()
    assert again_model.calls == []
    assert again.body == first.body and again.blocking == first.blocking
