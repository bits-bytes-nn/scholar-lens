"""Survey write stage: CLI and job contract, plan re-validation, citations,
figure credits and references, the survey validator rules and the whole
stage with the evidence engine faked (post validated and, when the blog
checkout is there, run through its lint). No network, no model calls."""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import boto3
import pytest
from PIL import Image
from pydantic import ValidationError

from scholar_lens import cli
from scholar_lens.adapters.bedrock.strands_models import RoleModel
from scholar_lens.application import survey_engine
from scholar_lens.application.survey_engine import EvidenceSurvey
from scholar_lens.configs.config import Config, Survey
from scholar_lens.domain.conventions import (
    BlogPost,
    PostConventions,
    TypeSpec,
    load_local_conventions,
)
from scholar_lens.domain.evidence import Evidence
from scholar_lens.domain.survey.plan import SourceRecord, SurveyInputError, SurveyPlan
from scholar_lens.domain.survey.post import (
    author_year,
    credit_figures,
    equation_in_source,
    foreign_paper_links,
    further_reading,
    reference_list,
    render_citations,
    unlink_foreign_papers,
)
from scholar_lens.domain.validation import validate_post
from scholar_lens.src import evidence_models, survey_writer
from scholar_lens.src.constants import LanguageModelId
from scholar_lens.src.jobs import JobRequest
from scholar_lens.src.metrics import TokenUsageTracker
from scholar_lens.src.publisher import Publisher
from scholar_lens.src.runtime import RunContext
from scholar_lens.src.survey_writer import (
    PaperIndex,
    SurveyWriter,
    load_plan,
    read_plan_text,
    stage_plan_for_batch,
    survey_body,
)
from scholar_lens.src.web_research import PageContent, ResearchCorpus
from tests.fakes import FakeLLMFactory

# The blog checkout next to this repository, when there is one.
BLOG = Path(__file__).resolve().parents[2] / "tech-blog"
SURVEY_TYPE = "Paper Surveys"
LOSS = r"\mathcal{L}_\text{DPO} = -\log \sigma(\beta \Delta)"


def survey_spec() -> PostConventions:
    """The bundled conventions; they list the blog's survey type."""
    return load_local_conventions()


def test_bundled_spec_has_the_survey_type() -> None:
    survey_category = Config().resources.github.survey_category
    assert survey_category in load_local_conventions().types


def plan_dict() -> dict[str, Any]:
    paper = {
        "authors": "Rafael Rafailov, Archit Sharma, Eric Mitchell et al.",
        "venue": "arXiv",
        "citation_count": 100,
        "reason": "r",
    }
    return {
        "schema_version": 1,
        "created_at": "2026-10-01T09:30:00Z",
        "topic": "preference optimization",
        "scope": "How LLMs learn from preferences.",
        "time_range": {"since": None, "until": None},
        "lineages": [
            {"id": "rlhf", "name": "RLHF", "description": "reward model + RL"},
            {"id": "dpo", "name": "Direct methods", "description": "no reward model"},
        ],
        "papers": [
            {
                **paper,
                "id": "1707.06347",
                "title": "Proximal Policy Optimization Algorithms",
                "authors": "John Schulman, Filip Wolski, Prafulla Dhariwal et al.",
                "year": 2017,
                "url": "https://arxiv.org/abs/1707.06347",
                "lineage": "rlhf",
                "role": "core",
            },
            {
                **paper,
                "id": "2203.02155",
                "title": "Training language models to follow instructions",
                "authors": "Long Ouyang, Jeff Wu et al.",
                "year": 2022,
                "url": "https://arxiv.org/abs/2203.02155",
                "lineage": "rlhf",
                "role": "supporting",
            },
            {
                **paper,
                "id": "2305.18290",
                "title": "Direct Preference Optimization",
                "year": 2023,
                "url": "https://arxiv.org/abs/2305.18290",
                "lineage": "dpo",
                "role": "core",
            },
        ],
        "edges": [{"from": "2305.18290", "to": "1707.06347"}],
    }


def plan() -> SurveyPlan:
    return SurveyPlan.model_validate(plan_dict())


# ---- CLI and job contract -------------------------------------------------------------


class TestCliAndJob:
    def _request(self, argv: list[str]) -> JobRequest:
        return cli._request_from_args(cli.parse_args(argv))

    def test_survey_write_options(self) -> None:
        request = self._request(["survey", "write", "--plan", "p.json", "--no-pr"])
        assert (request.kind, request.plan, request.create_pr) == (
            "survey_write",
            "p.json",
            False,
        )
        assert request.describe() == "p.json"
        assert self._request(["survey", "write", "--plan", "p.json"]).create_pr

    def test_plan_is_required(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit):
            cli.parse_args(["survey", "write"])
        assert json.loads(capsys.readouterr().out)["status"] == "rejected"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "survey_write"},
            {"kind": "survey_write", "plan": " "},
            {"kind": "survey_write", "plan": "p.json", "topic": "x"},
            {"kind": "survey_write", "plan": "p.json", "source": "2305.18290"},
            {"kind": "survey_plan", "topic": "x", "plan": "p.json"},
            {"kind": "review", "source": "2305.18290", "plan": "p.json"},
        ],
    )
    def test_invalid_requests(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            JobRequest(**kwargs)

    def test_payload_round_trip(self) -> None:
        request = JobRequest(kind="survey_write", plan="s3://b/k.json", create_pr=False)
        sent = request.model_dump_json(exclude_defaults=True)
        assert json.loads(sent) == {
            "kind": "survey_write",
            "plan": "s3://b/k.json",
            "create_pr": False,
        }
        assert JobRequest.model_validate_json(sent) == request
        assert cli._raw_kind('{"kind": "survey_write"}') == "survey_write"

    def test_remote_uploads_a_local_plan(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan_dict()))
        s3 = MagicMock(bucket_name="bucket")
        request = JobRequest(kind="survey_write", plan=str(path))
        staged = stage_plan_for_batch(request, s3, "scholar-lens")
        key = s3.put_text.call_args.args[0]
        assert re.fullmatch(
            r"scholar-lens/surveys/survey-preference-optimization/plan-[0-9a-f]{16}\.json",
            key,
        )
        assert staged.plan == f"s3://bucket/{key}"
        # An S3 plan is submitted as it is; without a bucket there is nowhere
        # to upload a local one.
        assert stage_plan_for_batch(staged, None, "x") is staged
        with pytest.raises(SurveyInputError):
            stage_plan_for_batch(request, None, "x")

    def test_submit_stages_the_plan(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan_dict()))
        context = MagicMock()
        context.config = Config()
        context.s3_handler.bucket_name = "bucket"
        submitted = MagicMock()
        with (
            mock.patch("scholar_lens.cli._aws_context", return_value=context),
            mock.patch(
                "scholar_lens.src.jobs.submit_job",
                return_value=submitted,
            ) as submit,
            mock.patch("scholar_lens.cli._emit"),
        ):
            cli._submit(JobRequest(kind="survey_write", plan=str(path)))
        assert submit.call_args.args[1].plan.startswith("s3://bucket/")

    def test_config_template_matches_defaults(self) -> None:
        from scholar_lens.configs.config import CONFIG_PATH

        template = Config.from_yaml(CONFIG_PATH.with_name("config-template.yaml"))
        assert template.survey == Survey()
        assert template.resources.github.survey_category == SURVEY_TYPE
        with pytest.raises(ValidationError):
            Survey(min_total_chars=50_000, max_total_chars=40_000)


# ---- plan re-validation -----------------------------------------------------------------


class FakeIndex(PaperIndex):
    def __init__(self, records: dict[str, SourceRecord]) -> None:
        self.found = records
        self.asked: list[list[str]] = []

    def records(self, paper_ids: Any) -> dict[str, SourceRecord]:
        self.asked.append(list(paper_ids))
        return {i: self.found[i] for i in paper_ids if i in self.found}


KTO = SourceRecord(
    provider="arxiv",
    title="KTO: Model Alignment as Prospect Theoretic Optimization",
    authors=["Kawin Ethayarajh", "Winnie Xu", "Niklas Muennighoff"],
    published=__import__("datetime").date(2024, 2, 2),
    arxiv_id="2402.01306",
    abstract="KTO aligns models with binary feedback; it matches DPO at 1B to 30B.",
)


class TestPlanRevalidation:
    def test_edits_are_normalized_and_a_hand_added_paper_completed(self) -> None:
        data = plan_dict()
        # Removed: InstructGPT. Added by hand, as a URL: KTO. Moved: DPO's role.
        data["papers"] = [p for p in data["papers"] if p["id"] != "2203.02155"]
        data["papers"][1]["role"] = "supporting"
        data["papers"].append(
            {
                "id": "https://arxiv.org/abs/2402.01306v2",
                "lineage": "dpo",
                "role": "core",
            }
        )
        data["edges"].append({"from": "2203.02155", "to": "1707.06347"})
        index = FakeIndex({"2402.01306": KTO})
        loaded, notes = load_plan(json.dumps(data), index)
        kto = next(p for p in loaded.papers if p.id == "2402.01306")
        assert kto.title.startswith("KTO") and kto.year == 2024
        assert kto.url == "https://arxiv.org/abs/2402.01306"
        assert kto.authors == "Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff"
        assert [(e.citing, e.cited) for e in loaded.edges] == [
            ("2305.18290", "1707.06347")
        ]
        assert index.asked == [["2402.01306"]]
        assert any("edge" in n for n in notes) and any("by hand" in n for n in notes)

    def test_a_lineage_left_empty_is_dropped(self) -> None:
        data = plan_dict()
        data["papers"] = [p for p in data["papers"] if p["lineage"] == "dpo"]
        loaded, notes = load_plan(json.dumps(data), FakeIndex({}))
        assert [x.id for x in loaded.lineages] == ["dpo"]
        assert any("rlhf" in n for n in notes)

    @pytest.mark.parametrize(
        "edit, message",
        [
            (
                lambda d: d["papers"].append(
                    {"id": "2412.99999", "lineage": "dpo", "role": "core"}
                ),
                "No index knows",
            ),
            (lambda d: d["papers"][0].update(lineage="renamed"), "unknown lineage"),
            (lambda d: d["papers"][0].update(id="not a paper"), "Not an arXiv id"),
            (lambda d: d["papers"][0].update(role="key"), "not valid"),
        ],
    )
    def test_bad_edits_are_rejected(self, edit: Any, message: str) -> None:
        data = plan_dict()
        edit(data)
        with pytest.raises(SurveyInputError, match=message):
            load_plan(json.dumps(data), FakeIndex({}))

    def test_not_json_is_rejected(self) -> None:
        with pytest.raises(SurveyInputError, match="JSON"):
            load_plan("{not json", FakeIndex({}))

    def test_a_plan_that_is_not_utf8_is_an_input_error(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.json"
        path.write_bytes('{"topic": "선호"}'.encode("cp949"))
        with pytest.raises(SurveyInputError, match="not UTF-8"):
            read_plan_text(str(path), None)
        s3 = MagicMock(bucket_name="bucket")
        s3.get_text.side_effect = UnicodeDecodeError("utf-8", b"\xbc", 0, 1, "bad")
        with pytest.raises(SurveyInputError, match="not UTF-8"):
            read_plan_text("s3://bucket/a/plan.json", s3)

    @pytest.mark.parametrize(
        "edit, key",
        [
            (lambda d: d["papers"][0].update(lineage=["rlhf"]), "papers"),
            (lambda d: d["lineages"][0].update(id=1), "lineages"),
            (lambda d: d["edges"].append({"from": 1, "to": "x"}), "edges"),
        ],
    )
    def test_wrong_typed_plan_fields_are_input_errors(
        self, edit: Any, key: str
    ) -> None:
        data = plan_dict()
        edit(data)
        with pytest.raises(SurveyInputError, match=f"'{key}' entries need text"):
            load_plan(json.dumps(data), FakeIndex({}))

    def test_plan_locations(self, tmp_path: Path) -> None:
        path = tmp_path / "plan.json"
        path.write_text("{}")
        assert read_plan_text(str(path), None) == "{}"
        s3 = MagicMock(bucket_name="bucket")
        s3.get_text.return_value = "{}"
        assert read_plan_text("s3://bucket/a/plan.json", s3) == "{}"
        s3.get_text.assert_called_once_with("a/plan.json")
        with pytest.raises(SurveyInputError, match="bucket"):
            read_plan_text("s3://other/a/plan.json", s3)
        with pytest.raises(SurveyInputError):
            read_plan_text(str(tmp_path / "missing.json"), None)

    def test_index_prefers_arxiv_abstract_for_a_doi_paper(self) -> None:
        openalex = MagicMock()
        openalex.by_doi.return_value = [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.18653/v1/x",
                "title": "ORPO",
                "locations": [{"landing_page_url": "https://arxiv.org/abs/2403.07691"}],
                "abstract_inverted_index": {"wrong": [0]},
            }
        ]
        datacite = MagicMock()
        datacite.arxiv_papers.return_value = [
            {
                "id": "10.48550/arxiv.2403.07691",
                "attributes": {
                    "doi": "10.48550/arxiv.2403.07691",
                    "titles": [{"title": "ORPO"}],
                    "descriptions": [
                        {"descriptionType": "Abstract", "description": "Right one."}
                    ],
                },
            }
        ]
        found = PaperIndex(datacite, openalex).records(["10.18653/v1/x"])
        assert found["10.18653/v1/x"].abstract == "Right one."
        assert found["10.18653/v1/x"].arxiv_id == "2403.07691"
        datacite.arxiv_papers.assert_called_once_with(["2403.07691"])


# ---- equations, citations, figures, references --------------------------------------

SOURCE = (
    "## Method\nWe optimize $$" + LOSS + "$$ directly. DPO reaches a 61% win rate "
    "on TL;DR summarization against PPO's 57%."
)
FIGURE = "https://arxiv.org/html/2305.18290v3/x1.png"




class TestCitations:
    PAPERS = {p.id: p for p in plan().papers}

    def test_keys_render_as_author_year_links(self) -> None:
        text, unknown = render_citations(
            "DPO[@2305.18290]는 PPO [@1707.06347; @2203.02155]를 대체하고 [@2499.0] "
            "끝입니다. `[@2305.18290]`",
            self.PAPERS,
        )
        assert text == (
            "DPO([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))는 PPO "
            "([Schulman et al., 2017](https://arxiv.org/abs/1707.06347); "
            "[Ouyang et al., 2022](https://arxiv.org/abs/2203.02155))를 대체하고 "
            "끝입니다. `[@2305.18290]`"
        )
        assert unknown == ["2499.0"]

    def test_author_year(self) -> None:
        one = self.PAPERS["1707.06347"].model_copy(update={"authors": "John Schulman"})
        assert author_year(one) == "Schulman, 2017"

    def test_equations_are_found_in_the_source_latex(self) -> None:
        assert equation_in_source(f"$$ {LOSS.replace(' ', '')} $$", SOURCE)
        assert not equation_in_source("", SOURCE)
        assert not equation_in_source(r"\mathcal{L} = 0", SOURCE)

    def test_figures_are_credited_to_their_paper(self) -> None:
        dpo = self.PAPERS["2305.18290"]
        body = f"앞 문장입니다.\n\n![그림]({FIGURE})\n\n뒤 문장입니다."
        credited = credit_figures(body, {FIGURE: dpo}, "Korean")
        assert (
            f"![그림]({FIGURE})\n\n*출처: [Rafailov et al., 2023]"
            "(https://arxiv.org/abs/2305.18290)*"
        ) in credited

    def test_reference_list(self) -> None:
        paper = plan().papers[0]
        odd = paper.model_copy(
            update={"title": "The *best* $x_1$ [policy]", "venue": None}
        )
        refs = reference_list([paper, odd])
        assert refs == (
            "- - -\n### References\n* John Schulman, Filip Wolski, Prafulla "
            "Dhariwal et al. (2017). [Proximal Policy Optimization Algorithms]"
            "(https://arxiv.org/abs/1707.06347). arXiv.\n"
            "* John Schulman, Filip Wolski, Prafulla Dhariwal et al. (2017). "
            "[The &#42;best&#42; &#36;x&#95;1&#36; &#91;policy&#93;]"
            "(https://arxiv.org/abs/1707.06347).\n"
        )


# ---- validator survey rules -------------------------------------------------------------


def survey_post(body: str, **front: str) -> str:
    fields = {
        "title": '"Preference Optimization Survey"',
        "date": "2026-10-14 00:00:00",
        "description": '"'
        + "선호 최적화 연구의 흐름을 RLHF에서 DPO까지 계보별로 정리하고 각 방법의 아이디어와 한계를 비교합니다. "
        * 2
        + '"',
        "categories": f'["{SURVEY_TYPE}", "Language-Models"]',
        "tags": '["Alignment", "DPO"]',
        "cover": "/assets/images/language-models.jpg",
        **front,
    }
    head = "\n".join(f"{k}: {v}" for k, v in fields.items())
    return f"---\nlayout: post\n{head}\n---\n{body}"


GOOD_BODY = (
    "## 들어가며\n\nDPO([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))를 "
    "다룹니다.\n\n- - -\n### References\n* Rafailov et al. (2023). [DPO]"
    "(https://arxiv.org/abs/2305.18290). arXiv.\n"
)


def rules(markdown: str) -> list[str]:
    return [
        f"{v.level}:{v.rule}:{v.message}"
        for v in validate_post(markdown, survey_spec(), "survey")
    ]


class TestValidator:
    def test_a_good_survey_passes(self) -> None:
        assert [r for r in rules(survey_post(GOOD_BODY)) if r.startswith("error")] == []

    def test_survey_citation_rules(self) -> None:
        body = GOOD_BODY.replace(
            "다룹니다.",
            "다룹니다 [@2402.01306]. [PPO](https://arxiv.org/abs/1707.06347)",
        )
        body += (
            "* [Ouyang et al. (2022). InstructGPT](https://arxiv.org/abs/2203.02155)\n"
        )
        found = rules(survey_post(body))
        assert "error:citations:unresolved citation [@2402.01306]" in found
        assert (
            "error:references:reference never cited: https://arxiv.org/abs/2203.02155"
            in found
        )
        assert (
            "error:citations:cited paper not in References: https://arxiv.org/abs/1707.06347"
            in found
        )
        no_refs = rules(survey_post("## 들어가며\n\n본문입니다.\n"))
        assert any(r.startswith("error:references:a survey ends with") for r in no_refs)

    def test_further_reading_is_not_a_list_of_cited_papers(self) -> None:
        body = GOOD_BODY + further_reading(
            [("RLHF: Lil'Log", ARTICLE), ("A post", "https://blog.x.io/a")], "Korean"
        )
        assert body.endswith(
            "arXiv.\n\n### 참고 자료\n* [RLHF: Lil'Log]"
            f"({ARTICLE})\n* [A post](https://blog.x.io/a)\n"
        )
        assert [r for r in rules(survey_post(body)) if r.startswith("error")] == []
        # An uncited paper in References is still an error.
        extra = body.replace(
            "\n\n### 참고 자료",
            "\n* Ouyang et al. (2022). [IG](https://arxiv.org/abs/2203.02155).\n"
            "\n### 참고 자료",
        )
        assert (
            "error:references:reference never cited: https://arxiv.org/abs/2203.02155"
            in rules(survey_post(extra))
        )
        assert further_reading([], "Korean") == ""
        assert further_reading([("T", ARTICLE)], "English").startswith(
            "\n### Further reading\n"
        )

    def test_a_paper_linked_only_by_the_timeline_is_flagged(self) -> None:
        # The code-built timeline links every plan paper; the prose skipping
        # one is flagged for the reviewer without blocking the post.
        timeline = (
            "\n\n| 연도 | 논문 | 계보 | 핵심 아이디어 |\n| --- | --- | --- | --- |\n"
            "| 2022 | [InstructGPT](https://arxiv.org/abs/2203.02155) | RLHF | x |\n"
        )
        body = GOOD_BODY.replace("다룹니다.", "다룹니다." + timeline)
        body += (
            "* Ouyang et al. (2022). [InstructGPT](https://arxiv.org/abs/2203.02155).\n"
        )
        found = rules(survey_post(body))
        assert not [r for r in found if r.startswith("error")]
        assert (
            "warning:references:cited only in a table: https://arxiv.org/abs/2203.02155"
            in found
        )

    def test_survey_front_matter(self) -> None:
        found = rules(survey_post(GOOD_BODY, paper_author='"Stanford"'))
        assert "error:paper-author:a survey does not take paper_author" in found
        found = rules(survey_post(GOOD_BODY, paper_doi='"10.48550/arXiv.2305.18290"'))
        assert "error:paper-doi:a survey does not take paper_doi" in found
        wrong_type = rules(
            survey_post(GOOD_BODY, categories='["Paper Reviews", "Language-Models"]')
        )
        assert (
            "error:categories:type 'Paper Reviews' does not fit a survey" in wrong_type
        )

    def test_a_guide_keeps_its_rules(self) -> None:
        found = [
            f"{v.rule}:{v.message}"
            for v in validate_post(
                survey_post(
                    "## A\n\nText.\n", categories='["Tech Guides", "Language-Models"]'
                ),
                survey_spec(),
                "guide",
            )
            if v.level == "error"
        ]
        assert found == []


# ---- the stage with the evidence engine faked ----------------------------------------------


class FakeChain:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.payloads: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> Any:
        self.payloads.append(payload)
        return self.answer(payload) if callable(self.answer) else self.answer


DESCRIPTION = (
    "선호 최적화 연구를 보상 모델과 강화학습을 쓰는 RLHF 계보와 보상 모델 없이 정책을 "
    "직접 학습하는 DPO 계보로 나누어 각 방법의 핵심 아이디어와 한계, 서로의 관계를 "
    "시간순으로 정리합니다."
)


def fake_writer() -> SurveyWriter:
    writer = SurveyWriter(
        finalization_model_id=LanguageModelId.CLAUDE_V4_5_HAIKU,
        boto_session=boto3.Session(region_name="us-east-1"),
        language="Korean",
        thinking_effort="medium",
        llm_factory=FakeLLMFactory(),  # type: ignore[arg-type]
    )
    writer.finalization_chain = FakeChain(
        {
            "category": "Language-Models",
            "controlled_tags": "Alignment",
            "keywords": "Preference Optimization, Reward Modeling",
            "description": DESCRIPTION,
        }
    )
    return writer


ARTICLE = "https://lilianweng.github.io/posts/2024-07-07-rlhf/"
ARTICLE_TEXT = (
    "## Reward hacking\n\nReward hacking is the central open problem of RLHF: "
    "the policy exploits flaws of the reward model."
)


class FakeResearcher:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def research(self, urls: list[str], *, discover_subpages: bool) -> ResearchCorpus:
        self.asked += urls
        if urls[0] == ARTICLE:
            return ResearchCorpus(
                pages=[PageContent(ARTICLE, "RLHF | Lil'Log", ARTICLE_TEXT)]
            )
        if "arxiv.org" not in urls[0]:
            return ResearchCorpus()  # unreadable
        arxiv_id = urls[0].rsplit("/", 1)[1]
        html = f"https://arxiv.org/html/{arxiv_id}v1"
        figure = f"{html}/x1.png"
        return ResearchCorpus(
            pages=[
                PageContent(
                    urls[0],
                    "paper",
                    SOURCE,
                    image_urls=[figure],
                    image_captions={figure: "Figure 1: Overview."},
                    final_url=html,
                )
            ]
        )

    def close(self) -> None:
        pass


ABSTRACT = SourceRecord(
    provider="arxiv", title="InstructGPT", abstract="InstructGPT follows instructions."
)


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return buffer.getvalue()


# The stage's translation guideline (the engine gets it).
GUIDELINE = [{"original": "helpfulness", "translated": "유용성"}]
TITLE = "선호 최적화: 보상 모델에서 직접 최적화까지"
NAMES = {"1707.06347": "PPO", "2203.02155": "InstructGPT", "2305.18290": "DPO"}
LINEAGE_TITLES = {"rlhf": "보상 모델 계보", "dpo": "직접 최적화 계보"}
# The DPO figure of FakeResearcher's arXiv HTML rendering.
DPO_FIGURE = "https://arxiv.org/html/2305.18290v1/x1.png"


def figure_evidence(url: str, n: int = 1) -> Evidence:
    return Evidence(
        id=f"E{n}",
        paper_id="2305.18290",
        kind="figure",
        claim="DPO의 구조입니다.",
        quote="Figure 1: Overview.",
        anchor="Method",
        figure_id=f"F{n}",
        figure_url=url,
        caption="Figure 1: Overview.",
    )


def evidence_body(plan: SurveyPlan, figures: list[str]) -> str:
    """A post as the evidence engine writes it: ``[@id]`` citations of every
    paper, the ledger's figures and an equation."""
    parts = ["## 들어가며", "선호 학습은 사람의 판단을 모델에 반영하는 방법입니다."]
    for lineage in plan.lineages:
        parts.append(f"## {LINEAGE_TITLES[lineage.id]}")
        for p in plan.papers:
            if p.lineage == lineage.id:
                name = NAMES.get(p.id, p.id)
                parts += [
                    f"### {name}",
                    f"{name}[@{p.id}]는 이전 방법의 불안정성을 줄이려고 제안되었습니다.",
                ]
    for url in figures:
        parts += ["아래 그림은 방법의 구조를 보여줍니다.", f"![방법 구조]({url})"]
    parts += [
        "핵심 손실은 다음과 같습니다.",
        f"$$\n{LOSS}\n$$",
        "이 식은 선호 응답의 확률을 높입니다.",
        "## 맺으며",
        "선호 최적화는 보상 모델 없이도 정책을 학습할 수 있음을 보여 줍니다.",
    ]
    return "\n\n".join(parts)


class FakeEngine:
    """Stands in for ``EvidenceSurveyEngine``: records what
    ``write_survey_with_evidence`` gives it and returns a written survey."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.figures = [DPO_FIGURE]
        self.notes: list[str] = []
        self.error: Exception | None = None

    def __call__(
        self, model: Any, documents: Any, settings: Any, store: Any = None
    ) -> FakeEngine:
        self.created.append({"documents": documents, "settings": settings})
        return self

    async def write(
        self, plan: SurveyPlan, *, references: str = "", guideline: str = ""
    ) -> EvidenceSurvey:
        self.calls.append({"plan": plan, "references": references, "guideline": guideline})
        if self.error is not None:
            raise self.error
        return EvidenceSurvey(
            title=TITLE,
            body=evidence_body(plan, self.figures),
            outline=MagicMock(),
            ledger=[figure_evidence(u, n) for n, u in enumerate(self.figures, 1)],
            plan=plan,
            notes=list(self.notes),
        )


@pytest.fixture
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The write stage wired to fakes: the evidence engine, researcher,
    index, model roles, figure check, S3-less context with the survey spec,
    a publisher whose image fetch is local."""
    writer = fake_writer()
    researcher = FakeResearcher()
    index = FakeIndex({"2203.02155": ABSTRACT})
    index.openalex = MagicMock()
    engine = FakeEngine()
    broken: set[str] = set()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_dict()))
    monkeypatch.setattr(survey_writer, "SurveyWriter", lambda **_: writer)
    monkeypatch.setattr(survey_writer, "WebResearcher", lambda **_: researcher)
    monkeypatch.setattr(survey_writer, "PaperIndex", lambda *_: index)
    monkeypatch.setattr(survey_writer, "DataCiteIndex", MagicMock())
    monkeypatch.setattr(survey_writer, "OpenAlexIndex", MagicMock())
    monkeypatch.setattr(survey_engine, "EvidenceSurveyEngine", engine)
    monkeypatch.setattr(
        evidence_models,
        "role_model",
        lambda _ctx, model_id, effort, thinking: RoleModel(
            invoke_id=model_id, model_label=model_id, effort=effort, thinking=thinking
        ),
    )

    async def unservable(urls: list[str]) -> list[str]:
        return [u for u in urls if u in broken]

    monkeypatch.setattr(evidence_models, "unservable", unservable)

    async def guideline(_: RunContext) -> list[dict[str, str]]:
        return GUIDELINE

    monkeypatch.setattr(survey_writer, "load_translation_guideline", guideline)
    spec = survey_spec()

    def publisher(context: RunContext) -> Publisher:
        return Publisher(
            context.config.resources.github,
            root_dir=tmp_path,
            conventions=spec,
            fetch_image=lambda url: png(),
        )

    session = boto3.Session(region_name="us-east-1")
    context = RunContext(
        config=Config(),
        default_boto_session=session,
        bedrock_boto_session=session,
        root_dir=tmp_path,
        publisher_factory=publisher,
        conventions=spec,
        blog_posts=[],
        blog_tags={},
    )
    return {
        "writer": writer,
        "researcher": researcher,
        "index": index,
        "engine": engine,
        "broken": broken,
        "context": context,
        "request": JobRequest(
            kind="survey_write", plan=str(plan_path), create_pr=False
        ),
        "tmp": tmp_path,
    }


def run_stage(stage: dict[str, Any]) -> Any:
    return asyncio.run(
        survey_writer.run_survey_write_pipeline(
            stage["context"], stage["request"], TokenUsageTracker()
        )
    )


class TestStage:
    def test_post_passes_the_validator(self, stage: dict[str, Any]) -> None:
        outcome = run_stage(stage)
        post = outcome.markdown_path.read_text()
        assert outcome.markdown_path.parent.name == "survey-preference-optimization"
        errors = [
            v
            for v in validate_post(post, survey_spec(), "survey")
            if v.level == "error"
        ]
        assert errors == []
        assert f'title: "{TITLE}"' in post
        assert f'categories: ["{SURVEY_TYPE}", "Language-Models"]' in post
        assert "paper_author" not in post and "paper_doi" not in post
        assert f'description: "{DESCRIPTION}"' in post
        assert "[@" not in post
        # The figure is the paper's, localized, credited; the equation shows.
        assert re.search(
            r"!\[방법 구조\]\(/assets/images/[^)]+\.png\)\n\n\*출처: \[Rafailov", post
        )
        assert LOSS in post and "use_math: true" in post
        # Dated by the plan, so a rerun replaces the same post.
        assert "\ndate: 2026-10-01 09:30:00\n" in post
        assert post.rstrip().endswith("(https://arxiv.org/abs/2305.18290). arXiv.")
        # The engine got the plan, the guideline and the survey's settings.
        engine = stage["engine"]
        [call] = engine.calls
        assert [p.id for p in call["plan"].papers] == [p.id for p in plan().papers]
        assert call["guideline"] == str(GUIDELINE) and call["references"] == ""
        settings = engine.created[0]["settings"]
        assert (settings.chars_per_core, settings.chars_per_supporting) == (3_000, 800)
        assert settings.title_max_length == survey_spec().seo_title_max_length
        assert settings.reader_tag == "anthropic-claude-sonnet-5-5"
        # The supporting paper's abstract reaches the engine's documents.
        documents = engine.created[0]["documents"]
        assert documents.abstracts == {"2203.02155": ABSTRACT.abstract}

    def test_post_passes_the_blog_lint(self, stage: dict[str, Any]) -> None:
        lint = BLOG / "script" / "lint-posts.rb"
        spec = BLOG / "_data" / "post_conventions.yml"
        if not (shutil.which("ruby") and lint.is_file() and spec.is_file()):
            pytest.skip("the blog checkout or ruby is not available")
        outcome = run_stage(stage)
        copy = stage["tmp"] / "lint" / outcome.markdown_path.name
        copy.parent.mkdir()
        copy.write_text(outcome.markdown_path.read_text())
        done = subprocess.run(
            ["ruby", str(lint), str(copy)], capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stdout + done.stderr

    def test_a_rerun_keeps_the_published_translation_pair(
        self, stage: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        published = BlogPost(
            path="_posts/2026-10-01-preference-optimization.md",
            categories=[SURVEY_TYPE, "Language-Models"],
            lang="ko",
            translation_id="preference-optimization",
        )

        async def find(*args: Any, **kwargs: Any) -> tuple[BlogPost, list[str]]:
            assert kwargs["category"] == SURVEY_TYPE
            return published, []

        monkeypatch.setattr(survey_writer, "find_published_post", find)
        post = run_stage(stage).markdown_path.read_text()
        assert '\nlang: "ko"\ntranslation_id: "preference-optimization"\n' in post

    def test_a_spec_without_the_survey_type_fails_first(
        self, stage: dict[str, Any]
    ) -> None:
        stage["context"].conventions = load_local_conventions().model_copy(
            update={"types": {"Paper Reviews": TypeSpec(paper=True)}}
        )
        with pytest.raises(ValueError, match=f"no '{SURVEY_TYPE}' type"):
            run_stage(stage)
        assert stage["engine"].calls == []

    def test_figures_arxiv_does_not_serve_are_left_out(
        self, stage: dict[str, Any]
    ) -> None:
        gone = "https://arxiv.org/html/2305.18290v1/x9.png"
        stage["engine"].figures = [DPO_FIGURE, gone]
        stage["broken"].add(gone)
        outcome = run_stage(stage)
        post = outcome.markdown_path.read_text()
        assert "x9.png" not in post and post.count("*출처: [Rafailov") == 1
        assert any(
            "Figures arXiv does not serve" in w and gone in w for w in outcome.warnings
        )

    def test_engine_notes_reach_the_pr(self, stage: dict[str, Any]) -> None:
        stage["engine"].notes = ["Section 'x' kept a flagged sentence."]
        outcome = run_stage(stage)
        assert "Section 'x' kept a flagged sentence." in outcome.warnings

    def test_a_failed_run_keeps_the_plan_notes(self, stage: dict[str, Any]) -> None:
        data = plan_dict()
        data["edges"].append({"from": "2203.02155", "to": "2499.99999"})
        Path(stage["request"].plan).write_text(json.dumps(data))
        stage["engine"].error = RuntimeError("engine down")
        with pytest.raises(RuntimeError, match="engine down") as failed:
            run_stage(stage)
        assert any("citation edge" in n for n in failed.value.run_notes)


class TestReferenceArticles:
    def test_articles_are_read_again_used_and_listed(
        self, stage: dict[str, Any]
    ) -> None:
        data = plan_dict()
        data["schema_version"] = 3
        data["references"] = [
            {"url": ARTICLE, "title": "Lilian Weng: RLHF", "kind": "article"},
            # A paper reference was a candidate of the plan: not read again.
            {"url": "https://arxiv.org/abs/2009.01325", "kind": "paper"},
            {"url": "https://blog.x.io/gone", "title": "Gone"},
        ]
        Path(stage["request"].plan).write_text(json.dumps(data))
        outcome = run_stage(stage)
        post = outcome.markdown_path.read_text()
        assert post.endswith(
            "arXiv.\n\n### 참고 자료\n* [Lilian Weng: RLHF]" f"({ARTICLE})\n"
        )
        errors = [
            v
            for v in validate_post(post, survey_spec(), "survey")
            if v.level == "error"
        ]
        assert errors == []
        asked = stage["researcher"].asked
        assert ARTICLE in asked and "https://blog.x.io/gone" in asked
        assert "https://arxiv.org/abs/2009.01325" not in asked
        assert any(
            "Reference articles that could not be read" in w
            and "https://blog.x.io/gone" in w
            for w in outcome.warnings
        )
        # The engine gets the read article's excerpts as reference material.
        assert "Reward hacking" in stage["engine"].calls[0]["references"]


class TestSurveyBody:
    def test_citations_images_and_figures_are_settled(self) -> None:
        offered = "https://arxiv.org/html/2305.18290v1/x1.png"
        body = (
            "## 들어가며\n\nDPO[@2305.18290]와 추가 연구[@2501.99999], "
            "[ORPO](https://arxiv.org/abs/2403.07691)를 봅니다.\n\n"
            f"![구조]({offered})\n\n![로고](https://x.io/logo.png)\n"
        )
        survey = survey_writer.Survey(
            title=TITLE,
            body=body,
            plan=plan(),
            figures={
                offered: ("2305.18290", "Figure 1: Overview."),
                "https://x.io/other.png": ("2499.00001", "not a plan paper"),
            },
        )
        text, notes = survey_body(survey, "Korean")
        assert "([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290))" in text
        assert "2501.99999" not in text and "ORPO를" in text
        assert "logo.png" not in text
        assert f"]({offered})\n\n*출처: [Rafailov et al., 2023]" in text
        assert any("2501.99999" in n for n in notes)
        assert any("https://arxiv.org/abs/2403.07691" in n for n in notes)


class TestReviewFixes:
    def test_a_hand_added_paper_takes_its_place_in_time(self) -> None:
        data = plan_dict()
        data["papers"].insert(
            0, {"id": "2402.01306", "lineage": "dpo", "role": "supporting"}
        )
        data["papers"][1]["year"] = None  # a plan paper without a year stays
        loaded, _ = load_plan(json.dumps(data), FakeIndex({"2402.01306": KTO}))
        assert [p.id for p in loaded.papers] == [
            "2203.02155",
            "2305.18290",
            "2402.01306",
            "1707.06347",
        ]

    @pytest.mark.parametrize(
        "edit",
        [
            lambda d: d.update(edges=["2305.18290 -> 1707.06347"]),
            lambda d: d.update(lineages="rlhf"),
            lambda d: d["papers"][0].pop("id"),
        ],
    )
    def test_malformed_plan_parts_are_rejected(self, edit: Any) -> None:
        data = plan_dict()
        edit(data)
        with pytest.raises(SurveyInputError):
            load_plan(json.dumps(data), FakeIndex({}))

    def test_hand_typed_links_to_other_papers(self) -> None:
        papers = {p.id: p for p in plan().papers}
        text = (
            "DPO[@2305.18290]와 [ORPO](https://arxiv.org/abs/2403.07691), "
            "[DPO 원문](https://arxiv.org/abs/2305.18290), `[x](https://doi.org/1)`"
        )
        assert foreign_paper_links(text, papers.values()) == [
            "https://arxiv.org/abs/2403.07691"
        ]
        assert unlink_foreign_papers(text, papers.values()) == text.replace(
            "[ORPO](https://arxiv.org/abs/2403.07691)", "ORPO"
        )

    def test_a_description_the_validator_rejects_is_dropped(
        self, stage: dict[str, Any]
    ) -> None:
        writer = stage["writer"]
        writer.finalization_chain = FakeChain(
            {
                "category": "Language-Models",
                "controlled_tags": "Alignment",
                "keywords": "DPO",
                "description": "짧은 `설명`",
            }
        )
        outcome = run_stage(stage)
        assert len(writer.finalization_chain.payloads) == 2
        assert "description:" not in outcome.markdown_path.read_text()
        assert any("description still breaks" in w for w in outcome.warnings)


class TestStagingValidation:
    def test_an_invalid_plan_is_not_uploaded(self, tmp_path: Path) -> None:
        s3 = MagicMock(bucket_name="bucket")
        data = plan_dict()
        data["papers"][0]["lineage"] = "ghost"
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(data))
        with pytest.raises(SurveyInputError, match="unknown lineage"):
            stage_plan_for_batch(
                JobRequest(kind="survey_write", plan=str(path)), s3, "x"
            )
        s3.put_text.assert_not_called()
        # A paper added by hand gets its metadata in the job: it passes here.
        data = plan_dict()
        data["papers"].append({"id": "2402.01306", "lineage": "dpo", "role": "core"})
        path.write_text(json.dumps(data))
        stage_plan_for_batch(JobRequest(kind="survey_write", plan=str(path)), s3, "x")
        s3.put_text.assert_called_once()


def test_a_rerun_replaces_the_post_its_plan_published(
    stage: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun of the plan under a new model title updates the post the
    plan's branch published: its file name (URL), date, topic and image
    names stay, and the post passes the validator under that name."""
    from datetime import UTC, datetime

    from scholar_lens.configs import Github
    from scholar_lens.domain.conventions import BlogPost
    from tests.test_publisher import FakeGithub, FakeRepo

    old = "_posts/2026-10-02-preference-learning-for-llms.md"
    repo = FakeRepo({})
    repo.merge("paper-reviews/survey-preference-optimization", [old])
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.setattr(FakeGithub, "repo", repo)
    monkeypatch.setattr("scholar_lens.src.publisher.Github", FakeGithub)
    context = stage["context"]
    context.blog_posts = [
        BlogPost(
            path=old,
            date=datetime(2026, 10, 2, 5, tzinfo=UTC),
            categories=[SURVEY_TYPE, "Finetuning"],
        )
    ]
    github = Github(
        enabled=True,
        repo_name="owner/blog",
        author_name="Bot",
        author_email="bot@example.com",
    )
    context.publisher_factory = lambda ctx: Publisher(
        github,
        root_dir=stage["tmp"],
        conventions=survey_spec(),
        site_posts=ctx.blog_posts,
        fetch_image=lambda url: png(),
    )
    outcome = run_stage(stage)
    post = outcome.markdown_path.read_text()
    assert outcome.markdown_path.name == "2026-10-02-preference-learning-for-llms.md"
    assert "\ndate: 2026-10-02 05:00:00\n" in post
    assert f'categories: ["{SURVEY_TYPE}", "Finetuning"]' in post
    assert "cover: /assets/images/finetuning.jpg" in post
    assert "](/assets/images/preference-learning-for-llms-1.png)" in post
    errors = [
        v
        for v in validate_post(
            post, survey_spec(), "survey", file_name=outcome.markdown_path.name
        )
        if v.level == "error"
    ]
    assert errors == []
