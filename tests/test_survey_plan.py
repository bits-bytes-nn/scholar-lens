"""Deterministic parts of the survey plan stage: ids, record merging, citation
edges, ranking, model-output parsing and checks, and the plan file."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from scholar_lens.src.survey_plan import (
    NO_CORE_PAPER,
    Candidate,
    CitationData,
    PlanReference,
    SourceRecord,
    SurveyInputError,
    SurveyPlan,
    SurveyPlanError,
    SurveyScope,
    SurveySelection,
    TimeRange,
    add_citation_data,
    build_plan,
    check_selection,
    chronological,
    citation_edges,
    cited_paper_ids,
    in_degrees,
    in_time_range,
    merge_records,
    normalize_doi,
    paper_id_of,
    parse_paper_id,
    parse_tagged_json,
    rank_candidates,
    reference_kind,
    semantic_scholar_id,
    shortlist,
    shortlist_digest,
)

TODAY = date(2026, 10, 14)
DPO = "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"


def rec(title: str, **kwargs: object) -> SourceRecord:
    kwargs.setdefault("provider", "openalex")
    return SourceRecord(title=title, **kwargs)  # type: ignore[arg-type]


def cand(cid: str, **kwargs: object) -> Candidate:
    values: dict[str, object] = {
        "id": cid,
        "title": f"Paper {cid}",
        "authors": [],
        "published": date(2023, 1, 1),
        "venue": None,
        "arxiv_id": None,
        "doi": None,
        "abstract": None,
        "openalex_citations": None,
        "keys": {f"openalex:{cid}"},
        "reference_keys": set(),
        "found_by": set(),
    }
    values.update(kwargs)
    return Candidate(**values)  # type: ignore[arg-type]


class TestIds:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("2305.18290", "2305.18290"),
            ("2305.18290v3", "2305.18290"),
            ("https://arxiv.org/abs/2305.18290v2", "2305.18290"),
            ("arXiv:2305.18290", "2305.18290"),
            ("10.48550/arXiv.2305.18290", "2305.18290"),
            (
                "https://doi.org/10.18653/V1/2024.emnlp-main.626",
                "10.18653/v1/2024.emnlp-main.626",
            ),
            ("doi:10.1109/CVPR.2024.00786", "10.1109/cvpr.2024.00786"),
        ],
    )
    def test_paper_ids_become_arxiv_ids_or_dois(
        self, value: str, expected: str
    ) -> None:
        assert parse_paper_id(value) == expected

    @pytest.mark.parametrize(
        "value", ["DPO", "https://example.com/paper.pdf", "11.1/x"]
    )
    def test_other_values_are_rejected(self, value: str) -> None:
        with pytest.raises(SurveyInputError):
            parse_paper_id(value)

    @pytest.mark.parametrize(
        ("url", "kind"),
        [
            ("https://arxiv.org/abs/2305.18290v2", "paper"),
            ("https://arxiv.org/pdf/2305.18290", "paper"),
            ("https://doi.org/10.18653/v1/2024.emnlp-main.626", "paper"),
            ("https://huggingface.co/blog/rlhf", "article"),
            ("https://example.com/2305.18290", "article"),
        ],
    )
    def test_reference_kind(self, url: str, kind: str) -> None:
        assert reference_kind(url) == kind
        assert (paper_id_of(url) is not None) == (kind == "paper")

    def test_cited_papers_from_links_and_text(self) -> None:
        links = [
            "https://arxiv.org/abs/2203.02155",
            "https://arxiv.org/pdf/2203.02155v3",  # the same paper
            "http://arxiv.org/pdf/1707.06347.pdf",
            "https://arxiv.org/html/2402.01306v2",
            "https://ar5iv.labs.arxiv.org/html/1706.03741",
            "https://doi.org/10.18653/v1/2024.emnlp-main.626",
            "https://doi.org/10.48550%2FarXiv.2009.01325",  # an arXiv DOI
            "https://doi.org/10.1234",  # not a DOI
            "https://arxiv.org/list/cs.CL/recent",  # a listing, no paper
            "https://huggingface.co/blog/dpo-trl",
            "https://arxiv.org/abs/2305.18290",  # the page's own paper
        ]
        text = (
            "See Ouyang et al. (arXiv:2203.02155) and arXiv: 2210.10760v1, (2401.0001)."
        )
        assert cited_paper_ids(links, text, own="2305.18290") == [
            "2203.02155",
            "1707.06347",
            "2402.01306",
            "1706.03741",
            "10.18653/v1/2024.emnlp-main.626",
            "2009.01325",
            "2210.10760",
        ]
        assert cited_paper_ids([], "no ids here") == []

    def test_normalize_doi_needs_a_doi(self) -> None:
        assert normalize_doi(None) is None
        assert normalize_doi("not a doi") is None


class TestMerge:
    def test_arxiv_and_venue_records_of_one_paper_merge(self) -> None:
        venue = rec(
            DPO,
            doi="10.52202/075280-2338",
            openalex_id="W1",
            citation_count=460,
            venue="NeurIPS",
            published=date(2023, 1, 1),
            found_by={"query:a"},
        )
        preprint = rec(
            DPO,
            arxiv_id="2305.18290",
            openalex_id="W2",
            citation_count=293,
            found_by={"query:b"},
        )
        arxiv = rec(
            DPO,
            provider="arxiv",
            arxiv_id="2305.18290",
            authors=["Rafael Rafailov"],
            published=date(2023, 5, 29),
            abstract="We propose DPO.",
        )
        [paper] = merge_records([venue, preprint, arxiv])
        assert paper.id == "2305.18290"
        assert paper.doi == "10.52202/075280-2338"
        assert paper.published == date(2023, 5, 29)  # arXiv's v1, not YYYY-01-01
        assert paper.venue == "NeurIPS"
        assert paper.openalex_citations == 460
        assert paper.authors == ["Rafael Rafailov"]
        assert paper.search_hits == 2
        assert {"openalex:W1", "openalex:W2", "arxiv:2305.18290"} <= paper.keys

    def test_arxiv_title_replaces_a_wrong_index_title(self) -> None:
        # OpenAlex files some arXiv ids under another work's title.
        wrong = rec("Diagnosing Non-Intermittent Anomalies", arxiv_id="1707.06347")
        arxiv = rec(
            "Proximal Policy Optimization Algorithms",
            provider="arxiv",
            arxiv_id="1707.06347",
        )
        [paper] = merge_records([wrong, arxiv])
        assert paper.title == "Proximal Policy Optimization Algorithms"

    def test_a_record_matching_two_papers_joins_them(self) -> None:
        title = "Training language models to follow instructions with human feedback"
        venue = rec(title, doi="10.52202/068431-2011", openalex_id="W1")
        mislabeled = rec("Discernment and Social Learning", arxiv_id="2203.02155")
        arxiv = rec(title, provider="arxiv", arxiv_id="2203.02155")
        [paper] = merge_records([venue, mislabeled, arxiv])
        assert paper.id == "2203.02155"
        assert paper.doi == "10.52202/068431-2011"
        assert paper.title == title

    def test_shared_doi_merges_despite_different_titles(self) -> None:
        a = rec("A title", doi="10.1/x", openalex_id="W1")
        b = rec("Another title", doi="10.1/x", openalex_id="W2")
        assert len(merge_records([a, b])) == 1

    def test_matching_titles_with_two_arxiv_ids_stay_apart(self) -> None:
        a = rec("Scaling Laws", provider="arxiv", arxiv_id="2001.08361")
        b = rec("Scaling Laws", provider="arxiv", arxiv_id="2203.15556")
        assert [c.id for c in merge_records([a, b])] == ["2001.08361", "2203.15556"]

    def test_sibling_titles_stay_apart(self) -> None:
        a = rec("LoRA: Low-Rank Adaptation of Large Language Models", openalex_id="W1")
        b = rec("LoRA+: Low-Rank Adaptation of Large Language Models", openalex_id="W2")
        assert len(merge_records([a, b])) == 2

    def test_order_follows_first_records(self) -> None:
        records = [rec(f"Topic {n} study", openalex_id=f"W{n}") for n in (3, 1, 2)]
        assert [c.id for c in merge_records(records)] == ["W3", "W1", "W2"]

    def test_candidate_without_arxiv_or_doi_uses_the_openalex_id(self) -> None:
        [paper] = merge_records([rec("Old paper", openalex_id="W9")])
        assert paper.id == "W9"
        assert paper.url == "https://openalex.org/W9"

    def test_urls_and_author_labels(self) -> None:
        names = ["A One", "B Two", "C Three", "D Four"]
        [a] = merge_records([rec("X", provider="arxiv", arxiv_id="2305.18290")])
        [b] = merge_records([rec("Y", doi="10.1/y", authors=names, openalex_id="W1")])
        assert a.url == "https://arxiv.org/abs/2305.18290"
        assert b.url == "https://doi.org/10.1/y"
        assert b.authors_label == "A One, B Two, C Three et al."
        assert a.venue == "arXiv"


class TestEdgesAndRanking:
    def test_edges_link_pool_papers_through_any_key(self) -> None:
        a = cand("A", reference_keys={"openalex:B", "openalex:Z", "arxiv:2305.18290"})
        b = cand("B")
        c = cand("2305.18290", keys={"arxiv:2305.18290", "openalex:C"})
        assert citation_edges([a, b, c]) == [("A", "2305.18290"), ("A", "B")]
        assert in_degrees(citation_edges([a, b, c])) == {"2305.18290": 1, "B": 1}

    def test_a_key_two_candidates_share_links_neither(self) -> None:
        # Two arXiv papers sharing a DOI stay apart (merge_records); a
        # reference by that DOI cannot tell which one is cited.
        recs = [
            rec("Alpha method", arxiv_id="2301.00001", doi="10.1/a", openalex_id="W1"),
            rec("Beta method", arxiv_id="2302.00002", doi="10.1/a", openalex_id="W2"),
            rec("Gamma", openalex_id="W3", references=["W1"]),
        ]
        merged = merge_records(recs)
        assert [c.id for c in merged] == ["2301.00001", "2302.00002", "W3"]
        merged[2].reference_keys.add("doi:10.1/a")
        assert citation_edges(merged) == [("W3", "2301.00001")]

    def test_self_citations_are_no_edges(self) -> None:
        a = cand("A", reference_keys={"openalex:A"})
        assert citation_edges([a]) == []

    def test_semantic_scholar_data_adds_counts_venue_and_edges(self) -> None:
        a = cand("A", arxiv_id="2305.18290", openalex_citations=300)
        b = cand("10.1/b", doi="10.1/b", keys={"doi:10.1/b"})
        assert semantic_scholar_id(a) == "ARXIV:2305.18290"
        assert semantic_scholar_id(b) == "DOI:10.1/b"
        assert semantic_scholar_id(cand("W1")) is None
        add_citation_data(a, CitationData(5000, "NeurIPS", {"doi:10.1/b"}))
        add_citation_data(b, None)
        assert a.citation_count == 5000 and a.venue == "NeurIPS"
        assert citation_edges([a, b]) == [("A", "10.1/b")]

    @pytest.mark.parametrize(
        ("venue", "s2_venue", "expected"),
        [
            ("ICML", "arXiv.org", "ICML"),
            ("arXiv", "arXiv.org", "arXiv"),
            (None, "arXiv.org", "arXiv.org"),
            ("arXiv", "NeurIPS", "NeurIPS"),
        ],
    )
    def test_semantic_scholar_preprint_venue_keeps_an_existing_venue(
        self, venue: str | None, s2_venue: str, expected: str
    ) -> None:
        a = cand("A", arxiv_id="2305.18290", venue=venue)
        add_citation_data(a, CitationData(1, s2_venue))
        assert a.venue == expected

    def test_ranking_is_evidence_then_citations_per_year_then_id(self) -> None:
        old = cand("old", published=date(2017, 6, 1), openalex_citations=900)
        new = cand("new", published=date(2025, 6, 1), openalex_citations=200)
        tie = cand("tie", published=date(2025, 6, 1), openalex_citations=200)
        hit = cand("hit", found_by={"query:q", "query:r", "landmark:t"})
        cited = cand("cited")
        edges = [("old", "cited"), ("new", "cited")]
        ranked = rank_candidates([old, new, tie, hit, cited], edges, today=TODAY)
        assert [c.id for c in ranked] == ["cited", "hit", "new", "tie", "old"]
        # Same input, same order.
        again = rank_candidates([tie, cited, hit, old, new], edges, today=TODAY)
        assert [c.id for c in again] == [c.id for c in ranked]

    def test_references_break_ties_after_topic_evidence(self) -> None:
        referenced = cand(
            "ref", found_by={"reference:https://a.io/p", "reference:https://b.io/q"}
        )
        searched = cand("hit", found_by={"query:q"})
        popular = cand("pop", openalex_citations=10_000)
        ranked = rank_candidates([popular, searched, referenced], [], today=TODAY)
        # A search hit outranks any number of references (a blog post links
        # off-topic papers too); references outrank citations per year.
        assert [c.id for c in ranked] == ["hit", "ref", "pop"]
        assert referenced.referenced == 2 and referenced.search_hits == 0
        # Among equal topic evidence, the more referenced paper goes first.
        both = cand("both", found_by={"query:q", "reference:https://a.io/p"})
        ranked = rank_candidates([searched, both], [], today=TODAY)
        assert [c.id for c in ranked] == ["both", "hit"]
        # Referenced papers are no surer than others: the rank decides.
        assert [c.id for c in shortlist(ranked, 1)] == ["both"]

    def test_search_hits_count_topic_queries_only(self) -> None:
        c = cand("x", found_by={"query:a", "landmark:b", "cites:y", "required"})
        assert c.search_hits == 1 and c.landmark
        assert not cand("y", found_by={"query:a"}).landmark

    def test_shortlist_keeps_required_papers_and_landmarks(self) -> None:
        ranked = [cand(str(i)) for i in range(5)] + [
            cand("req", required=True),
            cand("lm", found_by={"landmark:t"}),
        ]
        picked = shortlist(ranked, 3)
        assert [c.id for c in picked] == ["0", "req", "lm"]
        assert [c.id for c in shortlist(ranked, 0)] == ["req", "lm"]

    def test_time_range(self) -> None:
        c = cand("c", published=date(2016, 1, 1))
        assert in_time_range(c, None, None)
        assert not in_time_range(c, 2017, None)
        assert in_time_range(c, 2015, 2016)
        assert not in_time_range(cand("u", published=None), 2017, None)
        assert in_time_range(cand("r", published=None, required=True), 2017, 2018)

    def test_chronological_puts_undated_last(self) -> None:
        papers = [
            cand("b", published=date(2023, 1, 1)),
            cand("u", published=None),
            cand("a", published=date(2017, 1, 1)),
        ]
        assert [c.id for c in chronological(papers)] == ["a", "b", "u"]

    def test_digest_shows_metadata_and_flags(self) -> None:
        c = cand(
            "2305.18290",
            title=DPO,
            openalex_citations=460,
            required=True,
            abstract="x " * 500,
            found_by={"required", "query:a", "landmark:" + DPO},
        )
        text = shortlist_digest([c], [("A", "2305.18290")], TODAY)
        assert "id: 2305.18290 [REQUIRED] [LANDMARK]" in text
        assert (
            "citations: 460 (~115/yr) | cited by 1 pool papers"
            " | found by 1 topic queries"
        ) in text
        assert len(text) < 800
        flagged = cand("x", found_by={"reference:https://a.io/p", "query:q"})
        assert "- id: x [REFERENCED by 1]\n" in shortlist_digest([flagged], [], TODAY)
        assert "REFERENCED" not in shortlist_digest([cand("y")], [], TODAY)


SCOPE = {
    "topic": "Preference optimization for LLM alignment.",
    "lineages": [
        {"id": "rlhf", "name": "RLHF", "description": "reward model + PPO"},
        {"id": "dpo", "name": "DPO family", "description": "offline objectives"},
    ],
    "queries": ["q1", "q2", "q3", "q4"],
    "landmarks": [DPO],
}


class TestModelOutputs:
    def test_scope_parses_from_tags_and_fences(self) -> None:
        raw = f"thinking…\n<scope>\n```json\n{json.dumps(SCOPE)}\n```\n</scope>"
        scope = parse_tagged_json(raw, "scope", SurveyScope)
        assert scope.landmarks == [DPO]

    def test_the_last_block_wins(self) -> None:
        first = {**SCOPE, "topic": "draft"}
        raw = f"<scope>{json.dumps(first)}</scope> <scope>{json.dumps(SCOPE)}</scope>"
        assert parse_tagged_json(raw, "scope", SurveyScope).topic == SCOPE["topic"]

    @pytest.mark.parametrize(
        "change",
        [
            {"queries": ["only one"]},
            {"lineages": [SCOPE["lineages"][0]]},
            {"lineages": [{"id": "Bad Id", "name": "x", "description": "y"}] * 2},
            # The user's years bound the survey; the scope proposes none.
            {"time_range": {"since": 2017, "until": None}},
            {"extra": 1},
        ],
    )
    def test_invalid_scopes_raise(self, change: dict) -> None:
        with pytest.raises(SurveyPlanError):
            parse_tagged_json(
                f"<scope>{json.dumps({**SCOPE, **change})}</scope>",
                "scope",
                SurveyScope,
            )

    def test_missing_json_raises(self) -> None:
        with pytest.raises(SurveyPlanError, match="no JSON"):
            parse_tagged_json("no answer", "scope", SurveyScope)


def selection(papers: list[tuple[str, str]], **kwargs: object) -> SurveySelection:
    return SurveySelection.model_validate(
        {
            "paper_count_reason": "narrow topic",
            "lineages": kwargs.get(
                "lineages",
                [
                    {"id": "rlhf", "name": "RLHF", "description": "d"},
                    {"id": "dpo", "name": "DPO", "description": "d"},
                ],
            ),
            "papers": [
                {"id": i, "lineage": lin, "role": "core", "reason": "r"}
                for i, lin in papers
            ],
            "excluded": kwargs.get("excluded", []),
        }
    )


POOL = [
    cand("1706.03741", published=date(2017, 6, 12), arxiv_id="1706.03741"),
    cand("2305.18290", published=date(2023, 5, 29), required=True),
    cand("2203.02155", published=date(2022, 3, 4)),
    cand("10.1/x", published=None),
]


class TestCheckSelection:
    def test_a_valid_selection_has_no_problems(self) -> None:
        chosen = selection([("2305.18290", "dpo"), ("1706.03741", "rlhf")])
        assert check_selection(chosen, POOL, min_papers=2, max_papers=3) == []

    def test_every_problem_is_listed(self) -> None:
        chosen = selection(
            [("9999.99999", "dpo"), ("1706.03741", "nope"), ("1706.03741", "rlhf")],
            excluded=[{"id": "ghost", "reason": "r"}],
        )
        problems = check_selection(chosen, POOL, min_papers=2, max_papers=3)
        assert "unknown paper id '9999.99999'" in problems
        # An excluded id outside the shortlist costs no repair call.
        assert not any("ghost" in p for p in problems)
        assert "paper '1706.03741' is selected twice" in problems
        assert "required paper '2305.18290' is not selected" in problems
        assert "paper '1706.03741' has unknown lineage 'nope'" in problems
        assert "1 papers selected; select between 2 and 3" in problems

    def test_too_many_papers(self) -> None:
        chosen = selection([(c.id, "dpo") for c in POOL])
        assert check_selection(chosen, POOL, min_papers=2, max_papers=3) == [
            "4 papers selected; select between 2 and 3"
        ]

    def test_a_small_pool_lowers_the_lower_bound(self) -> None:
        chosen = selection([(c.id, "dpo") for c in POOL])
        assert check_selection(chosen, POOL, min_papers=6, max_papers=25) == []

    def test_a_selection_without_a_core_paper(self) -> None:
        chosen = selection([("2305.18290", "dpo"), ("1706.03741", "rlhf")])
        for paper in chosen.papers:
            paper.role = "supporting"
        assert check_selection(chosen, POOL, min_papers=2, max_papers=3) == [
            NO_CORE_PAPER
        ]

    def test_duplicate_lineage_ids(self) -> None:
        lineage = {"id": "dpo", "name": "DPO", "description": "d"}
        chosen = selection(
            [("2305.18290", "dpo"), ("1706.03741", "dpo")], lineages=[lineage, lineage]
        )
        assert check_selection(chosen, POOL, min_papers=2, max_papers=3) == [
            "lineage ids are not unique"
        ]


def make_plan() -> SurveyPlan:
    chosen = selection(
        [("2305.18290", "dpo"), ("1706.03741", "rlhf"), ("10.1/x", "dpo")],
        lineages=[
            {"id": "rlhf", "name": "RLHF", "description": "d"},
            {"id": "dpo", "name": "DPO", "description": "d"},
            {"id": "unused", "name": "Unused", "description": "d"},
        ],
        excluded=[
            {"id": "2203.02155", "reason": "covered by others"},
            {"id": "2305.18290", "reason": "selected after all"},
            {"id": "9999.00001", "reason": "not a candidate"},
        ],
    )
    return build_plan(
        topic="preference optimization",
        scope=SurveyScope.model_validate(SCOPE),
        selection=chosen,
        pool=POOL,
        edges=[("2305.18290", "1706.03741"), ("2305.18290", "2203.02155")],
        time_range=TimeRange(since=2017),
        sources=["OpenAlex: 3 requests (0 failed)"],
        notes=["a note"],
        created_at=datetime(2026, 10, 14, tzinfo=UTC),
    )


class TestPlan:
    def test_build_orders_papers_and_keeps_edges_among_them(self) -> None:
        plan = make_plan()
        assert [p.id for p in plan.papers] == ["1706.03741", "2305.18290", "10.1/x"]
        assert [lin.id for lin in plan.lineages] == ["rlhf", "dpo"]
        assert [(e.citing, e.cited) for e in plan.edges] == [
            ("2305.18290", "1706.03741")
        ]
        assert [e.id for e in plan.excluded] == ["2203.02155"]
        assert plan.notes == [
            "a note",
            "Dropped 1 excluded id(s) not among the candidates: 9999.00001",
        ]
        assert plan.papers[0].url == "https://arxiv.org/abs/1706.03741"
        assert plan.papers[0].date == "2017-06-12"

    def test_json_round_trip_uses_from_and_to(self) -> None:
        plan = make_plan()
        data = json.loads(plan.to_json())
        assert data["schema_version"] == 3 and data["references"] == []
        assert data["edges"] == [{"from": "2305.18290", "to": "1706.03741"}]
        assert SurveyPlan.model_validate_json(plan.to_json()) == plan

    def test_references_round_trip(self) -> None:
        plan = make_plan().model_copy(
            update={
                "references": [
                    PlanReference(
                        url="https://x.io/post", title="A post", kind="article"
                    )
                ]
            }
        )
        again = SurveyPlan.model_validate_json(plan.to_json())
        assert again.references == plan.references

    def test_a_version_2_plan_loads_without_references(self) -> None:
        data = json.loads(make_plan().to_json())
        del data["references"]
        data["schema_version"] = 2
        plan = SurveyPlan.model_validate(data)
        assert plan.schema_version == 3 and plan.references == []

    def test_a_version_1_plan_converts_its_reference_urls(self) -> None:
        data = json.loads(make_plan().to_json())
        del data["references"]
        data.update(
            schema_version=1,
            references=["https://x.io/a", "https://arxiv.org/abs/2305.18290", "notes"],
        )
        plan = SurveyPlan.model_validate(data)
        assert [(r.url, r.kind) for r in plan.references] == [
            ("https://x.io/a", "article"),
            ("https://arxiv.org/abs/2305.18290", "paper"),
        ]

    def test_a_hand_added_reference_gets_its_kind_from_the_url(self) -> None:
        data = json.loads(make_plan().to_json())
        data["references"] = [
            {"url": "https://doi.org/10.18653/v1/2024.emnlp-main.626"},
            {"url": "https://lilianweng.github.io/posts/2024-07-07-hallucination/"},
        ]
        kinds = [r.kind for r in SurveyPlan.model_validate(data).references]
        assert kinds == ["paper", "article"]

    @pytest.mark.parametrize(
        ("url", "kind"),
        [
            ("https://lilianweng.github.io/posts/2024-07-07-hallucination/", "paper"),
            ("https://arxiv.org/abs/2305.18290", "article"),
            ("https://doi.org/10.48550/arXiv.2305.18290", "article"),
        ],
    )
    def test_a_reference_kind_that_contradicts_the_url_is_rejected(
        self, url: str, kind: str
    ) -> None:
        with pytest.raises(ValidationError, match="kind follows the URL"):
            PlanReference.model_validate({"url": url, "kind": kind})

    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            (lambda d: d["papers"].append(d["papers"][0]), "duplicate paper id"),
            (lambda d: d["papers"][0].update(lineage="x"), "unknown lineage"),
            (
                lambda d: d["lineages"].append({**d["lineages"][0], "id": "e"}),
                "no papers",
            ),
            (lambda d: d["edges"].append({"from": "a", "to": "b"}), "not in the plan"),
            (lambda d: d.update(schema_version=4), "schema_version"),
            (lambda d: d.update(references=[{"url": "x.io/a"}]), "references"),
            (lambda d: d.update(references=["https://x.io/a"]), "references"),
            (lambda d: d.update(papers=[]), "papers"),
            (
                lambda d: [p.update(role="supporting") for p in d["papers"]],
                "no paper has role 'core'",
            ),
        ],
    )
    def test_edited_plans_are_validated(self, edit, message: str) -> None:  # type: ignore[no-untyped-def]
        data = json.loads(make_plan().to_json())
        edit(data)
        with pytest.raises(ValidationError, match=message):
            SurveyPlan.model_validate(data)
