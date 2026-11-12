"""Numbers of a record are grounded on the source's own text (the quote, its
code and LaTeX, a quoted table row's header), never on the reader's scope
or cell; a quoted table row keeps its header for later checks."""

from __future__ import annotations

import pytest

from scholar_lens.domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    evidence_problems,
    table_header,
)

TABLE = (
    "Results on the held-out set.\n\n"
    "| Method | Steps | Top-5 accuracy |\n"
    "| --- | ---: | :---: |\n"
    "| Baseline | 10 | 71.2 |\n"
    "| Ours | 20 | 84.6 |\n\n"
    "Ours trains twice as long and gains thirteen points.\n"
)


def _doc() -> SourceDocument:
    return SourceDocument.from_markdown(
        "P1",
        "A synthetic paper",
        "full_text",
        "# Intro\n\nThe method reaches 84.6 accuracy after training.\n\n"
        f"# Results\n\n{TABLE}",
    )


def _number(**fields: str) -> list[str]:
    record = {
        "kind": "number",
        "claim": "the method reaches 84.6 accuracy",
        "quote": "The method reaches 84.6 accuracy after training.",
        "anchor": "S1",
        "scope": "held-out set, accuracy",
        **fields,
    }
    return evidence_problems(_doc(), **record)


def test_a_number_only_in_scope_is_unsourced() -> None:
    problems = _number(
        claim="the method reaches 84.6 accuracy with 512 samples",
        scope="512 samples, held-out set",
    )
    assert any("claim states numbers" in p and "512" in p for p in problems)


def test_a_cell_number_must_be_in_the_quote() -> None:
    problems = _number(cell="accuracy 90.1", scope="held-out set, 90.1")
    assert any("cell states numbers" in p and "90.1" in p for p in problems)
    assert not _number(cell="accuracy 84.6")


def test_a_cell_number_only_in_code_is_unsourced() -> None:
    problems = _number(kind="code", cell="batch 64", code="ignored", quote="")
    assert any("cell states numbers" in p for p in problems)


def test_a_table_rows_header_grounds_its_numbers_and_is_stored() -> None:
    quote = "| Ours | 20 | 84.6 |"
    header = "| Method | Steps | Top-5 accuracy |"
    assert table_header(TABLE, quote) == header
    # The header's numbers are the source's too; a claim may name the column.
    assert not _number(
        claim="Ours reaches 84.6 top-5 accuracy after 20 steps",
        quote=quote,
        anchor="S2",
    )
    ledger = EvidenceLedger()
    item = ledger.add(
        _doc(),
        kind="number",
        claim="Ours reaches 84.6 accuracy",
        quote=quote,
        anchor="S2",
        scope="held-out set",
    )
    assert item.table_header == header


def test_quotes_that_are_not_a_table_body_row_have_no_header() -> None:
    assert table_header(TABLE, "Ours trains twice as long") == ""
    assert table_header(TABLE, "| Method | Steps | Top-5 accuracy |") == ""
    assert table_header("a | b in prose", "a | b") == ""
    ledger = EvidenceLedger()
    item = ledger.add(
        _doc(),
        kind="finding",
        claim="it gains",
        quote="Ours trains twice as long and gains thirteen points.",
        anchor="S2",
    )
    assert item.table_header == ""


def test_cached_evidence_without_a_header_still_loads() -> None:
    cached = {
        "id": "E1",
        "paper_id": "P1",
        "kind": "number",
        "claim": "c",
        "quote": "q",
        "anchor": "S1",
    }
    assert Evidence.model_validate(cached).table_header == ""


# What the sources write: names with versions, and one real score.
_SOURCES = [
    "We fine-tune Llama 3.1 and compare with Qwen 2.5, Gemini 1.5 and Claude 3.5 "
    "Sonnet; experiments run on Python 3.12 (see Section 4.2). Our model "
    "reaches 88.7% on MMLU.",
]


@pytest.mark.parametrize(
    ("sentence", "stated"),
    [
        # Glued names, whatever the sources.
        ("ResNet-50과 T5, Llama-3.1-8B를 비교했다.", []),
        # A version the sources write after a name, with or without a particle.
        ("이 논문은 Llama 3.1을 기반 모델로 쓴다.", []),
        ("Qwen 2.5는 오픈 가중치 모델이다.", []),
        ("Gemini 1.5의 긴 문맥을 활용한다.", []),
        ("Claude 3.5 Sonnet을 판정 모델로 쓴다.", []),
        ("Python 3.12에서 실행한다.", []),
        # A score is stated (the sources never write it after the name).
        ("GPT-4o는 MMLU 86.4%를 기록했다.", ["86.4"]),
        ("HumanEval 85.4 Pass@1을 얻었다.", ["85.4"]),
        ("BLEU 28.4를 얻었다.", ["28.4"]),
        ("Llama 3.1 70B와 128K 문맥을 쓴다.", ["70B", "128K"]),
        # A pair's number elsewhere is still stated (Section 4.2 is no licence
        # for a 4.2% claim).
        ("정확도가 4.2% 올랐다.", ["4.2"]),
    ],
)
def test_a_version_in_a_name_is_no_stated_number(
    sentence: str, stated: list[str]
) -> None:
    from scholar_lens.domain.evidence_draft import (
        free_numbers,
        named_versions,
        stated_numbers,
    )

    free = free_numbers().with_names(named_versions(_SOURCES))
    assert stated_numbers(sentence, free) == stated


def test_a_cited_sentence_naming_a_versioned_model_is_not_cut() -> None:
    from scholar_lens.domain.evidence import EvidenceLedger, SourceDocument
    from scholar_lens.domain.evidence_draft import (
        free_numbers,
        named_versions,
        unsourced_number_sentences,
    )

    doc = SourceDocument.from_markdown(
        "p", "t", "full_text", "## 4 Results\n\nOur model reaches 88.7% on MMLU.\n"
    )
    ledger = EvidenceLedger()
    ledger.add(
        doc,
        kind="number",
        claim="c",
        quote="Our model reaches 88.7% on MMLU",
        anchor=doc.sections[0].anchor,
        scope="MMLU",
        cell="MMLU 88.7%",
    )
    free = free_numbers().with_names(named_versions(_SOURCES))
    draft = (
        "## 결과\n\nClaude 3.5 Sonnet과 비교해 MMLU에서 88.7%를 기록했습니다 [[E1]].\n"
    )
    assert unsourced_number_sentences(draft, ledger, free, {}) == []


def test_a_marker_with_bracketed_ids_is_read_and_stripped() -> None:
    from scholar_lens.domain.evidence_draft import MARKER, cited_ids

    sentence = "출력 값은 그대로 두고 행렬곱 외 연산만 덜어냅니다 [[E4], [E14]]."
    assert cited_ids(sentence) == ["E4", "E14"]
    assert "[[" not in MARKER.sub("", sentence)
