"""The pairwise decision rule (eval/JUDGE.md, section 0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.compare import compare, confirm_items


def _judge(run: Path, item: str, session: int, overall: str, f: str = "tie") -> None:
    data = {"item": item, "preference": {"overall": overall, "per_criterion": {"F": f}}}
    (run / f"{item}-{session}.json").write_text(json.dumps(data))


def test_two_of_three_sessions_decide_an_item_and_an_f_loss_vetoes(
    tmp_path: Path,
) -> None:
    mapping = {"a/X": "new", "a/Y": "old", "b/X": "old", "b/Y": "new"}
    (tmp_path / "blind-mapping.json").write_text(json.dumps(mapping))
    for s, pick in enumerate(["X", "X", "Y"]):  # a: new 2, old 1
        _judge(tmp_path, "a", s, pick)
    for s, pick in enumerate(["Y", "tie", "X"]):  # b: new 1, old 1, tie 1
        _judge(tmp_path, "b", s, pick)
    out = compare(tmp_path, {"a", "b"})
    assert out["items"]["a"]["winner"] == "new"
    assert out["items"]["b"]["winner"] == "tie"
    assert out["decision"] == "new"
    for s in range(3):  # new loses item b on F in every session
        _judge(tmp_path, "b", s, "tie", f="X")
    assert compare(tmp_path, {"a", "b"})["decision"] == "no measurable difference"


def test_only_confirm_items_decide_and_bad_runs_fail(tmp_path: Path) -> None:
    mapping = {"a/X": "new", "a/Y": "old", "b/X": "old", "b/Y": "new"}
    (tmp_path / "blind-mapping.json").write_text(json.dumps(mapping))
    for s in range(3):
        _judge(tmp_path, "a", s, "X")  # new wins a
        _judge(tmp_path, "b", s, "X")  # old wins b
    assert compare(tmp_path, {"a"})["decision"] == "new"
    assert compare(tmp_path, {"a"})["items"]["b"]["decides"] is False
    with pytest.raises(ValueError, match="not judged: c"):
        compare(tmp_path, {"a", "c"})
    _judge(tmp_path, "b", 3, "X")
    with pytest.raises(ValueError, match="4 sessions"):
        compare(tmp_path, {"a"})
    _judge(tmp_path, "b", 3, "Z")
    (tmp_path / "b-0.json").unlink()
    with pytest.raises(ValueError, match="not in the blind mapping"):
        compare(tmp_path, {"a"})


def test_the_baseline_set_has_confirm_items() -> None:
    assert {"review-mamba", "review-flashattention2", "review-kan"} <= confirm_items(
        Path("eval/sets/baseline.yaml")
    )
