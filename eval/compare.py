"""Unblind a pairwise evaluation run and count it (eval/JUDGE.md, section 0).

    poetry run python -m eval.compare eval/runs/<name> [--set eval/sets/baseline.yaml]

The run directory holds ``blind-mapping.json`` ({"<item>/X": "<version>",
"<item>/Y": "<version>"}) and one judgment file per session,
``<item>[-<session>].json``, each with ``preference.overall`` ("X", "Y" or
"tie") and per-criterion preferences. Only the set's ``confirm`` items
decide; others are reported. An item needs exactly three sessions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

SESSIONS = 3
_PICKS = {"X", "Y", "tie"}


def _sessions(run: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(run.glob("*.json")):
        if path.name == "blind-mapping.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "preference" in data and "item" in data:
            out[data["item"]].append(data)
    return out


def confirm_items(eval_set: Path) -> set[str]:
    """The set's items marked ``split: confirm``."""
    items = yaml.safe_load(eval_set.read_text(encoding="utf-8"))["items"]
    return {name for name, item in items.items() if item.get("split") == "confirm"}


def _version(mapping: dict[str, str], item: str, pick: str) -> str:
    if pick not in _PICKS or (pick != "tie" and f"{item}/{pick}" not in mapping):
        raise ValueError(f"{item}: preference {pick!r} is not in the blind mapping")
    return "tie" if pick == "tie" else mapping[f"{item}/{pick}"]


def compare(run: Path, confirm: set[str]) -> dict[str, Any]:
    """Per item: the sessions' votes by version and the item's winner (two
    of three sessions, else a tie), and whether a version lost on F; then
    the totals and the decision over the ``confirm`` items.

    Raises:
        ValueError: a confirm item not judged, an item without exactly three
            sessions, or a preference the mapping does not resolve.
    """
    mapping = json.loads((run / "blind-mapping.json").read_text(encoding="utf-8"))
    items: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    lost_on_f: set[str] = set()
    judged = _sessions(run)
    if missing := sorted(confirm - set(judged)):
        raise ValueError(f"confirm items not judged: {', '.join(missing)}")
    for item, sessions in judged.items():
        if len(sessions) != SESSIONS:
            raise ValueError(f"{item}: {len(sessions)} sessions, need {SESSIONS}")
        votes: Counter[str] = Counter()
        f_votes: Counter[str] = Counter()
        for judgment in sessions:
            preference = judgment["preference"]
            pick = preference.get("overall", "tie")
            votes[_version(mapping, item, pick)] += 1
            f_pick = preference.get("per_criterion", {}).get("F", "tie")
            f_votes[_version(mapping, item, f_pick)] += 1
        need = len(sessions) // 2 + 1
        winner = next((v for v, n in votes.items() if v != "tie" and n >= need), "tie")
        f_winner = next(
            (v for v, n in f_votes.items() if v != "tie" and n >= need), "tie"
        )
        decides = item in confirm
        if decides and f_winner != "tie":
            versions = {mapping[f"{item}/X"], mapping[f"{item}/Y"]}
            lost_on_f |= versions - {f_winner}
        if decides:
            totals[winner] += 1
        items[item] = {
            "decides": decides,
            "sessions": len(sessions),
            "votes": dict(votes),
            "winner": winner,
            "f_winner": f_winner,
        }
    names = sorted(set(mapping.values()))
    decided = [
        v
        for v in names
        if totals[v] > max((totals[o] for o in names if o != v), default=0)
        and v not in lost_on_f
    ]
    return {
        "items": items,
        "totals": dict(totals),
        "lost_on_f": sorted(lost_on_f),
        "decision": decided[0] if decided else "no measurable difference",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--set", type=Path, default=Path("eval/sets/baseline.yaml"), dest="eval_set"
    )
    args = parser.parse_args(argv)
    result = compare(args.run, confirm_items(args.eval_set))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
