"""Unblind a pairwise evaluation run and count it (eval/JUDGE.md, section 0).

    poetry run python -m eval.compare eval/runs/<name>

The run directory holds ``blind-mapping.json`` ({"<item>/X": "<version>",
"<item>/Y": "<version>"}) and one judgment file per session,
``<item>[-<session>].json``, each with ``preference.overall`` ("X", "Y" or
"tie") and per-criterion preferences.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sessions(run: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(run.glob("*.json")):
        if path.name == "blind-mapping.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "preference" in data and "item" in data:
            out[data["item"]].append(data)
    return out


def compare(run: Path) -> dict[str, Any]:
    """Per item: the sessions' votes by version and the item's winner (two
    of three sessions, else a tie), and whether a version lost on F; then
    the totals and the decision."""
    mapping = json.loads((run / "blind-mapping.json").read_text(encoding="utf-8"))
    items: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    lost_on_f: set[str] = set()
    for item, sessions in _sessions(run).items():
        votes: Counter[str] = Counter()
        f_votes: Counter[str] = Counter()
        for judgment in sessions:
            preference = judgment["preference"]
            pick = preference.get("overall", "tie")
            votes[mapping.get(f"{item}/{pick}", "tie")] += 1
            f_pick = preference.get("per_criterion", {}).get("F", "tie")
            f_votes[mapping.get(f"{item}/{f_pick}", "tie")] += 1
        need = len(sessions) // 2 + 1
        winner = next((v for v, n in votes.items() if v != "tie" and n >= need), "tie")
        f_winner = next(
            (v for v, n in f_votes.items() if v != "tie" and n >= need), "tie"
        )
        if f_winner != "tie":
            versions = {mapping[f"{item}/X"], mapping[f"{item}/Y"]}
            lost_on_f |= versions - {f_winner}
        totals[winner] += 1
        items[item] = {
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
    args = parser.parse_args(argv)
    json.dump(compare(args.run), sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
