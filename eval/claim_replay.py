"""Replay the sections engine's claim check on finished posts and score it
against the hand-fixed sentences of ``eval/sets/claim_truth.yaml``.

    # Paid: one Bedrock call per case (about $0.3-0.6 each on Opus 5.5).
    poetry run python -m eval.claim_replay run WORK_DIR [--only NAME ...]
    # Free: score a saved replay, or check that every anchor is in its post.
    poetry run python -m eval.claim_replay score WORK_DIR/replay.json
    poetry run python -m eval.claim_replay anchors WORK_DIR

``run`` builds the check as the engine does (``PaperTextCheckPrompt``, the
configured evaluation model and thinking effort, number sources capped at
32k characters for a review and 16k for a summary) and reads the paper from
arXiv HTML, not the run's parsed PDF, so a figure reading the run had is
missing here. Posts and papers are cached in WORK_DIR, which should be a
private scratch directory (``mktemp -d "$HOME/.cache/claim-replay.XXXX"``).

A check line counts as finding a hand-fixed sentence when the sentence it
copies (before ``||``) holds the entry's anchor, whitespace and markup aside.
Flags on a clean baseline are false-alarm candidates; flags elsewhere that
match no entry are unconfirmed (the hand fixes are not exhaustive).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
import httpx
import yaml
from bs4 import BeautifulSoup
from langchain_core.runnables import RunnableLambda

from scholar_lens.adapters.bedrock.chat_models import BedrockLanguageModelFactory
from scholar_lens.adapters.bedrock.output_parsers import HTMLTagOutputParser
from scholar_lens.adapters.bedrock.prompts.base import with_cache_points
from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
from scholar_lens.adapters.documents.html import extract_text_from_html
from scholar_lens.configs import Config
from scholar_lens.domain.markdown.front_matter import split_front_matter
from scholar_lens.domain.prompt_safety import neutralize_prompt_tags
from scholar_lens.domain.text import parse_lines
from scholar_lens.domain.validation import number_sources
from scholar_lens.sections_engine.prompts.review import PaperTextCheckPrompt

TRUTH = Path(__file__).parent / "sets" / "claim_truth.yaml"
REPO = Path(__file__).parent.parent
# The engine's caps (sections_engine/explainer.py, summarizer.py).
SOURCE_CAPS = {"review": 32_000, "summary": 16_000}
TEXT_NAMES = {"review": "review body", "summary": "summary"}

_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_MARKS = re.compile(r"[*_`$]")


def squash(text: str) -> str:
    """``text``'s words: whitespace folded, emphasis/code/math marks and link
    targets dropped."""
    return " ".join(_MARKS.sub("", _LINK.sub(r"\1", text)).split())


def disputed(line: str) -> str:
    """The sentence a check line copies (before ``||``)."""
    return squash(line.split("||", 1)[0].strip().removeprefix("-").strip(' "“”'))


@dataclass
class CaseScore:
    name: str
    flags: int
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    clean: bool = False


def score_case(name: str, case: dict[str, Any], problems: list[str]) -> CaseScore:
    """Which truth entries of ``case`` the check ``problems`` found."""
    truth = case.get("truth") or []
    score = CaseScore(name, len(problems), clean=bool(case.get("clean")))
    matched: set[int] = set()
    for entry in truth:
        anchor = squash(entry["anchor"])
        hits = [i for i, p in enumerate(problems) if anchor in disputed(p)]
        (score.found if hits else score.missed).append(entry["id"])
        matched.update(hits)
    score.other = [p for i, p in enumerate(problems) if i not in matched]
    return score


def load_truth(path: Path = TRUTH) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise SystemExit(f"{path}: unknown version {data.get('version')!r}")
    return dict(data["cases"])


def report(scores: list[CaseScore]) -> dict[str, Any]:
    truth = sum(len(s.found) + len(s.missed) for s in scores)
    found = sum(len(s.found) for s in scores)
    return {
        "recall": f"{found}/{truth}",
        "clean_flags": sum(s.flags for s in scores if s.clean),
        "unconfirmed_flags": sum(len(s.other) for s in scores if not s.clean),
        "cases": [s.__dict__ for s in scores],
    }


# ---- I/O (run and anchors) -------------------------------------------------


def _post_text(work: Path, ref: str) -> str:
    if not ref.startswith("s3:"):
        return (REPO / ref).read_text(encoding="utf-8")
    cached = work / Path(ref[3:]).name
    if not cached.exists():
        res = Config.load().resources
        key = "/".join(p for p in (res.s3_prefix.strip("/"), ref[3:]) if p)
        session = boto3.Session(profile_name=res.profile_name)
        body = session.client("s3", region_name=res.default_region_name).get_object(
            Bucket=res.s3_bucket_name, Key=key
        )["Body"]
        cached.write_bytes(body.read())
    return cached.read_text(encoding="utf-8")


def _body(post: str) -> str:
    return split_front_matter(post)[1]


def _paper_text(work: Path, arxiv_id: str) -> str:
    cached = work / f"paper-{arxiv_id}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    html = httpx.get(
        f"https://arxiv.org/html/{arxiv_id}", timeout=60, follow_redirects=True
    )
    html.raise_for_status()
    soup = BeautifulSoup(html.text, "html.parser")
    text = extract_text_from_html(str(soup.select_one(".ltx_page_main") or soup.body))
    cached.write_text(text, encoding="utf-8")
    return text


async def _run(work: Path, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cfg = Config.load()
    session = boto3.Session(
        profile_name=cfg.resources.profile_name,
        region_name=cfg.resources.bedrock_region_name,
    )
    factory = BedrockLanguageModelFactory(boto_session=session)
    tracker = TokenUsageTracker()
    models = {
        "review": (
            cfg.explanation.paper_evaluation_model_id,
            cfg.explanation.thinking_effort,
        ),
        "summary": (cfg.summary.summary_model_id, cfg.summary.thinking_effort),
    }
    out: dict[str, Any] = {"problems": {}, "models": {}}
    for name, case in cases.items():
        kind = case["kind"]
        model_id, effort = models[kind]
        llm = factory.get_model(
            model_id,
            temperature=0.0,
            enable_thinking=True,
            thinking_effort=effort,
            supports_1m_context_window=True,
            callbacks=[tracker],
        )
        chain = (
            PaperTextCheckPrompt.get_prompt()
            | RunnableLambda(with_cache_points(PaperTextCheckPrompt))
            | llm
            | HTMLTagOutputParser(tag_names=PaperTextCheckPrompt.output_variables)
        )
        paper = neutralize_prompt_tags(_paper_text(work, case["arxiv"]))
        body = _body(_post_text(work, case["post"]))
        result = await chain.ainvoke(
            {
                "paper_text": paper,
                "number_sources": number_sources(
                    body, paper, max_chars=SOURCE_CAPS[kind]
                )
                or "- none",
                "text": neutralize_prompt_tags(body),
                "text_name": TEXT_NAMES[kind],
                "language": "Korean",
            }
        )
        out["problems"][name] = parse_lines(result.get("problems"))
        out["models"][name] = str(model_id)
        print(f"{name}: {len(out['problems'][name])} problems", flush=True)
    out["cost_usd"] = round(tracker.cost_usd, 2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="replay the check (paid Bedrock calls)")
    run.add_argument("work", type=Path)
    run.add_argument("--only", nargs="*", default=None)
    score = sub.add_parser("score", help="score a saved replay")
    score.add_argument("replay", type=Path)
    anchors = sub.add_parser("anchors", help="check every anchor is in its post")
    anchors.add_argument("work", type=Path)
    args = parser.parse_args(argv)
    cases = load_truth()

    if args.command == "anchors":
        missing = [
            f"{name}/{entry['id']}"
            for name, case in cases.items()
            for entry in case.get("truth") or []
            if squash(entry["anchor"])
            not in squash(_body(_post_text(args.work, case["post"])))
        ]
        print("\n".join(missing) or "every anchor is in its post")
        return 1 if missing else 0
    if args.command == "run":
        chosen = {k: v for k, v in cases.items() if not args.only or k in args.only}
        replay = asyncio.run(_run(args.work, chosen))
        path = args.work / "replay.json"
        path.write_text(json.dumps(replay, ensure_ascii=False, indent=1), "utf-8")
        print(f"cost_usd={replay['cost_usd']} -> {path}")
        args.replay = path
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    scores = [
        score_case(name, cases[name], problems)
        for name, problems in replay["problems"].items()
        if name in cases
    ]
    print(json.dumps(report(scores), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
