"""Replay the sections engine's claim check on finished posts and score it
against the hand-fixed sentences of ``eval/sets/claim_truth.yaml``.

    # Paid: one Bedrock call per case (about $0.3-0.6 each on Opus 5.5).
    poetry run python -m eval.claim_replay run WORK_DIR [--only NAME ...]
    # Free: score a saved replay, or check that every anchor is in its post.
    poetry run python -m eval.claim_replay score WORK_DIR/replay.json
    poetry run python -m eval.claim_replay anchors WORK_DIR

``run`` builds the check as the engine does (``PaperTextCheckPrompt``, the
configured evaluation model and thinking effort, number sources capped at
32k characters for a review and 16k for a summary) on the text the engine
checks: a review's article without its TL;DR and closing (those have their
own check; the closing is taken to be the last ``##`` section, so a review
whose closing is missing loses its last section here), a summary without its
reference list. Differences that remain: the paper is the current arXiv HTML
(no version pinned), not the run's parsed PDF, so a figure reading the run
had is missing; the post is the published one, after the run's own claim
fixes, edits and math normalization, where the engine checks the article
before them. Posts and papers are cached in WORK_DIR, a private scratch
directory (``mktemp -d "$HOME/.cache/claim-replay.XXXX"``). A replay keeps
the raw check lines (``score`` drops the withdrawn ones as the engine does)
with the commit, a hash of the check prompt, the scope and the efforts.

A check line counts as finding a hand-fixed sentence when the sentence it
copies (before ``||``) holds the entry's anchor, whitespace, markup and the
line's ellipses aside (as ``domain/quoted.py`` compares). The match is by
sentence, not by reason: a line may dispute the sentence for another
defect than the one fixed. Flags on a clean baseline are false-alarm
candidates; flags elsewhere that match no entry are unconfirmed (the hand
fixes are not exhaustive).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import subprocess
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
from scholar_lens.domain.quoted import disputed, reported_claims, squash
from scholar_lens.domain.text import parse_lines
from scholar_lens.domain.validation import number_sources
from scholar_lens.sections_engine.explainer import BODY_NUMBER_SOURCE_CHARS
from scholar_lens.sections_engine.prompts.review import PaperTextCheckPrompt

TRUTH = Path(__file__).parent / "sets" / "claim_truth.yaml"
REPO = Path(__file__).parent.parent
# The engine's caps: the review body's, and number_sources' default the
# summary check keeps.
SOURCE_CAPS = {
    "review": BODY_NUMBER_SOURCE_CHARS,
    "summary": inspect.signature(number_sources).parameters["max_chars"].default,
}
TEXT_NAMES = {"review": "review body", "summary": "summary"}


def checked_text(kind: str, body: str) -> str:
    """The part of a published post's body the engine's check reads: for a
    review, the article between the TL;DR's rule and the closing assessment
    (the last ``##``; the TL;DR and closing have their own check); for a
    summary, everything before the reference list's rule."""
    lines = body.splitlines()
    rules = [i for i, line in enumerate(lines) if line.strip() == "- - -"]
    if kind == "review":
        start = rules[0] + 1 if rules else 0
        headings = [i for i, line in enumerate(lines) if line.startswith("## ")]
        end = headings[-1] if headings and headings[-1] > start else len(lines)
        return "\n".join(lines[start:end]).strip()
    end = rules[-1] if rules else len(lines)
    return "\n".join(lines[:end]).strip()


def found_by(line: str, anchor: str) -> bool:
    """Whether a check line copies a sentence holding ``anchor`` (markup,
    whitespace and the line's ellipses aside, as the engine compares)."""
    return squash(anchor) in " ".join(disputed(line))


@dataclass
class CaseScore:
    name: str
    flags: int
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    # Entries marked ``status: partial`` (only partly wrong), found or not.
    partial_found: list[str] = field(default_factory=list)
    partial_missed: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    clean: bool = False


def score_case(name: str, case: dict[str, Any], problems: list[str]) -> CaseScore:
    """Which truth entries of ``case`` the check ``problems`` found."""
    # Entries in the TL;DR or closing are the final check's (``check: final``).
    truth = [e for e in case.get("truth") or [] if e.get("check") != "final"]
    score = CaseScore(name, len(problems), clean=bool(case.get("clean")))
    matched: set[int] = set()
    for entry in truth:
        hits = [i for i, p in enumerate(problems) if found_by(p, entry["anchor"])]
        if entry.get("status") == "partial":
            (score.partial_found if hits else score.partial_missed).append(entry["id"])
        else:
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
    partial = sum(len(s.partial_found) + len(s.partial_missed) for s in scores)
    return {
        "recall": f"{found}/{truth}",
        "partial_found": f"{sum(len(s.partial_found) for s in scores)}/{partial}",
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


def _meta() -> dict[str, str]:
    """What produced a replay: the commit (with "+dirty" for uncommitted
    changes) and a hash of the check prompt."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip()

    prompt = PaperTextCheckPrompt.human_prompt_template
    prompt += PaperTextCheckPrompt.system_prompt_template
    return {
        "commit": git("rev-parse", "--short", "HEAD")
        + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else ""),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:12],
    }


async def _run(
    work: Path, cases: dict[str, dict[str, Any]], path: Path
) -> dict[str, Any]:
    """Replay ``cases`` and write the raw check lines to ``path`` after each
    one (a failed case keeps the earlier ones)."""
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
    meta = _meta() | {
        "scope": "engine",
        "efforts": {k: str(v[1]) for k, v in models.items()},
    }
    out: dict[str, Any] = {"meta": meta, "problems": {}, "models": {}}
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
        body = checked_text(kind, _body(_post_text(work, case["post"])))
        result = await chain.ainvoke(
            {
                "paper_text": paper,
                "number_sources": number_sources(
                    body, paper, max_chars=SOURCE_CAPS[kind]
                )
                or "- none",
                # The engine defangs a review body, not a summary.
                "text": neutralize_prompt_tags(body) if kind == "review" else body,
                "text_name": TEXT_NAMES[kind],
                "language": "Korean",
            }
        )
        # Raw lines: ``score`` applies the engine's filter.
        out["problems"][name] = parse_lines(result.get("problems"))
        out["models"][name] = model_id.value
        out["cost_usd"] = round(tracker.cost_usd, 2)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
        print(f"{name}: {len(out['problems'][name])} lines", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="replay the check (paid Bedrock calls)")
    run.add_argument("work", type=Path)
    run.add_argument("--only", nargs="*", default=None)
    run.add_argument("--out", default="replay.json", help="file name in WORK_DIR")
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
            not in squash(
                _body(_post_text(args.work, case["post"]))
                if entry.get("check") == "final"
                else checked_text(
                    case["kind"], _body(_post_text(args.work, case["post"]))
                )
            )
        ]
        print("\n".join(missing) or "every anchor is in its post")
        return 1 if missing else 0
    if args.command == "run":
        if unknown := set(args.only or ()) - set(cases):
            raise SystemExit(f"unknown cases: {sorted(unknown)}")
        chosen = {k: v for k, v in cases.items() if not args.only or k in args.only}
        path = args.work / args.out
        if path.exists():
            raise SystemExit(f"{path} exists: pass another --out")
        replay = asyncio.run(_run(args.work, chosen, path))
        print(f"cost_usd={replay['cost_usd']} -> {path}")
        args.replay = path
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    scores = [
        score_case(name, cases[name], reported_claims(problems))
        for name, problems in replay["problems"].items()
        if name in cases
    ]
    scope = (replay.get("meta") or {}).get("scope", "unrecorded")
    print(json.dumps({"scope": scope} | report(scores), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
