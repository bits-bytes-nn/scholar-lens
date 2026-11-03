"""Deterministic quality metrics for a generated post.

Measures what the redesign's acceptance criteria compare (docs/redesign.md,
section 7) so the same harness scores the current generator and the new one:
blog validation, numbers repeated across sections, meta sentences, visuals,
style frequencies and structure. Run on finished Markdown files:

    poetry run python -m eval.metrics KIND FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from scholar_lens.src.front_matter import split_front_matter
from scholar_lens.src.markdown_lint import undefined_macros
from scholar_lens.src.post_conventions import load_local_conventions
from scholar_lens.src.post_validator import ArtifactKind, validate_post
from scholar_lens.src.section_loop import prose_chars

_FENCE = re.compile(r"^(`{3,}|~{3,}).*?^\1[ \t]*$", re.M | re.S)
_SECTION = re.compile(r"^## (.+)$", re.M)
# Numbers that carry a fact: decimals, percentages, or 3+ digits that are not
# years (19xx/20xx) and not inside a link target.
_NUMBER = re.compile(r"(?<![\w./])(\d+(?:[.,]\d+)?)(\s?%p?|배|억|만)?(?![\w/])")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_YEAR = re.compile(r"^(19|20)\d\d$")
# Sentences about how the post was written (closed list from past defects).
_META = re.compile(
    r"초록(?:에서|만|과 공개|에 기대|에 따르면|수준)|공개 요약|카드(?:가|에|만)|입력 자료|"
    r"제공된 (?:자료|문서|출처)|자료에 (?:기대|따르면)|이 글(?:에서는|은) .{0,20}(?:다룹니다|살펴봅니다|정리합니다)|"
    r"본 글|이 절에서는|정량적(?:으로)? (?:수치를 )?제시하지"
)
_RESTATE = re.compile(r"이러한|이는|즉,|다시 말해|결국")
_INTENSIFIER = re.compile(r"매우|상당히|굉장히|크게")
_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
_TABLE_DELIM = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.M)
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
_BARE_BAR = re.compile(r"(?<![\\|])\|(?!\|)")
# Commands seen in generated math that MathJax does not define.
_MACRO = re.compile(r"\\([A-Za-z]+)")
_MIRRORED = re.compile(r"^(초록|요약|서론|Abstract|Introduction)$")


def _sections(body: str) -> dict[str, str]:
    text = _FENCE.sub("", body)
    parts = _SECTION.split(text)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _fact_numbers(text: str) -> set[str]:
    text = _LINK_TARGET.sub("]", text)
    found = set()
    for value, unit in _NUMBER.findall(text):
        unit = (unit or "").strip()
        if _YEAR.match(value) and not unit:
            continue
        if not unit and "." not in value and len(value) < 3:
            continue
        found.add(value.replace(",", "") + unit)
    return found


def measure(markdown: str, kind: ArtifactKind) -> dict[str, Any]:
    front, body, _ = split_front_matter(markdown)
    violations = validate_post(markdown, load_local_conventions(), kind)
    sections = _sections(body)
    owners: dict[str, set[str]] = defaultdict(set)
    for title, text in sections.items():
        if title.lower() in {"references", "참고 문헌", "참고 자료", "sources", "출처"}:
            continue
        for n in _fact_numbers(text):
            owners[n].add(title)
    distinct = len(owners)
    repeated = sorted(n for n, s in owners.items() if len(s) > 1)
    prose = prose_chars(body)
    k = max(prose, 1) / 1000
    images = _IMAGE.findall(body)
    math = _DISPLAY_MATH.findall(body)
    macros = {m for m in undefined_macros() if any(f"\\{m}" in x for x in math)}
    return {
        "prose_chars": prose,
        "errors": sum(v.level == "error" for v in violations),
        "warnings": sum(v.level != "error" for v in violations),
        "error_rules": sorted({v.rule for v in violations if v.level == "error"}),
        "sections": len(sections),
        "mirrored_sections": sorted(t for t in sections if _MIRRORED.match(t)),
        "distinct_numbers": distinct,
        "repeated_numbers": len(repeated),
        "repeated_share": round(len(repeated) / distinct, 3) if distinct else 0.0,
        "meta_sentences": [m.group(0) for m in _META.finditer(_FENCE.sub("", body))],
        "visuals_per_10k": round(
            (
                len(images)
                + len(_TABLE_DELIM.findall(body))
                + len(math)
                + len(_FENCE.findall(body))
            )
            / max(prose, 1)
            * 10000,
            2,
        ),
        "images": len(images),
        "non_http_images": sum(not i.startswith(("http", "/")) for i in images),
        "tables": len(_TABLE_DELIM.findall(body)),
        "display_math": len(math),
        "code_blocks": len(_FENCE.findall(body)),
        "undefined_macros": sorted(macros),
        "bare_bars_in_math": sum(len(_BARE_BAR.findall(x)) for x in math),
        "math_commands": sorted({c for x in math for c in _MACRO.findall(x)}),
        "restate_per_1k": round(len(_RESTATE.findall(body)) / k, 2),
        "intensifier_per_1k": round(len(_INTENSIFIER.findall(body)) / k, 2),
        "description_chars": len((front or {}).get("description") or ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["review", "summarize", "guide", "survey"])
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    out = {
        str(f): measure(f.read_text(encoding="utf-8"), args.kind) for f in args.files
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
