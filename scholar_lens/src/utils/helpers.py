import argparse
import asyncio
import functools
import re
import time
from collections.abc import Callable
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from ..logger import logger


def parse_quality_score(raw: Any) -> int:
    """Parse a 0-100 evaluator quality score robustly.

    Shared by the review and tech-guide evaluate-and-revise loops. The prompt
    asks for a bare integer, but a model may still pad it: "85", "85/100",
    "Score: 85", "85 (excellent)", "9/10", or "N/A". Precedence:

    1. a clean integer;
    2. otherwise a leading ``<n>/<total>`` fraction -> the NUMERATOR ("85/100" ->
       85, not 100);
    3. otherwise the FIRST integer anywhere in the string (so "Score: 85" -> 85,
       not 0 — an anchored match here would mis-score and force a wasted revision).

    The result is clamped to 0-100; a non-numeric value falls back to 0, which
    forces another attempt rather than crashing.
    """
    text = str(raw if raw is not None else "").strip()

    def clamp(n: int) -> int:
        return max(0, min(100, n))

    try:
        return clamp(int(text))
    except ValueError:
        pass
    fraction = re.search(r"(\d+)\s*/\s*\d+", text)
    if fraction:
        return clamp(int(fraction.group(1)))
    match = re.search(r"\d+", text)
    return clamp(int(match.group())) if match else 0


def arg_as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.lower().strip()
        if value in ("yes", "true", "t", "y", "1"):
            return True
        if value in ("no", "false", "f", "n", "0"):
            return False

    raise argparse.ArgumentTypeError("Boolean value expected")


_AFFIRMATIVE = frozenset({"y", "yes", "yeah", "yep", "true", "t", "1"})
_PLACEHOLDER = frozenset(
    {
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "unnamed",
        "anonymous",
        "not available",
        "empty",
    }
)


def is_affirmative(value: str | None) -> bool:
    """Whether an LLM yes/no field means "yes".

    Tolerant of the common ways a model says yes ("y", "yes", "true", …) instead
    of brittle equality with a single character. Anything else (incl. None) is no.
    """
    return value is not None and value.strip().lower() in _AFFIRMATIVE


def is_placeholder(value: str | None) -> bool:
    """Whether a value is a missing/placeholder marker (N/A, Unknown, None…)."""
    return not value or value.strip().lower() in _PLACEHOLDER


def escape_yaml_double_quoted(value: str) -> str:
    """Escape a string for use inside a double-quoted YAML scalar (``"..."``).

    arXiv/LLM-extracted titles routinely carry raw LaTeX (``\\mathcal``,
    ``\\nabla``, accented ``\\"o``). A double-quoted YAML scalar treats ``\\`` as
    an escape introducer, so an unescaped backslash either aborts the Jekyll/
    kramdown build (``\\mathcal`` -> unknown-escape ScannerError) or silently
    corrupts the value (``\\nabla`` -> literal newline). Escape backslashes FIRST,
    then double quotes, then the control chars YAML would otherwise mangle.
    """
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    value = value.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    return value


# Page chrome that carries no paper/doc content (running heads, nav menus).
_DROPPED_TAGS = (
    "head",
    "meta",
    "script",
    "style",
    "title",
    "noscript",
    "nav",
    "footer",
    "button",
    "form",
)
# Dropped only at page level: inside <article>/<main> a <header> holds the
# document title and an <aside> holds callouts/notes worth keeping.
_PAGE_LEVEL_TAGS = ("header", "aside")
# LaTeXML renders numbered display equations as tables; they are math, not data.
_EQUATION_TABLE_CLASSES = ("ltx_equation", "ltx_eqn_table", "ltx_equationgroup")
_MATH_SPAN = re.compile(r"\$\$.+?\$\$", re.DOTALL)
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "figure",
        "figcaption",
        "blockquote",
        "ul",
        "ol",
        "dl",
        "dt",
        "dd",
        "body",
        "html",
    }
)
_HEADING_TAGS = {f"h{n}": n for n in range(1, 7)}
_NUMERIC_CELL = re.compile(
    r"^[-+−]?(\d[\d,]*(\.\d+)?|\.\d+)\s*(%|±\s*[\d.]+|[kKMB×x])?$"
)
# Header rows detected by content (no <thead>/<th>) are capped so a text-only
# table can't swallow its body into the header.
_MAX_HEADER_ROWS = 3


def _inline(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _math_markdown(element: Tag) -> str:
    """LaTeX for a MathML element, delimited the way the blog renders it:
    ``$$…$$`` inline and display alike, display on its own paragraph."""
    tex = str(element.get("alttext") or "").strip()
    if not tex:
        annotation = element.find("annotation", attrs={"encoding": "application/x-tex"})
        tex = annotation.get_text().strip() if isinstance(annotation, Tag) else ""
    if not tex:
        tex = _inline(element.get_text(" ")).strip()
    if not tex:
        return ""
    if element.get("display") == "block":
        return f"\n\n$$ {tex} $$\n\n"
    return f"$${tex}$$"


def _cell_span(cell: Tag, attr: str) -> int:
    try:
        return max(1, min(int(str(cell.get(attr) or 1)), 50))
    except ValueError:
        return 1


def _own_rows(table: Tag) -> list[Tag]:
    """``<tr>`` rows of ``table`` itself, excluding rows of nested tables."""
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


def _escape_cell_pipes(cell: str) -> str:
    """Make a table cell pipe-safe without changing its meaning: prose pipes are
    escaped (``\\|``); inside ``$$…$$`` a norm ``\\|`` becomes ``\\Vert`` and a
    bare ``|`` becomes ``\\vert`` (escaping them would turn ``|x|`` into a norm
    or ``\\|W\\|`` into a line break)."""
    out, pos = [], 0
    for m in _MATH_SPAN.finditer(cell):
        out.append(cell[pos : m.start()].replace("|", "\\|"))
        math = m.group(0).replace("\\|", "\\Vert ")
        out.append(re.sub(r"(?<!\\)\|", r"\\vert ", math))
        pos = m.end()
    out.append(cell[pos:].replace("|", "\\|"))
    return "".join(out)


def _is_equation_table(table: Tag) -> bool:
    raw: Any = table.get("class") or []
    classes: list[str] = raw.split() if isinstance(raw, str) else [str(c) for c in raw]
    return any(c in _EQUATION_TABLE_CLASSES for c in classes)


def _equation_table_markdown(table: Tag) -> str:
    """Display math for a LaTeXML equation table (equation numbers dropped)."""
    blocks = []
    for math in table.find_all("math"):
        if math.find_parent(class_="ltx_eqn_eqno") is not None:
            continue
        tex = _math_markdown(math).strip().strip("$").strip()
        if tex:
            blocks.append(f"$$ {tex} $$")
    return "\n\n" + "\n\n".join(blocks) + "\n\n" if blocks else ""


def _nested_table_text(table: Tag, render_cell: Callable[[Tag], str]) -> str:
    """A table inside a table cell, flattened: cells ' / ', rows '; '."""
    rows = []
    for tr in _own_rows(table):
        cells = [
            _inline(render_cell(c)).strip()
            for c in tr.find_all(["td", "th"], recursive=False)
        ]
        if any(cells):
            rows.append(" / ".join(c for c in cells if c))
    return "; ".join(rows)


def table_to_markdown(table: Tag, render_cell: Callable[[Tag], str]) -> str:
    """Serialise an HTML table into one complete GFM table.

    colspan/rowspan are expanded into a grid; stacked header rows are merged
    column-wise ("AIME 2024" over "pass@1" -> "AIME 2024 pass@1"); every row has
    the header's width, empty cells become "-", and literal pipes are escaped.
    """
    # Each slot holds (text, spanned): ``spanned`` marks the copies a colspan
    # creates, which repeat in header rows and stay empty in body rows.
    grid: list[list[tuple[str, bool] | None]] = []
    is_th_row: list[bool] = []
    in_thead: list[bool] = []
    for r, tr in enumerate(_own_rows(table)):
        while len(grid) <= r:
            grid.append([])
        row = grid[r]
        cells = list(tr.find_all(["td", "th"], recursive=False))
        is_th_row.append(bool(cells) and all(c.name == "th" for c in cells))
        in_thead.append(tr.find_parent("thead") is not None)
        col = 0
        for cell in cells:
            while col < len(row) and row[col] is not None:
                col += 1
            text = _inline(render_cell(cell)).strip()
            colspan, rowspan = _cell_span(cell, "colspan"), _cell_span(cell, "rowspan")
            for dr in range(rowspan):
                while len(grid) <= r + dr:
                    grid.append([])
                target = grid[r + dr]
                while len(target) < col + colspan:
                    target.append(None)
                for dc in range(colspan):
                    target[col + dc] = (text, dc > 0)
            col += colspan
    keep = [i for i, row in enumerate(grid) if any(slot and slot[0] for slot in row)]
    if not keep:
        return ""
    width = max(len(grid[i]) for i in keep)
    slots = [grid[i] + [None] * (width - len(grid[i])) for i in keep]
    is_th_row = [is_th_row[i] if i < len(is_th_row) else False for i in keep]
    in_thead = [in_thead[i] if i < len(in_thead) else False for i in keep]
    rows = [
        [slot[0] if slot and not slot[1] else "" for slot in slot_row]
        for slot_row in slots
    ]

    header_count = sum(1 for flag in in_thead[: len(rows)] if flag)
    if not header_count:
        header_count = next(
            (i for i, flag in enumerate(is_th_row[: len(rows)]) if not flag), 0
        )
    if not header_count:
        # Leading label rows are header rows only when a value row follows
        # within the cap; a text-only table keeps its first row as the header.
        header_count = 1
        for i, candidate in enumerate(rows[: _MAX_HEADER_ROWS + 1]):
            if any(_NUMERIC_CELL.match(c) for c in candidate):
                header_count = max(1, i)
                break
    header_count = max(1, min(header_count, len(rows) - 1)) if len(rows) > 1 else 1

    header: list[str] = []
    for c in range(width):
        parts: list[str] = []
        for slot_row in slots[:header_count]:
            slot = slot_row[c]
            label = slot[0] if slot else ""
            if label and (not parts or parts[-1] != label):
                parts.append(label)
        header.append(" ".join(parts))
    body = rows[header_count:]

    def fmt(cells: list[str], *, is_header: bool) -> str:
        out = []
        for i, cell in enumerate(cells):
            if not cell and not (is_header and i == 0):
                cell = "-"
            out.append(_escape_cell_pipes(cell))
        return "| " + " | ".join(out) + " |"

    lines = [fmt(header, is_header=True), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(fmt(row, is_header=False) for row in body)
    caption = table.find("caption")
    caption_text = (
        _inline(render_cell(caption)).strip() if isinstance(caption, Tag) else ""
    )
    prefix = f"{caption_text}\n\n" if caption_text else ""
    return "\n\n" + prefix + "\n".join(lines) + "\n\n"


def extract_text_from_html(html_content: str) -> str:
    """Convert paper/doc HTML into compact Markdown-like text for the models.

    Keeps the structure the writers need: headings (``##``), paragraphs, list
    items, fenced ``<pre>`` code, ``$$``-delimited LaTeX from MathML, complete
    GFM tables, and ``[Image: alt=…, src=…]`` figure markers. Page chrome
    (nav/header/footer/aside, scripts) is dropped.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")
    for tag_name in _DROPPED_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag_name in _PAGE_LEVEL_TAGS:
        for tag in soup.find_all(tag_name):
            if tag.find_parent(["article", "main"]) is None:
                tag.decompose()

    def children(element: Tag) -> str:
        return "".join(parse(child) for child in element.children)

    def parse(element: Any) -> str:
        if isinstance(element, NavigableString):
            return _inline(str(element))
        if not isinstance(element, Tag):
            return ""
        name = element.name
        if name == "img":
            alt = _inline(str(element.get("alt", ""))).strip()
            return f"[Image: alt={alt}, src={element.get('src', '')}]"
        if (
            name == "object"
            and element.get("data")
            and "svg" in str(element.get("type", ""))
        ):
            # arXiv renders vector figures (often the main diagram) this way.
            return f"[Image: alt=, src={element.get('data')}]"
        if name == "math":
            return _math_markdown(element)
        if name == "table":
            if _is_equation_table(element):
                return _equation_table_markdown(element)
            if element.find_parent("table") is not None:
                return _nested_table_text(element, children)
            return table_to_markdown(element, children)
        if name == "pre":
            code = element.get_text().strip("\n")
            return f"\n\n```\n{code}\n```\n\n" if code.strip() else ""
        if name == "code":
            code = _inline(element.get_text()).strip()
            return f"`{code}`" if code else ""
        if name == "br":
            return "\n"
        if name in _HEADING_TAGS:
            text = _inline(children(element)).strip()
            return f"\n\n{'#' * max(2, _HEADING_TAGS[name])} {text}\n\n" if text else ""
        if name == "li":
            return f"\n- {children(element).strip()}\n"
        if name == "a":
            text = children(element)
            href = str(element.get("href", ""))
            if (
                not href
                or href.startswith(("#", "javascript:"))
                or href == text.strip()
            ):
                return text
            return f"{text} ({href})"
        if name in _BLOCK_TAGS:
            return f"\n\n{children(element)}\n\n"
        return children(element)

    text = parse(soup)
    # LaTeXML extraction artifacts (never touch real LaTeX such as "\times").
    for artifact in ("\\AND", "footnotemark:"):
        text = text.replace(artifact, "")
    lines: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_fence = not in_fence
            lines.append(line)
        elif in_fence:
            lines.append(line)
        else:
            lines.append(re.sub(r"[ \t]+", " ", line).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def measure_execution_time(func: Callable) -> Callable:
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(
            "'%s' execution time: %.2fs (%.2fmin)",
            func.__name__,
            execution_time,
            execution_time / 60,
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
