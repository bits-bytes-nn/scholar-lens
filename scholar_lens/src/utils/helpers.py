import asyncio
import functools
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PreformattedString

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
# LaTeXML apparatus with no reading value: footnote marks and numbers, code
# listing line numbers (algorithm step numbers are kept), and the base64
# download link attached to every listing.
_DROPPED_LATEXML = (
    ".ltx_note_mark",
    ".ltx_tag_note",
    ".ltx_note_type",
    ".ltx_lstlisting .ltx_tag_listingline",
    ".ltx_listing_data",
)
# Itemize markers LaTeXML renders as a tag in each <li>.
_BULLETS = frozenset({"•", "∙", "◦", "▪", "‣", "–", "-", "∗", "*", "⋅"})
_LISTING_SPACE = re.compile(r"(<span[^>]*\bltx_lst_space\b[^>]*>)( +)(</span>)")
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


# LaTeXML lays out some tables as nested <span>/<div> elements with classes,
# spans encoded as ``ltx_colspan_N``/``ltx_rowspan_N``.
_LATEXML_TABLE_PARTS = {
    "ltx_tabular": "table",
    "ltx_thead": "thead",
    "ltx_tbody": "tbody",
    "ltx_tr": "tr",
    "ltx_td": "td",
}
_LATEXML_SPAN = re.compile(r"^ltx_(colspan|rowspan)_(\d+)$")
# KaTeX renders each formula twice: MathML (kept) and visual HTML (dropped
# only when the MathML copy is there; HTML-only output is the formula).
_KATEX_VISUAL = ".katex-html, [aria-hidden=true]"
# Images that are page furniture, not content: tracking pixels, site logos
# and anything in the page chrome.
_CHROME_TAGS = ("noscript", "header", "nav", "footer")
_LOGO = re.compile(r"(?:^|[-_.])logo(?:$|[-_.])", re.I)


def is_page_furniture(img: Tag) -> bool:
    """A docs-page ``<img>`` that is chrome, a tracking pixel or a site logo."""
    if img.find_parent(_CHROME_TAGS):
        return True
    if {str(img.get("width", "")).strip(), str(img.get("height", "")).strip()} & {
        "0",
        "1",
    }:
        return True
    src = str(img.get("src") or img.get("data-src") or "")
    return bool(_LOGO.search(src.rsplit("/", 1)[-1]))


# Sphinx/MkDocs anchor glyphs after headings ("## torch.compile#").
_HEADER_LINK = "a.headerlink, a.anchor, a.headeranchor-link"


def _latexml_tables_to_html(soup: BeautifulSoup) -> None:
    """Rename LaTeXML ``span``/``div`` tables to real table elements so they
    go through :func:`table_to_markdown` instead of flattening into a line."""
    for element in soup.select(
        ", ".join(f"span.{c}, div.{c}" for c in _LATEXML_TABLE_PARTS)
    ):
        classes = _classes(element)
        part = next(c for c in classes if c in _LATEXML_TABLE_PARTS)
        element.name = _LATEXML_TABLE_PARTS[part]
        for name in classes:
            if match := _LATEXML_SPAN.match(name):
                element[match.group(1)] = match.group(2)


def _inline(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _math_tex(element: Tag) -> str:
    """LaTeX source of a MathML element (alttext, else its TeX annotation)."""
    tex = str(element.get("alttext") or "").strip()
    if not tex:
        annotation = element.find("annotation", attrs={"encoding": "application/x-tex"})
        tex = annotation.get_text().strip() if isinstance(annotation, Tag) else ""
    return tex or _inline(element.get_text(" ")).strip()


def _math_markdown(element: Tag) -> str:
    """LaTeX for a MathML element, delimited the way the blog renders it:
    ``$$…$$`` inline and display alike, display on its own paragraph."""
    tex = _math_tex(element)
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


def _classes(element: Tag) -> list[str]:
    raw: Any = element.get("class") or []
    return raw.split() if isinstance(raw, str) else [str(c) for c in raw]


def _has_class(element: Tag, name: str) -> bool:
    return name in _classes(element)


def _is_equation_table(table: Tag) -> bool:
    return any(c in _EQUATION_TABLE_CLASSES for c in _classes(table))


def _is_equation_cell(td: Tag) -> bool:
    """Not the equation number or the centering padding around a row."""
    return not any(c == "ltx_eqn_eqno" or "_pad" in c for c in _classes(td))


def _equation_table_markdown(table: Tag) -> str:
    """One display equation for a LaTeXML equation table, equation numbers
    dropped. ``align``/``eqnarray`` put each side of a row in its own cell;
    the cells of a row are rejoined (``&`` keeps the alignment) and several
    rows become one ``aligned`` block, so no fragment starts with ``=``."""
    rows: list[list[str]] = []
    for tr in _own_rows(table):
        cells = [
            " ".join(_math_tex(m) for m in td.find_all("math"))
            for td in tr.find_all("td", recursive=False)
            if _is_equation_cell(td)
        ]
        while cells and not cells[-1]:
            cells.pop()
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    if len(rows) == 1:
        tex = " ".join(c for c in rows[0] if c)
    else:
        lines = " \\\\ ".join(" & ".join(cells) for cells in rows)
        tex = f"\\begin{{aligned}} {lines} \\end{{aligned}}"
    return f"\n\n$$ {tex} $$\n\n"


def _nested_table_text(table: Tag, render_cell: Callable[[Tag], str]) -> str:
    """A table inside a table cell, flattened: cells ' / ', rows '; ' (a
    one-column stack, LaTeX's way of breaking a cell's text, joins with spaces)."""
    rows, widest = [], 0
    for tr in _own_rows(table):
        cells = [
            _inline(render_cell(c)).strip()
            for c in tr.find_all(["td", "th"], recursive=False)
        ]
        if any(cells):
            filled = [c for c in cells if c]
            widest = max(widest, len(filled))
            rows.append(" / ".join(filled))
    return (" " if widest == 1 else "; ").join(rows)


def _body_cells(slot_row: list[tuple[str, bool] | None]) -> list[str]:
    """Row texts; a single cell spanning the whole row (a group label) is kept
    once instead of repeated across every column."""
    texts = [slot[0] if slot else "" for slot in slot_row]
    if len(slot_row) > 1 and all(slot and slot[1] for slot in slot_row[1:]):
        return texts[:1] + [""] * (len(texts) - 1)
    return texts


def table_to_markdown(table: Tag, render_cell: Callable[[Tag], str]) -> str:
    """Serialise an HTML table into one complete GFM table.

    colspan/rowspan are expanded into a grid; stacked header rows are merged
    column-wise ("AIME 2024" over "pass@1" -> "AIME 2024 pass@1"); every row has
    the header's width, a value spanning columns is repeated in each (except a
    full-width label row), empty cells become "-", and literal pipes are
    escaped.
    """
    # Each slot holds (text, spanned): ``spanned`` marks the copies a colspan
    # creates.
    grid: list[list[tuple[str, bool] | None]] = []
    first_row_span = 1
    is_th_row: list[bool] = []
    in_thead: list[bool] = []
    # A row whose only own cell spans columns: a section label such as
    # "Closed-Source Base Model", never part of a stacked header.
    section_row: list[bool] = []
    for r, tr in enumerate(_own_rows(table)):
        while len(grid) <= r:
            grid.append([])
        row = grid[r]
        cells = list(tr.find_all(["td", "th"], recursive=False))
        is_th_row.append(bool(cells) and all(c.name == "th" for c in cells))
        in_thead.append(tr.find_parent("thead") is not None)
        section_row.append(len(cells) == 1 and _cell_span(cells[0], "colspan") > 1)
        col = 0
        for cell in cells:
            while col < len(row) and row[col] is not None:
                col += 1
            text = _inline(render_cell(cell)).strip()
            colspan, rowspan = _cell_span(cell, "colspan"), _cell_span(cell, "rowspan")
            if r == 0:
                first_row_span = max(first_row_span, rowspan)
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
    section_row = [section_row[i] if i < len(section_row) else False for i in keep]
    rows = [_body_cells(slot_row) for slot_row in slots]

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
        # A first-row cell spanning several rows ("Model" over a two-row
        # header) marks those rows as header even when one looks numeric.
        header_count = max(header_count, min(first_row_span, _MAX_HEADER_ROWS))
    header_count = max(1, min(header_count, len(rows) - 1)) if len(rows) > 1 else 1
    # A rowspan in the header must not pull a section label row into it (a
    # spanning sub-header inside <thead> is still header).
    header_count = next(
        (i for i in range(1, header_count) if section_row[i] and not in_thead[i]),
        header_count,
    )

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


def _is_source_indent(text: str) -> bool:
    return text.isspace() and "\n" in text


def _listing_code(listing: Tag) -> str:
    """A LaTeXML code listing (``lstlisting``), one ``ltx_listingline`` per
    line, verbatim."""
    lines = listing.select(".ltx_listingline")
    if not lines:
        return listing.get_text()

    def line_text(line: Tag) -> str:
        # Whitespace runs with a newline between the line's spans are HTML
        # source indentation, not code.
        parts = (
            child.get_text() if isinstance(child, Tag) else str(child)
            for child in line.children
            if isinstance(child, Tag)
            or (isinstance(child, NavigableString) and not _is_source_indent(child))
        )
        return "".join(parts).replace("\xa0", " ").rstrip()

    return "\n".join(line_text(line) for line in lines)


def preserve_listing_spaces(html: str) -> str:
    """Listing indentation as NBSPs, applied to raw HTML before any
    BeautifulSoup parse: the parser collapses whitespace-only strings, which
    would flatten the spaces LaTeXML puts in ``ltx_lst_space`` spans.
    Idempotent."""
    return _LISTING_SPACE.sub(
        lambda m: m.group(1) + "\xa0" * len(m.group(2)) + m.group(3), html
    )


def _code_span(code: str) -> str:
    """Inline code delimited by a backtick run longer than any inside it,
    padded when the code starts or ends with a backtick (CommonMark), so it
    can never read as a fence opener."""
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    ticks = "`" * (longest + 1)
    pad = " " if code.startswith("`") or code.endswith("`") else ""
    return f"{ticks}{pad}{code}{pad}{ticks}"


_FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$", re.MULTILINE)
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]")


def _fence_toggle(line: str, open_fence: str | None) -> str | None:
    """The fence still open after ``line``, given the one open before it. A
    fence closes on a bare run of its own character at least as long
    (CommonMark)."""
    match = _FENCE_OPEN.match(line)
    if open_fence is None:
        if match and match.group(1)[0] == "`" and "`" in match.group(2):
            return None  # a code span such as ```` ``` ````, not a fence
        return match.group(1) if match else None
    closes = (
        match is not None
        and match.group(1)[0] == open_fence[0]
        and len(match.group(1)) >= len(open_fence)
        and not match.group(2).strip()
    )
    return None if closes else open_fence


def _code_fence(code: str) -> str:
    """``code`` fenced with a marker none of its own lines can close: ```` ``` ````,
    else ``~~~``, else a backtick run longer than any inside."""
    runs = [m.group(1) for m in _FENCE_OPEN.finditer(code)]
    if not any(run[0] == "`" for run in runs):
        marker = "```"
    elif not any(run[0] == "~" for run in runs):
        marker = "~~~"
    else:
        marker = "`" * (max(len(run) for run in runs if run[0] == "`") + 1)
    return f"\n\n{marker}\n{code}\n{marker}\n\n"


def _list_line(line: str, in_list: bool) -> tuple[str, bool]:
    """A prose line outside fences, and whether a list item is still open.

    Item markers and the indented continuation lines right after them keep
    their indentation (paragraphs, tables or headings inside an item); any
    other line is stripped, so stray source indentation never reads as an
    indented code block.
    """
    if _LIST_MARKER.match(line) or (in_list and line[:1] in (" ", "\t")):
        indent = line[: len(line) - len(line.lstrip(" \t"))]
        return indent + re.sub(r"[ \t]+", " ", line).strip(), True
    return re.sub(r"[ \t]+", " ", line).strip(), False


def _tighten_outside_fences(text: str) -> str:
    """Drop blank lines (and the indentation around them) outside code fences,
    so a list item's paragraphs stay on its lines; fenced code and the
    indentation of nested list items are kept."""
    lines: list[str] = []
    fence: str | None = None
    in_list = False
    for line in text.split("\n"):
        was_open = fence is not None
        fence = _fence_toggle(line, fence)
        if was_open and fence is not None:
            lines.append(line)  # code
        elif was_open or fence is not None:
            lines.append(line.strip())  # a fence line
        elif line.strip():
            kept, in_list = _list_line(line, in_list)
            lines.append(kept)
    return "\n".join(lines)


def _list_item(item_text: str, marker: str) -> str:
    """One Markdown list item: continuation lines (paragraphs, nested items)
    indented under the marker; fenced code stays at the margin."""
    lines = _tighten_outside_fences(item_text.strip()).split("\n")
    if lines and _FENCE_OPEN.match(lines[0]):
        lines.insert(0, "")  # a fence must open on its own line
    pad = " " * len(marker)
    out: list[str] = [f"{marker}{lines[0]}" if lines else marker.rstrip()]
    fence: str | None = None
    for line in lines[1:]:
        was_open = fence is not None
        fence = _fence_toggle(line, fence)
        in_code = was_open or fence is not None
        out.append(line if in_code or not line else pad + line)
    return "\n" + "\n".join(out) + "\n"


def _list_start(element: Tag) -> int:
    try:
        return int(str(element.get("start", "1")).strip())
    except ValueError:
        return 1


def extract_text_from_html(
    html_content: str,
    *,
    base_url: str | None = None,
    skip_page_furniture: bool = False,
) -> str:
    """Convert paper/doc HTML into compact Markdown-like text for the models.

    Keeps the structure the writers need: headings (``##``), paragraphs, list
    items, fenced ``<pre>`` and LaTeXML listing code, ``$$``-delimited LaTeX
    from MathML, complete GFM tables, footnotes in place, and
    ``[Image: alt="…", src="…"]`` figure markers (``src`` resolved against
    ``base_url`` when given). Page chrome (nav/header/footer/aside, scripts) is
    dropped; with ``skip_page_furniture`` (docs pages) so are logo and
    tracking-pixel images, matching the page's image list.
    """
    if not html_content:
        return ""

    html_content = preserve_listing_spaces(html_content)
    soup = BeautifulSoup(html_content, "html.parser")
    for tag_name in _DROPPED_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag in soup.select(", ".join((*_DROPPED_LATEXML, _HEADER_LINK))):
        tag.decompose()
    for katex in soup.select(".katex"):
        if katex.select_one(".katex-mathml"):
            for tag in katex.select(_KATEX_VISUAL):
                tag.decompose()
    _latexml_tables_to_html(soup)
    for tag_name in _PAGE_LEVEL_TAGS:
        for tag in soup.find_all(tag_name):
            if tag.find_parent(["article", "main"]) is None:
                tag.decompose()

    def resolve(src: str) -> str:
        return urljoin(base_url, src) if base_url and src else src

    def children(element: Tag) -> str:
        return "".join(parse(child) for child in element.children)

    def list_item(element: Tag, marker: str) -> str:
        # LaTeXML items carry their marker as a tag: a bullet glyph is
        # dropped, an enumerate label ("(a)", "1.") is kept after "- ".
        label = element.find(class_="ltx_tag_item")
        if isinstance(label, Tag):
            text = label.get_text(strip=True)
            marker = "- " if text in _BULLETS else f"- {text} "
            label.decompose()
        return _list_item(children(element), marker)

    def parse(element: Any) -> str:
        if isinstance(element, PreformattedString):
            return ""  # comments, doctype, CDATA, processing instructions
        if isinstance(element, NavigableString):
            return _inline(str(element))
        if not isinstance(element, Tag):
            return ""
        name = element.name
        if name == "img":
            if skip_page_furniture and is_page_furniture(element):
                return ""
            src = str(element.get("src") or element.get("data-src") or "").strip()
            if src.lower().startswith("data:"):
                return ""  # inline bitmap: nothing a writer could reference
            alt = _inline(str(element.get("alt", ""))).strip()
            alt = alt.replace('"', "'")
            return f'[Image: alt="{alt}", src="{resolve(src)}"]'
        if (
            name == "object"
            and element.get("data")
            and "svg" in str(element.get("type", ""))
        ):
            # arXiv renders vector figures (often the main diagram) this way.
            data = str(element.get("data")).strip()
            if data.lower().startswith("data:"):
                return ""  # inline SVG payload, as for <img>
            return f'[Image: alt="", src="{resolve(data)}"]'
        if name == "math":
            return _math_markdown(element)
        if name == "table":
            if _is_equation_table(element):
                return _equation_table_markdown(element)
            if element.find_parent("table") is not None:
                return _nested_table_text(element, children)
            return table_to_markdown(element, children)
        # (The <figure> around a listing carries ``ltx_lstlisting`` too.)
        is_listing = name in ("div", "span") and _has_class(element, "ltx_listing")
        if name == "pre" or (is_listing and _has_class(element, "ltx_lstlisting")):
            code = (
                element.get_text() if name == "pre" else _listing_code(element)
            ).strip("\n")
            return _code_fence(code) if code.strip() else ""
        if is_listing:
            # An algorithm (``algorithmic``): numbered steps whose math must
            # stay LaTeX, so it is prose line by line, not a code fence.
            steps = (
                _inline(children(line)).strip()
                for line in element.select(".ltx_listingline")
            )
            text = "\n".join(step for step in steps if step)
            return f"\n\n{text}\n\n" if text else children(element)
        if name == "code":
            code = _inline(element.get_text()).strip()
            return _code_span(code) if code else ""
        if name == "br":
            return "\n"
        if name in _HEADING_TAGS:
            text = _inline(children(element)).strip()
            return f"\n\n{'#' * max(2, _HEADING_TAGS[name])} {text}\n\n" if text else ""
        if name in ("ul", "ol"):
            number = _list_start(element)
            parts: list[str] = []
            for child in element.children:
                if isinstance(child, Tag) and child.name == "li":
                    marker = f"{number}. " if name == "ol" else "- "
                    parts.append(list_item(child, marker))
                    number += 1
                else:
                    parts.append(parse(child))
            return f"\n\n{''.join(parts)}\n\n"
        if name == "li":  # outside a list
            return list_item(element, "- ")
        if _has_class(element, "ltx_note"):
            content = element.find(class_="ltx_note_content")
            note = _inline(children(content) if isinstance(content, Tag) else "")
            return f" (footnote: {note.strip()})" if note.strip() else ""
        if name == "a":
            text = children(element)
            href = str(element.get("href", ""))
            if (
                not href
                or href.strip().lower().startswith(("#", "javascript:", "data:"))
                or href == text.strip()
            ):
                return text
            return f"{text} ({resolve(href.strip())})"
        if name in _BLOCK_TAGS:
            return f"\n\n{children(element)}\n\n"
        return children(element)

    text = parse(soup)
    # LaTeXML extraction artifacts (never touch real LaTeX such as "\times").
    for artifact in ("\\AND", "footnotemark:"):
        text = text.replace(artifact, "")
    lines: list[str] = []
    fence: str | None = None
    in_list = False
    for line in text.split("\n"):
        was_open = fence is not None
        fence = _fence_toggle(line, fence)
        if was_open or fence is not None:
            lines.append(line)  # code (and its fence lines)
        elif line.strip():
            kept, in_list = _list_line(line, in_list)
            lines.append(kept)
        else:
            lines.append("")  # a whitespace-only line is a paragraph break
            in_list = False
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
