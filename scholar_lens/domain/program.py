"""What a post's Python code defines and whether it runs as pasted, read
with the standard library's parser (``ast``, ``symtable``), never with
patterns: the program check of the sections a reader runs, and the
definitions the next section's writer is shown (docs/engine.md, 4.1, 4.8)."""

from __future__ import annotations

import ast
import builtins
import re
import symtable
import textwrap
from collections.abc import Sequence

_PYTHON_INFO = ("python", "py", "python3")


def _bound_names(table: symtable.SymbolTable) -> set[str]:
    return {
        sym.get_name()
        for sym in table.get_symbols()
        if sym.is_assigned() or sym.is_imported() or sym.is_namespace()
    }


def _free_globals(table: symtable.SymbolTable) -> set[str]:
    """Names a block's code reads from module scope (directly, or in a
    function or class body) without binding them there."""
    names: set[str] = set()
    for sym in table.get_symbols():
        name = sym.get_name()
        if table.get_type() == "module":
            if sym.is_referenced() and not (
                sym.is_assigned() or sym.is_imported() or sym.is_namespace()
            ):
                names.add(name)
        elif sym.is_global() and sym.is_referenced():
            names.add(name)
    for child in table.get_children():
        names |= _free_globals(child)
    return names


# A fenced block at any indentation (a list item's code too).
_ANY_FENCE = re.compile(
    r"^([ \t]*)(`{3,}|~{3,})[ \t]*([\w+#-]*)[^\n]*\n(.*?)^\1\2[ \t]*$", re.M | re.S
)
_PROMPT = re.compile(r"^(?:>>> |\.\.\. )")
# Module-level definitions of a block that does not parse (best effort).
_DEFINED_LINE = re.compile(
    r"^(?:async\s+)?(?:def|class)\s+(\w+)|^(\w+)\s*(?::[^=]+)?=(?!=)|"
    r"^(?:from\s+\S+\s+)?import\s+(.+)$",
    re.M,
)


def python_blocks(markdown: str) -> list[str]:
    """The draft's Python code blocks (at any indentation), dedented."""
    return [
        textwrap.dedent(m.group(4))
        for m in _ANY_FENCE.finditer(markdown)
        if m.group(3).lower() in _PYTHON_INFO
    ]


def _session_code(code: str) -> str | None:
    """A REPL transcript or notebook cell as the code it runs (prompts and
    ``%``/``!`` lines left out); None for a plain script."""
    lines = code.split("\n")
    if not any(ln.lstrip().startswith((">>>", "%", "!")) for ln in lines):
        return None
    return "\n".join(
        _PROMPT.sub("", ln)
        for ln in lines
        if not ln.lstrip().startswith(("%", "!"))
        and (_PROMPT.match(ln) or not ln.strip())
    )


def _recovered_names(code: str) -> set[str]:
    names: set[str] = set()
    for m in _DEFINED_LINE.finditer(code):
        if m.group(1) or m.group(2):
            names.add(m.group(1) or m.group(2))
        elif m.group(3):
            for part in m.group(3).split(","):
                words = part.split()
                if words:
                    names.add(words[-1].split(".")[0].strip("()"))
    return names


# The first line of a block the writer marks as not meant to run as pasted
# (a signature, a stub, part of a larger file): its ``...`` placeholders
# are not reported. The writer prompt documents it.
EXCERPT_MARKER = "# excerpt"
# Decorators whose function is a stub by design (its body is ``...``).
_STUB_DECORATORS = frozenset({"overload", "abstractmethod"})


def is_excerpt(code: str) -> bool:
    """Whether a block's first non-blank line is the excerpt marker."""
    first = next((ln.strip() for ln in code.split("\n") if ln.strip()), "")
    return first.lower().startswith(EXCERPT_MARKER)


def _last_name(node: ast.expr) -> str:
    """``Protocol`` of ``Protocol``, ``typing.Protocol`` or ``t.Protocol[T]``."""
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def placeholder_lines(tree: ast.Module) -> list[int]:
    """Lines of the ``...`` statements of ``tree`` (Ellipsis as a statement,
    so a body that is only ``...`` too), leaving out ``.pyi``-style stubs: the
    bodies of a ``Protocol`` class and of ``@overload`` or
    ``@abstractmethod`` functions."""
    lines: list[int] = []

    def visit(body: Sequence[ast.stmt], stub: bool) -> None:
        for node in body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and node.value.value is Ellipsis
            ):
                if not stub:
                    lines.append(node.lineno)
                continue
            inner = stub
            if isinstance(node, ast.ClassDef):
                inner = stub or any(_last_name(b) == "Protocol" for b in node.bases)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = stub or any(
                    _last_name(d) in _STUB_DECORATORS for d in node.decorator_list
                )
            for field in ("body", "orelse", "finalbody"):
                visit(getattr(node, field, None) or [], inner)
            for handler in getattr(node, "handlers", None) or []:
                visit(handler.body, inner)
            for case in getattr(node, "cases", None) or []:
                visit(case.body, inner)

    visit(tree.body, False)
    return lines


def python_code_problems(
    sections: Sequence[tuple[str, str, bool]],
) -> list[tuple[str, str]]:
    """(section title, problem) for the guide's Python code read in order,
    as a reader pastes it: a block that does not parse, and a name a block
    uses that no block so far (nor Python itself) defines. ``sections`` are
    (title, markdown, checked): every block's definitions count (a REPL or
    notebook block's too, and a best-effort reading of a block that does not
    parse), problems are reported for ``checked`` sections only (the running
    example's usage, not an internals excerpt). After a star import later
    names are unknowable and are not reported. A checked block with a
    ``...`` placeholder (:func:`placeholder_lines`) is a problem unless its
    first line is :data:`EXCERPT_MARKER`. Parsing and scopes come from the
    standard library (``ast``, ``symtable``)."""
    known = set(dir(builtins)) | {"__name__", "__file__"}
    reliable = True
    problems: list[tuple[str, str]] = []
    for title, markdown, checked in sections:
        for block in python_blocks(markdown):
            session = _session_code(block)
            code = block if session is None else session
            report = checked and session is None
            try:
                tree = ast.parse(code)
                table = symtable.symtable(code, "<guide>", "exec")
            except (SyntaxError, ValueError) as e:
                known |= _recovered_names(code)
                if report:
                    reason = e.msg if isinstance(e, SyntaxError) else str(e)
                    problems.append(
                        (
                            title,
                            f"a python block does not parse ({reason}): "
                            f"{code.strip()[:120]}",
                        )
                    )
                continue
            if any(
                isinstance(node, ast.ImportFrom)
                and any(alias.name == "*" for alias in node.names)
                for node in ast.walk(tree)
            ):
                reliable = False
            if report and not is_excerpt(code) and placeholder_lines(tree):
                problems.append(
                    (
                        title,
                        "the block has placeholders; show runnable code or "
                        "mark it as an excerpt (its first line "
                        f"`{EXCERPT_MARKER}`): {code.strip()[:120]}",
                    )
                )
            bound = _bound_names(table)
            missing = sorted(_free_globals(table) - bound - known)
            if report and reliable and missing:
                problems.append(
                    (
                        title,
                        "a python block uses names no earlier block or import "
                        f"defines: {', '.join(missing)}; define or import them "
                        "where a reader pasting the blocks in order has them",
                    )
                )
            known |= bound
    return problems


# A top-level value shown with its definition is cut at this length.
MAX_DEFINITION_VALUE_CHARS = 80
_TOP_LEVEL_BLOCKS = (ast.If, ast.Try, ast.TryStar, ast.With)


def code_definitions(markdowns: Sequence[str], limit: int = 60) -> list[str]:
    """What the Python code of ``markdowns`` defines, read in order with
    ``ast``: the signature of each function and class (with its decorators;
    a class's methods as ``Class.method``) and each top-level name with its
    value, one line each. A name defined again keeps its last definition, in
    the place of that definition; the last ``limit`` lines are returned. A
    REPL or notebook block counts as the code it runs; a block that does not
    parse defines nothing here (the code check reports it)."""
    found: dict[str, str] = {}
    for markdown in markdowns:
        for block in python_blocks(markdown):
            session = _session_code(block)
            try:
                tree = ast.parse(block if session is None else session)
            except (SyntaxError, ValueError):
                continue
            for key, line in _definitions(tree.body, ""):
                found.pop(key, None)
                found[key] = line
    return list(found.values())[-limit:] if limit > 0 else []


def _definitions(body: Sequence[ast.stmt], owner: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{owner}{node.name}"
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            signature = f"{prefix} {name}({ast.unparse(node.args)}){returns}"
            out.append((name, _decorated(node, signature)))
        elif isinstance(node, ast.ClassDef):
            name = f"{owner}{node.name}"
            bases = [ast.unparse(b) for b in (*node.bases, *node.keywords)]
            signature = f"class {name}" + (f"({', '.join(bases)})" if bases else "")
            out.append((name, _decorated(node, signature)))
            out += _definitions(node.body, f"{name}.")
        elif owner:
            continue
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _shortened(ast.unparse(node.value))
            for target in targets:
                if isinstance(target, ast.Name):
                    annotation = (
                        f": {ast.unparse(node.annotation)}"
                        if isinstance(node, ast.AnnAssign)
                        else ""
                    )
                    out.append((target.id, f"{target.id}{annotation} = {value}"))
        elif isinstance(node, _TOP_LEVEL_BLOCKS):
            # Code under ``if __name__ == "__main__":`` or ``with`` runs too.
            parts = [node.body]
            if isinstance(node, ast.If):
                parts.append(node.orelse)
            elif isinstance(node, (ast.Try, ast.TryStar)):
                parts += [*(h.body for h in node.handlers), node.orelse]
                parts.append(node.finalbody)
            for part in parts:
                out += _definitions(part, owner)
    return out


def _decorated(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, signature: str
) -> str:
    decorators = " ".join(f"@{ast.unparse(d)}" for d in node.decorator_list)
    return f"{decorators} {signature}" if decorators else signature


def _shortened(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_DEFINITION_VALUE_CHARS:
        return text
    return text[: MAX_DEFINITION_VALUE_CHARS - 1].rstrip() + "…"
