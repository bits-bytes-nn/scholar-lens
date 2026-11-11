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
from dataclasses import dataclass

from scholar_lens.domain.verifiers import Problem

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


_TIMING_VALUED = frozenset({"-n", "-r", "-p", "-l", "-s", "-T", "-D"})


def _timed_statement(line: str) -> str:
    """The statement of ``%time``/``%timeit``/``%prun`` without the
    magic's options (``%timeit -n 10 -r 3 f()`` -> ``f()``)."""
    rest = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
    while rest.startswith("-"):
        option, _, rest = rest.partition(" ")
        rest = rest.lstrip()
        if option in _TIMING_VALUED and rest:
            rest = rest.split(None, 1)[1] if len(rest.split(None, 1)) > 1 else ""
    return rest


# Cell magics whose body is Python.
_PYTHON_CELL_MAGICS = frozenset({"time", "timeit", "capture", "prun"})


def _cell_code(lines: list[str]) -> str:
    """A notebook cell as the Python it runs: a ``%%writefile`` cell writes
    its body to the file (a later import reads it), a cell magic whose body
    is not Python runs nothing, and line magics and shell lines are left
    out."""
    body = [ln for ln in lines if ln.strip()]
    first = body[0].strip() if body else ""
    if first.startswith("%%"):
        name, _, arg = first[2:].partition(" ")
        rest = "\n".join(lines[lines.index(body[0]) + 1 :])
        if name in ("writefile", "file") and (
            path := [a for a in arg.split() if not a.startswith("-")]
        ):
            append = "-a" in arg.split() or "--append" in arg.split()
            return (
                f"with open({path[-1]!r}, {'a' if append else 'w'!r}) as _cell:\n"
                f"    _cell.write({(rest + chr(10))!r})"
            )
        if name not in _PYTHON_CELL_MAGICS:
            return ""
        lines = rest.split("\n")
    out = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("%run ") and (args := stripped[5:].split()):
            # Runs the file in the session's namespace, as IPython does.
            indent = ln[: len(ln) - len(stripped)]
            out.append(
                f"{indent}exec(compile(open({args[0]!r}).read(), {args[0]!r}, 'exec'))"
            )
        elif stripped.startswith(("%time ", "%prun ", "%timeit ")) and (
            statement := _timed_statement(stripped)
        ):
            # The statement a timing magic runs (and binds names with).
            indent = ln[: len(ln) - len(stripped)]
            out.append(f"{indent}{statement}")
        elif script := re.match(r"!\s*python3?\s+([\w./-]+\.py)\s*$", stripped):
            # A script a cell wrote, run as a reader's shell runs it.
            indent = ln[: len(ln) - len(stripped)]
            out.append(
                f"{indent}__import__('runpy').run_path({script.group(1)!r}, "
                "run_name='__main__')"
            )
        elif not stripped.startswith(("%", "!")):
            out.append(ln)
        elif ln != stripped:
            # An indented magic or shell line (an except body): a pass
            # keeps the block's structure.
            out.append(f"{ln[: len(ln) - len(stripped)]}pass")
    return "\n".join(out)


def _session_code(code: str) -> str | None:
    """A REPL transcript or notebook cell as the code it runs (prompts and
    ``%``/``!`` lines left out); None for a plain script."""
    lines = code.split("\n")
    if not any(ln.lstrip().startswith((">>>", "%", "!")) for ln in lines):
        return None
    try:
        ast.parse(code)
        return None  # a script whose continuation line starts with "%" or "!"
    except (SyntaxError, ValueError):
        pass
    if not any(ln.lstrip().startswith(">>>") for ln in lines):
        return _cell_code(lines)
    magic = [ln for ln in lines if not ln.lstrip().startswith(("%", "!"))]
    # A REPL transcript: the prompted lines (the rest is output).
    return "\n".join(
        _PROMPT.sub("", ln) for ln in magic if _PROMPT.match(ln) or not ln.strip()
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


@dataclass(frozen=True)
class _CodeFinding:
    """One problem of the code check: its section, its message, the part of
    the message that quotes the block, and the names a block uses that
    nothing defines (a names problem)."""

    title: str
    message: str
    passage: str = ""
    names: tuple[str, ...] = ()


def python_code_problems(
    sections: Sequence[tuple[str, str, bool]],
) -> list[tuple[str, str]]:
    """(section title, problem) of :func:`_code_findings`."""
    return [(f.title, f.message) for f in _code_findings(sections)]


# The problem kind of the whole post's program check.
PROGRAM_PROBLEM = "program"


def program_problems(sections: Sequence[tuple[str, str, bool]]) -> list[Problem]:
    """:func:`python_code_problems` as problems an edit transaction compares:
    a block that does not parse or has placeholders quotes the block as its
    passage (a rewritten block with the same fault is the same problem),
    and each name nothing defines is a problem of its own (a fix that
    leaves one more name undefined adds one).

    Args:
        sections: (title, markdown, checked), in reading order.

    Returns:
        The problems, in order.
    """
    out: list[Problem] = []
    for f in _code_findings(sections):
        if f.names:
            out += [
                Problem(
                    PROGRAM_PROBLEM,
                    f"{f.title}: a python block uses a name no earlier block or "
                    f"import defines: {name}",
                )
                for name in f.names
            ]
        else:
            out.append(Problem(PROGRAM_PROBLEM, f"{f.title}: {f.message}", f.passage))
    return out


def _call_signature(node: ast.stmt) -> tuple[str, ast.arguments, bool] | None:
    """A top-level definition's name, parameters and whether a call passes
    ``self`` implicitly: an undecorated function, or an undecorated class
    with its own ``__init__`` (the last one that is not an ``@overload``
    stub). A decorator may change what a call takes (``@click.command``),
    so a decorated definition is not checked."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None if node.decorator_list else (node.name, node.args, False)
    # A base or metaclass may change what calling the class does (an Enum).
    if (
        isinstance(node, ast.ClassDef)
        and not node.decorator_list
        and not node.keywords
        and all(isinstance(b, ast.Name) and b.id == "object" for b in node.bases)
    ):
        inits = [
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef)
            and item.name == "__init__"
            and not item.decorator_list
        ]
        if inits:
            return node.name, inits[-1].args, True
    return None


def _bound_by(node: ast.stmt) -> set[str]:
    """The names a top-level statement binds (imports, assignments, loop and
    ``with`` targets, definitions), not inside a function or class body."""
    names: set[str] = set()

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(n.name)
            return
        if isinstance(n, ast.Lambda):
            return
        if isinstance(n, ast.Import | ast.ImportFrom):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            names.add(n.rest)
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return names


def _star_import(node: ast.stmt) -> bool:
    return isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)


def _declared_global(node: ast.stmt) -> set[str]:
    """Names a function in ``node`` declares ``global`` (it may rebind them
    when called)."""
    return {
        name for n in ast.walk(node) if isinstance(n, ast.Global) for name in n.names
    }


def _calls_run_now(node: ast.stmt) -> list[ast.Call]:
    """The calls a top-level statement makes when it runs: not those in a
    function, lambda or class body it defines (they run later, under their
    own scope)."""
    calls: list[ast.Call] = []

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            # Decorators and default values run now; the body runs later.
            for d in [
                *getattr(n, "decorator_list", []),
                *n.args.defaults,
                *(d for d in n.args.kw_defaults if d is not None),
            ]:
                visit(d)
            return
        if isinstance(n, ast.ClassDef):
            # A class body runs at definition (its methods later).
            for d in [*n.decorator_list, *n.bases, *(k.value for k in n.keywords)]:
                visit(d)
            for stmt in n.body:
                visit(stmt)
            return
        if isinstance(n, ast.Call):
            calls.append(n)
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(node)
    return calls


def _call_mismatch(call: ast.Call, args: ast.arguments, bound: bool) -> str:
    """Why ``call`` cannot bind to ``args`` ("" when it can or cannot be
    told: a ``*``/``**`` argument at the call)."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(
        k.arg is None for k in call.keywords
    ):
        return ""
    positional = [*args.posonlyargs, *args.args][1 if bound else 0 :]
    given = {k.arg for k in call.keywords if k.arg}
    names = {a.arg for a in [*positional, *args.kwonlyargs]}
    if args.kwarg is None and (unknown := sorted(given - names)):
        return f"unknown keyword argument(s) {', '.join(unknown)}"
    if args.vararg is None and len(call.args) > len(positional):
        return f"{len(call.args)} positional arguments for {len(positional)}"
    required = positional[: len(positional) - len(args.defaults)]
    missing = [
        a.arg
        for i, a in enumerate(required)
        if i >= len(call.args) and a.arg not in given
    ]
    missing += [
        a.arg
        for a, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
        if default is None and a.arg not in given
    ]
    return f"missing required argument(s) {', '.join(missing)}" if missing else ""


def _code_findings(
    sections: Sequence[tuple[str, str, bool]],
) -> list[_CodeFinding]:
    """The problems of the guide's Python code read in order,
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
    problems: list[_CodeFinding] = []
    # The post's own top-level functions and classes, as last defined.
    signatures: dict[str, tuple[ast.arguments, bool]] = {}
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
                    passage = f"({reason}): {code.strip()[:120]}"
                    problems.append(
                        _CodeFinding(
                            title, f"a python block does not parse {passage}", passage
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
                passage = code.strip()[:120]
                problems.append(
                    _CodeFinding(
                        title,
                        "the block has placeholders; show runnable code or "
                        "mark it as an excerpt (its first line "
                        f"`{EXCERPT_MARKER}`): {passage}",
                        passage,
                    )
                )
            bound = _bound_names(table)
            missing = sorted(_free_globals(table) - bound - known)
            if report and reliable and missing:
                problems.append(
                    _CodeFinding(
                        title,
                        "a python block uses names no earlier block or import "
                        f"defines: {', '.join(missing)}; define or import them "
                        "where a reader pasting the blocks in order has them",
                        names=tuple(missing),
                    )
                )
            # The running example's own definitions (an excerpt or an
            # internals sketch defines none a reader runs).
            if checked and not is_excerpt(code):
                problems += _signature_findings(title, tree, signatures, report=report)
            else:
                for stmt in tree.body:
                    if _star_import(stmt):
                        signatures.clear()
                    for name in _bound_by(stmt) | _declared_global(stmt):
                        signatures.pop(name, None)
            known |= bound
    return problems


def _signature_findings(
    title: str,
    tree: ast.Module,
    signatures: dict[str, tuple[ast.arguments, bool]],
    *,
    report: bool,
) -> list[_CodeFinding]:
    """Calls a block's top-level statements make, in order, to a function
    or class the running example defined before them, that do not bind to
    its parameters: a running example that defeats itself when pasted in
    order. ``signatures`` is updated as the statements run (a definition
    records one; any other binding of the name drops it); findings only
    with ``report``."""
    findings = []
    for stmt in tree.body:
        for call in _calls_run_now(stmt):
            if not (isinstance(call.func, ast.Name) and call.func.id in signatures):
                continue
            args, bound = signatures[call.func.id]
            if report and (why := _call_mismatch(call, args, bound)):
                text = ast.unparse(call)[:120]
                findings.append(
                    _CodeFinding(
                        title,
                        f"a call does not match the post's own definition of "
                        f"{call.func.id} ({why}): {text}",
                        text,
                    )
                )
        if _star_import(stmt):
            signatures.clear()
        for name in _bound_by(stmt) | _declared_global(stmt):
            signatures.pop(name, None)
        if (signature := _call_signature(stmt)) is not None:
            signatures[signature[0]] = signature[1:]
    return findings


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
