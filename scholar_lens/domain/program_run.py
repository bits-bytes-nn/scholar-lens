"""Running a guide's example as a reader pastes it (docs/engine.md, 4.7).

The static program check (:mod:`scholar_lens.domain.program`) reads the
code; this module decides what runs, wraps each block so its outcome comes
back as one line of JSON, and reads that outcome. Running it is a port's
(a sandbox without the reader's credentials, network settings or data).

The post is blamed only for failures that point at the product's API
itself, whatever the sandbox lacks: a call that does not match a product
callable's signature, an import of a product name or submodule that does
not exist, a product module's missing attribute, and an SDK's own
parameter or service validation. Every other failure leaves the check
partial (a PR note), never a fix: telling a missing credential from a
wrong value is guesswork the sandbox cannot settle.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
from collections.abc import Collection, Sequence

from pydantic import BaseModel, field_validator

from scholar_lens.domain.program import (
    _ANY_FENCE,
    _session_code,
    is_excerpt,
    python_blocks,
)

# The line a wrapped block prints its outcome on (stdout may hold more).
RESULT_MARK = "@@SCHOLAR_LENS_RESULT@@"
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# A TypeError of a call that does not match the callee's signature, and the
# callee it names ("Agent.__init__() got an unexpected keyword argument").
_SIGNATURE_PHRASES = (
    "unexpected keyword argument",
    "required positional argument",
    "required keyword-only argument",
    "positional argument but",
    "positional arguments but",
)
_CALLEE = re.compile(r"^([A-Za-z_][\w.]*)\(\)")
# An SDK's own validation of a call before it is sent (botocore): a wrong
# parameter or service name. A wrong value (None from an unset variable,
# an empty string) is not among them.
_SDK_CALL_ERRORS = {
    "ParamValidationError": ("unknown parameter", "missing required parameter"),
    "UnknownServiceError": ("",),
}


class BlockRun(BaseModel):
    """What one wrapped block did (the message without terminal colours,
    on one line)."""

    index: int
    ok: bool
    error_type: str = ""
    message: str = ""
    # The modules of the traceback's frames, innermost first ("__main__":
    # a block or a file of the working directory; import machinery left
    # out): who raised it.
    raised_in: list[str] = []

    @field_validator("message")
    @classmethod
    def _plain(cls, value: str) -> str:
        return " ".join(_ANSI.sub("", value).split())


class RunFinding(BaseModel):
    """A block of a runnable section that fails as pasted, for the section's
    fix."""

    section: str
    problem: str
    block: str


class ProgramRun(BaseModel):
    """The outcome of running the post's code in order: the finding (the
    first block the code is to blame for), the blocks judged, and what
    limited the check ("" when every block ran and counted)."""

    findings: list[RunFinding] = []
    blocks_run: int = 0
    blocks_total: int = 0
    limited: str = ""


def program_blocks(
    sections: Sequence[tuple[str, str, bool]],
) -> list[tuple[str, str, bool]]:
    """(section title, code, checked) of every block a reader pastes, in
    order: the Python blocks of every section (definitions in an earlier
    section count), a REPL transcript or notebook cell as the code it runs.
    ``checked``: the block is in a runnable section and not an excerpt (an
    excerpt runs, so what it defines is there later, as the static check
    counts it, but is not blamed)."""
    out = []
    for title, markdown, runnable in sections:
        for block in python_blocks(markdown):
            session = _session_code(block)
            code = block if session is None else session
            # A transcript that shows a traceback demonstrates the error.
            shows_error = (
                session is not None and "Traceback (most recent call last)" in block
            )
            if code.strip():
                out.append(
                    (
                        title,
                        code,
                        runnable and not is_excerpt(block) and not shows_error,
                    )
                )
    return out


def _distribution(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


# The shell blocks a reader runs (an unlabelled block too).
_SHELL_INFO = frozenset({"", "bash", "sh", "shell", "zsh", "console", "terminal"})
_PIP = re.compile(
    r"(?:^|[\s;&|(])(?:uv\s+)?(?:python3?\s+-m\s+)?pip3?\s+install\s+(.+)"
)
# pip's and uv pip's options that take a value as the next word (the value
# is not a requirement).
_VALUED = frozenset(
    {
        "--abi",
        "--cache-dir",
        "--exists-action",
        "--implementation",
        "--index-strategy",
        "--keyring-provider",
        "--log",
        "--progress-bar",
        "--python-version",
        "--retries",
        "--src",
        "--timeout",
        "--torch-backend",
        "--trusted-host",
        "--upgrade-strategy",
        "-r",
        "--requirement",
        "-c",
        "--constraint",
        "-t",
        "--target",
        "-i",
        "--index-url",
        "--extra-index-url",
        "-f",
        "--find-links",
        "-e",
        "--editable",
        "--prefix",
        "--root",
        "--python",
        "--platform",
    }
)
# A PEP 508 name, extras and version specifiers (no URL, path or marker).
_SPEC = re.compile(
    r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?:\[([A-Za-z0-9._,\s-]*)\])?"
    r"((?:\s*(?:===|==|!=|~=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+\s*,?)*)$"
)


Requirement = tuple[str, frozenset[str], str]


def _line_requirements(line: str) -> list[Requirement]:
    """(distribution, extras, specifier) of the requirements one shell
    line ``pip install``s: options and their values, files, paths and URLs
    left out; [] for a line that installs nothing."""
    line = line.strip().removeprefix("$ ").split(" #")[0]
    if not (m := _PIP.search(line)):
        return []
    out: list[Requirement] = []
    command = re.split(r"\s*(?:&&|\|\||;|\|)\s*", m.group(1))[0]
    try:
        # Quotes keep a requirement whole ("ray[serve] >= 2.9").
        words = shlex.split(command)
    except ValueError:
        words = [w.strip("'\"") for w in command.split()]
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        if word.startswith("-"):
            skip = word in _VALUED
            continue
        if (spec := _SPEC.match(word)) is None or word.endswith(
            (".txt", ".whl", ".gz", ".zip")
        ):
            continue
        extras = frozenset(
            e.strip().lower() for e in (spec.group(2) or "").split(",") if e.strip()
        )
        out.append((spec.group(1), extras, spec.group(3).replace(" ", "")))
    return out


def post_installs(markdown: str) -> list[Requirement]:
    """What the post's shell blocks and notebook cells ``pip install``
    (prose is not read)."""
    # Fences are paired as the static check pairs them (a block's closing
    # fence never opens another).
    shell = [
        m.group(4)
        for m in _ANY_FENCE.finditer(markdown)
        if m.group(3).lower() in _SHELL_INFO
    ]
    # A notebook cell's %pip / !pip lines install too.
    shell += [
        "\n".join(
            ln.lstrip()[1:]
            for ln in block.splitlines()
            if ln.lstrip().startswith(("%pip ", "!pip ", "!python ", "%uv ", "!uv "))
        )
        for block in python_blocks(markdown)
    ]
    return [
        r
        for block in shell
        for line in block.splitlines()
        for r in _line_requirements(line)
    ]


# A line that is a pip command once its list marker, shell prompt or
# notebook prefix is off ("- ", "$ ", "(.venv) $ ", "!", "%").
_COMMAND_LINE = re.compile(r"^(?:uv\s+)?(?:python3?\s+-m\s+)?pip3?\s+install\s")
_LINE_LEAD = re.compile(r"^\s*(?:[-*+]\s+)?(?:\(\S+\)\s*)?(?:\$\s*)?[!%]?")
_CODE_SPAN = re.compile(r"`([^`\n]+)`")


def command_installs(text: str) -> list[Requirement]:
    """What a docs page's pip commands install: a line that is one (its
    code may have lost its fences), and an inline code span that is one
    ("Install it with `pip install x`"); a sentence that only mentions pip
    is not read."""
    out: list[Requirement] = []
    for line in text.splitlines():
        commands = [m.group(1) for m in _CODE_SPAN.finditer(line)]
        commands.append(_LINE_LEAD.sub("", line, count=1))
        for command in commands:
            command = _LINE_LEAD.sub("", command, count=1)
            if _COMMAND_LINE.match(command):
                out += _line_requirements(command)
    return out


def run_requirements(pins: Sequence[tuple[str, str]], markdown: str) -> list[str]:
    """What the sandbox installs before the post's code: the packages the
    docs install at the run's versions (``pins``), with the extras the post
    names, and the other packages the post's shell blocks ``pip install``,
    with their version specifiers."""
    asked: dict[str, tuple[str, set[str], str]] = {}
    for name, extras, specifier in post_installs(markdown):
        key = _distribution(name)
        _, known, earlier = asked.get(key, (name, set(), ""))
        asked[key] = (name, known | extras, earlier or specifier)

    def named(name: str, extras: set[str]) -> str:
        return f"{name}[{','.join(sorted(extras))}]" if extras else name

    out = []
    for name, version in pins:
        _, wanted, _ = asked.pop(_distribution(name), (name, set(), ""))
        out.append(f"{named(name, wanted)}=={version}")
    return out + [named(n, e) + v for n, e, v in asked.values()]


def program_script(
    codes: Sequence[str], *, block_timeout_s: int, mark: str = RESULT_MARK
) -> str:
    """One Python program that runs ``codes`` in order in one namespace (a
    reader's session: state carries over, ``__name__`` is ``"__main__"``),
    each under an alarm, writing each block's outcome as one JSON line after
    ``mark`` (:data:`RESULT_MARK` and a per-run nonce, so code printing
    a mark by accident is not read as an outcome; code set on forging one
    could, but gains nothing outside the sandbox). A failing block's later
    statements are skipped, the next block still runs (what it needs may be
    defined). A fresh interpreter,
    not a notebook kernel: ``asyncio.run`` works as pasted.

    The outcomes go to a copy of the output descriptor taken before any
    block runs (a block that replaces or closes ``sys.stdout`` cannot hide
    them), each in one write; the timeout is a ``BaseException`` (an
    ``except Exception`` retry loop does not swallow it)."""
    return f"""
import json as _sl_json, os as _sl_os, signal as _sl_signal, sys as _sl_sys
import ast as _sl_ast, asyncio as _sl_asyncio, inspect as _sl_inspect
_SL_CODES = {list(codes)!r}
_SL_MARK = {mark!r}
_sl_fd = _sl_os.dup(1)
# The working directory is importable, as in a notebook (a module a
# %%writefile cell wrote).
_sl_sys.path.insert(0, _sl_os.getcwd())
class BlockTimeout(BaseException):
    pass
def _sl_alarm(signum, frame):
    raise BlockTimeout("the block ran over {block_timeout_s}s")
def _sl_where(e):
    cwd, out, tb = _sl_os.getcwd() + _sl_os.sep, [], e.__traceback__
    while tb is not None:
        f = tb.tb_frame
        name = f.f_code.co_filename
        if not name.startswith("<frozen"):
            mod = str(f.f_globals.get("__name__", ""))
            out.append("__main__" if name.startswith(cwd) else mod)
        tb = tb.tb_next
    return out[::-1][:40]
def _sl_text(e):
    try:
        return str(e)[:400]
    except BaseException:
        return "<unprintable>"
import __future__ as _sl_future, linecache as _sl_linecache, types as _sl_types
if __name__ == "__main__":
    # The blocks run in a __main__ module of their own (with a __file__).
    _sl_main = _sl_types.ModuleType("__main__")
    _sl_main.__file__ = _sl_os.path.abspath(
        globals().get("__file__") or _sl_sys.argv[0] or "main.py"
    )
    _sl_sys.modules["__main__"] = _sl_main
    _sl_ns = _sl_main.__dict__
    # A notebook's display() (prints, here).
    _sl_ns["display"] = print
else:
    # A child multiprocessing spawns runs this script as "__mp_main__" and
    # keeps its globals as the parent's __main__: the blocks define there
    # (a guarded block does not run, and the child writes no outcome).
    _sl_ns = globals()
_SL_FUTURE = 0
for _sl_f in _sl_future.all_feature_names:
    _SL_FUTURE |= getattr(_sl_future, _sl_f).compiler_flag
# A __future__ import holds for the later blocks, as in one file.
_sl_flags = 0
for _sl_i, _sl_code in enumerate(_SL_CODES, 1):
    _sl_name = f"<block {{_sl_i}}>"
    _sl_linecache.cache[_sl_name] = (
        len(_sl_code), None, _sl_code.splitlines(True), _sl_name
    )
    _sl_signal.signal(_sl_signal.SIGALRM, _sl_alarm)
    _sl_signal.alarm({block_timeout_s})
    try:
        _sl_c = compile(
            _sl_code,
            _sl_name,
            "exec",
            flags=_sl_ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | _sl_flags,
            dont_inherit=True,
        )
        _sl_flags |= _sl_c.co_flags & _SL_FUTURE
        if _sl_c.co_flags & _sl_inspect.CO_COROUTINE:
            # Top-level await, as a notebook or `python -m asyncio` runs it.
            _sl_asyncio.run(eval(_sl_c, _sl_ns))
        else:
            exec(_sl_c, _sl_ns)
        _sl_out = {{"index": _sl_i, "ok": True}}
    except BaseException as _sl_e:
        _sl_ok = isinstance(_sl_e, SystemExit) and _sl_e.code in (0, None)
        _sl_out = {{"index": _sl_i, "ok": True}} if _sl_ok else {{
            "index": _sl_i,
            "ok": False,
            "error_type": type(_sl_e).__name__,
            "message": _sl_text(_sl_e),
            "raised_in": _sl_where(_sl_e),
        }}
    finally:
        _sl_signal.alarm(0)
    try:
        _sl_sys.stdout.flush()
    except BaseException:
        pass
    if __name__ == "__main__":
        _sl_os.write(
            _sl_fd, ("\\n" + _SL_MARK + _sl_json.dumps(_sl_out) + "\\n").encode()
        )
"""


def read_outcomes(stdout: str, mark: str = RESULT_MARK) -> list[BlockRun]:
    """The outcomes :func:`program_script` wrote after ``mark``, in block
    order (fewer than its blocks when it was killed); a line that does not
    parse or is out of order is skipped."""
    out: list[BlockRun] = []
    for line in stdout.splitlines():
        _, found, rest = line.partition(mark)
        if not found:
            continue
        try:
            run = BlockRun.model_validate(json.loads(rest))
        except ValueError:
            continue
        if run.index == len(out) + 1:
            out.append(run)
    return out


def run_mark(nonce: str) -> str:
    """The outcome mark of one run (``nonce``: random, unknown to the
    code)."""
    return f"{RESULT_MARK}{nonce}@@"


def _product_names(
    blocks: Sequence[tuple[str, str, bool]], products: Collection[str]
) -> set[str]:
    """Names the blocks import from a product module and never rebind
    (a ``def``, ``class`` or assignment of the same name, an illustration
    of the product's class, makes it the post's own)."""
    imported: set[str] = set()
    rebound: set[str] = set()
    for _, code, _ in blocks:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if _is_product(node.module, products):
                    # The error names the original ("Agent.__init__()").
                    imported |= {a.asname or a.name for a in node.names}
                    imported |= {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {
                    (a.asname or a.name).split(".")[0]
                    for a in node.names
                    if _is_product(a.name, products)
                }
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                rebound.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                rebound.add(node.id)
        # A product name called through its module ("serve.deployment(...)"
        # fails as "deployment()").
        imported |= {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in imported
        }
    return imported - rebound


def _is_product(module: str, products: Collection[str]) -> bool:
    """Whether ``module`` is one of the product's (``products``: its module
    names, dotted, lower-cased; a namespace package's root is not)."""
    module = module.lower()
    return any(module == p or module.startswith(p + ".") for p in products)


def _quoted(message: str) -> list[str]:
    return message.split("'")[1::2]


# The SDKs whose own validation of a call is the post's misuse.
_SDK_MODULES = ("botocore", "boto3")


def _raised_by_post(run: BlockRun, own: Collection[str]) -> bool:
    """Whether the post's code raised it: the innermost frame outside the
    ``own`` modules (the product's, raising for the call it got) is a block
    or a file of the working directory, not another package that uses the
    product (a plugin built for another version). No frames: the call."""
    for module in run.raised_in:
        if module == "__main__":
            return True
        if not _is_product(module, own):
            return False
    return True


def blamed(run: BlockRun, products: Collection[str], product_names: set[str]) -> bool:
    """Whether ``run`` failed on the product's API itself (the module
    docstring's four cases), whatever else the sandbox lacks, raised by the
    post's own code (:func:`_raised_by_post`)."""
    if run.ok:
        return False
    message = run.message
    lowered = message.lower()
    quoted = _quoted(message)
    if run.error_type in _SDK_CALL_ERRORS:
        return _raised_by_post(run, {*products, *_SDK_MODULES}) and any(
            phrase in lowered for phrase in _SDK_CALL_ERRORS[run.error_type]
        )
    if not _raised_by_post(run, products):
        return False
    if run.error_type == "TypeError":
        callee = _CALLEE.match(message)
        return bool(
            callee
            and callee.group(1).split(".")[0] in product_names
            and any(phrase in message for phrase in _SIGNATURE_PHRASES)
        )
    if run.error_type == "ImportError" and "cannot import name" in lowered:
        # "cannot import name 'X' from 'pkg.mod' (...)".
        return len(quoted) > 1 and _is_product(quoted[1], products)
    if run.error_type == "ModuleNotFoundError" and quoted:
        # A submodule missing under a product module that imported fine.
        parent = quoted[0].rpartition(".")[0]
        return bool(parent) and parent.lower() in {p.lower() for p in products}
    if run.error_type == "AttributeError" and message.startswith("module "):
        # "module 'pkg' has no attribute 'X'".
        return len(quoted) > 1 and quoted[0].lower() in {p.lower() for p in products}
    return False


def _imported_modules(code: str) -> set[str]:
    """The dotted modules ``code``'s import statements name (any depth)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def run_finding(title: str, block: str, run: BlockRun) -> RunFinding:
    """A finding for the section whose block failed."""
    first = next((ln for ln in block.splitlines() if ln.strip()), "")[:80]
    return RunFinding(
        section=title,
        problem=(
            "a python block fails when run after the earlier blocks "
            f"({run.error_type}: {run.message[:200]}); block starts: {first}"
        ),
        block=block,
    )


def judge_run(
    blocks: Sequence[tuple[str, str, bool]],
    outcomes: Sequence[BlockRun],
    product_modules: Collection[str],
    *,
    exact: bool,
) -> ProgramRun:
    """What running ``blocks`` (:func:`program_blocks`) showed: a finding
    for the first block of a runnable section that fails on the product's
    API (:func:`blamed`), and otherwise what limited the check.

    With ``exact`` False (the docs' packages are not at their versions) the
    product's API may differ from the post's version: nothing is blamed. A
    module attribute a skipped import would have bound (an earlier block
    failed before ``import pkg.sub``) is not blamed either."""
    products = {m.lower() for m in product_modules}
    names = _product_names(blocks, products)
    total = len(blocks)
    limited = ""
    failed_before = False
    for run in outcomes:
        if run.ok:
            continue
        title, code, checked = blocks[run.index - 1]
        is_code = exact and checked and blamed(run, products, names)
        if is_code and run.error_type == "AttributeError" and failed_before:
            quoted = _quoted(run.message)
            submodule = f"{quoted[0]}.{quoted[1]}"
            is_code = not any(
                name == submodule or name.startswith(submodule + ".")
                for _, earlier, _ in blocks[: run.index]
                for name in _imported_modules(earlier)
            )
        if is_code:
            return ProgramRun(
                findings=[run_finding(title, code, run)],
                blocks_run=run.index,
                blocks_total=total,
                limited=limited,
            )
        failed_before = True
        limited = limited or (
            f"block {run.index} of {total} ({title}) failed in the sandbox: "
            f"{run.error_type}: {run.message[:160]}"
        )
    done = len(outcomes)
    if done < total:
        ended = f"the run ended after block {done} of {total} (killed or timed out)"
        limited = f"{limited}; {ended}" if limited else ended
    return ProgramRun(blocks_run=done, blocks_total=total, limited=limited)
