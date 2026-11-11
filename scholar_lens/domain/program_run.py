"""Running a guide's example as a reader pastes it (docs/engine.md, 4.7).

The static program check (:mod:`scholar_lens.domain.program`) reads the
code; this module decides what runs, wraps each block so its outcome comes
back as one line of JSON, and reads that outcome. Running it is a port's
(a sandbox with no credentials): a failure that only an environment with
credentials, network or a model would avoid stops the run without blaming
the post, and a failure of the code itself becomes a finding for the
section that holds it.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Collection, Sequence
from typing import Literal

from pydantic import BaseModel, field_validator

from scholar_lens.domain.program import (
    _ANY_FENCE,
    _session_code,
    is_excerpt,
    python_blocks,
)

# The line a wrapped block prints its outcome on (stdout may hold more).
RESULT_MARK = "@@SCHOLAR_LENS_RESULT@@"

# Exceptions a block's own text causes: the reader's paste fails the same
# way anywhere.
_SOURCE_ERRORS = frozenset(
    {"SyntaxError", "IndentationError", "TabError", "NameError", "UnboundLocalError"}
)
# Modules whose exceptions mean the sandbox lacks what the reader would have
# (credentials, network, a model or service endpoint).
_ENVIRONMENT_MODULES = (
    "botocore",
    "boto3",
    "urllib3",
    "requests",
    "httpx",
    "httpcore",
    "aiohttp",
    "socket",
    "ssl",
    "openai",
    "anthropic",
    "google",
    "grpc",
)
_ENVIRONMENT_ERRORS = frozenset(
    {
        "TimeoutError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "PermissionError",
        "BlockTimeout",
        "KeyboardInterrupt",
    }
)
# A TypeError of a call that does not match the callee's signature.
_SIGNATURE_PHRASES = (
    "unexpected keyword argument",
    "required positional argument",
    "required keyword-only argument",
    "positional argument but",
    "positional arguments but",
)
# Words of a message that point at the environment, not the code.
_ENVIRONMENT_WORDS = (
    "credential",
    "authenticat",
    "api key",
    "api_key",
    "token",
    "unauthorized",
    "forbidden",
    "connection",
    "timed out",
    "timeout",
    "network",
    "no such host",
    "name resolution",
    "cuda",
    "gpu",
)


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class BlockRun(BaseModel):
    """What one wrapped block did (the message without terminal colours,
    on one line)."""

    index: int
    ok: bool
    error_type: str = ""
    error_module: str = ""
    message: str = ""
    # The innermost frame that raised it is a block's (False: a package's).
    in_post: bool = True

    @field_validator("message")
    @classmethod
    def _plain(cls, value: str) -> str:
        return " ".join(_ANSI.sub("", value).split())


Verdict = Literal["ok", "code", "environment", "unclear"]


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
            if code.strip():
                out.append((title, code, runnable and not is_excerpt(block)))
    return out


def _distribution(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


# The shell blocks a reader runs (an unlabelled block too).
_SHELL_INFO = frozenset({"", "bash", "sh", "shell", "zsh", "console", "terminal"})
_PIP = re.compile(
    r"(?:^|[\s;&|(])(?:uv\s+)?(?:python3?\s+-m\s+)?pip3?\s+install\s+(.+)"
)
# pip options that take a value (the value is not a requirement).
_VALUED = frozenset(
    {
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


def post_installs(markdown: str) -> list[tuple[str, frozenset[str], str]]:
    """(distribution, extras, specifier) of each requirement the post's
    shell blocks ``pip install``: options and their values, files, paths and
    URLs left out (prose is not read)."""
    out: list[tuple[str, frozenset[str], str]] = []
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
    for block in shell:
        for line in block.splitlines():
            line = line.strip().removeprefix("$ ").split(" #")[0]
            if not (m := _PIP.search(line)):
                continue
            words = re.split(r"\s*(?:&&|\|\||;|\|)\s*", m.group(1))[0].split()
            skip = False
            for word in words:
                if skip:
                    skip = False
                    continue
                word = word.strip("'\"")
                if word.startswith("-"):
                    skip = word in _VALUED
                    continue
                if (spec := _SPEC.match(word)) is None or word.endswith(
                    (".txt", ".whl", ".gz", ".zip")
                ):
                    continue
                extras = frozenset(
                    e.strip().lower()
                    for e in (spec.group(2) or "").split(",")
                    if e.strip()
                )
                out.append((spec.group(1), extras, spec.group(3).replace(" ", "")))
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
    could, but gains nothing outside the sandbox). A failing block's later statements are skipped, the next
    block still runs (what it needs may be defined). A fresh interpreter,
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
_SL_CWD = _sl_os.path.realpath(_sl_os.getcwd())
class BlockTimeout(BaseException):
    pass
def _sl_alarm(signum, frame):
    raise BlockTimeout("the block ran over {block_timeout_s}s")
def _sl_in_post(e):
    # Whether the innermost frame that raised it (import machinery aside)
    # is a block's: else an installed package raised it.
    # A file under the working directory is the post's too (a module a
    # %%writefile cell wrote).
    tb, last = e.__traceback__, ""
    while tb is not None:
        name = tb.tb_frame.f_code.co_filename
        if not name.startswith("<frozen"):
            last = name
        tb = tb.tb_next
    if last.startswith("<block "):
        return True
    if last.startswith("<"):
        return False
    return _sl_os.path.realpath(last).startswith(_SL_CWD + _sl_os.sep)
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
            "error_module": str(getattr(type(_sl_e), "__module__", "")),
            "message": _sl_text(_sl_e),
            "in_post": _sl_in_post(_sl_e),
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


def _environment(run: BlockRun) -> bool:
    """Whether ``run``'s exception points at the environment: its type or
    module, or a word of its message outside the quoted names ("unexpected
    keyword argument 'max_tokens'" is the code's, "Could not resolve
    authentication" is not) and the callee's name ("requires_access_token()
    got ...")."""
    unquoted = re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`", "", run.message.lower())
    unquoted = re.sub(r"^[\w.<>]+\(\)\s*", "", unquoted)
    return (
        run.error_type in _ENVIRONMENT_ERRORS
        or run.error_module.split(".")[0] in _ENVIRONMENT_MODULES
        or any(word in unquoted for word in _ENVIRONMENT_WORDS)
    )


def verdict(run: BlockRun, product_modules: Collection[str]) -> Verdict:
    """Whose failure ``run`` is: the code's (a reader's paste fails the
    same), the environment's (credentials, network, a service, time), or
    unclear (an exception the code may or may not cause)."""
    if run.ok:
        return "ok"
    message = run.message.lower()
    if run.error_type in _SOURCE_ERRORS:
        return "code"
    if run.error_type in ("ModuleNotFoundError", "ImportError"):
        if not run.in_post:
            # A package's own import failed (a dependency at another
            # version).
            return "environment"
        products = {m.lower() for m in product_modules}
        named = message.split("'")[1] if message.count("'") >= 2 else ""
        if run.error_type == "ImportError" and "cannot import name" in message:
            # "cannot import name 'X' from 'pkg.mod' (...)": a product's.
            quoted = message.split("'")
            source = quoted[3] if len(quoted) > 3 else ""
            return "code" if source in products else "unclear"
        # A product module's submodule that does not exist is the code's
        # (its parent imported, so the product is installed); a product
        # root or a namespace package's part missing is an install that
        # failed (or a sandbox that installs nothing).
        parent = named.rpartition(".")[0]
        return "code" if parent and parent in products else "environment"
    if run.error_type == "TypeError" and any(
        phrase in message for phrase in _SIGNATURE_PHRASES
    ):
        # A call that does not match its signature is the code's, whatever
        # the callee is called, when the post's own line raised it (inside
        # a package, a keyword forwarded down or a dependency at another
        # version may be to blame).
        return "code" if run.in_post else "unclear"
    if _environment(run):
        # Before the call checks: an SDK raises a TypeError for a missing
        # key ("Could not resolve authentication method").
        return "environment"
    if run.error_type in ("TypeError", "AttributeError"):
        # A wrong call or attribute on the post's line, unless a missing
        # setting made a value None.
        if not run.in_post or "nonetype" in message:
            return "unclear"
        return "code"
    return "unclear"


def run_finding(
    title: str, block: str, run: BlockRun, *, after_gap: bool = False
) -> RunFinding:
    """A finding for the section whose block failed."""
    first = next((ln for ln in block.splitlines() if ln.strip()), "")[:80]
    gap = (
        " (an earlier block could not run in the sandbox; this failure does "
        "not depend on it)"
        if after_gap
        else ""
    )
    return RunFinding(
        section=title,
        problem=(
            f"a python block fails when run after the earlier blocks "
            f"({run.error_type}: {run.message[:200]}){gap}; block starts: {first}"
        ),
        block=block,
    )


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


def _reliable_after_gap(
    run: BlockRun, blocks: Sequence[tuple[str, str, bool]] = ()
) -> bool:
    """Whether a code failure after a block that could not run still says
    the code is wrong: an import, a module attribute or a call signature
    does not depend on what the failed block would have made (a name it
    would have bound, or an attribute of its object, may)."""
    message = run.message
    if run.error_type in ("ModuleNotFoundError", "ImportError"):
        return True
    if run.error_type == "AttributeError":
        # "module 'pkg' has no attribute 'storage'": unless an earlier block
        # imports pkg.storage (the import the gap may have skipped binds it).
        quoted = message.split("'")
        if not message.startswith("module ") or len(quoted) < 4:
            return False
        submodule = f"{quoted[1]}.{quoted[3]}"
        return not any(
            name == submodule or name.startswith(submodule + ".")
            for _, code, _ in blocks[: run.index]
            for name in _imported_modules(code)
        )
    if run.error_type == "TypeError":
        return any(phrase in message for phrase in _SIGNATURE_PHRASES)
    return False


def _bindings(code: str) -> tuple[set[str], set[str]]:
    """(names the block's top level defines or assigns, names it imports);
    empty when it does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set(), set()
    defined: set[str] = set()
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                defined |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return defined, imported


def _used(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def _missing_name(run: BlockRun) -> str:
    quoted = run.message.split("'")
    return quoted[1] if len(quoted) > 2 else ""


def _stored(blocks: Sequence[tuple[str, str, bool]], upto: int) -> set[str]:
    """Every name blocks 1..``upto`` bind anywhere (any depth, imports,
    function parameters and handlers included)."""
    names: set[str] = set()
    for _, code, _ in blocks[:upto]:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                names.add(node.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names.add(node.name)
    return names


def _shadowed(blocks: Sequence[tuple[str, str, bool]], before: int) -> set[str]:
    """Names the blocks before index ``before`` (1-based) last bound in an
    excerpt or a section not meant to run, by a definition or assignment
    (an illustration, such as a simplified class of the product, that the
    runnable code then uses in place of the real one)."""
    shadowed: set[str] = set()
    for _, code, checked in blocks[: before - 1]:
        defined, imported = _bindings(code)
        if checked:
            shadowed -= defined | imported
        else:
            shadowed = (shadowed - imported) | defined
    return shadowed


def judge_run(
    blocks: Sequence[tuple[str, str, bool]],
    outcomes: Sequence[BlockRun],
    product_modules: Collection[str],
    *,
    exact: bool = True,
) -> ProgramRun:
    """What running ``blocks`` (:func:`program_blocks`) showed: a finding
    for the first block of a runnable section that fails because of its
    code, and otherwise what limited the check.

    With ``exact`` False (the docs' packages are not at their versions)
    only failures of the code's own text count. A name the code binds
    somewhere but the run never bound is not blamed.

    A block that fails for the environment (or a reason the check cannot
    attribute, or in a section not meant to run) leaves a gap: what it would
    have defined is missing later, so after it only failures that cannot
    come from the gap count (:func:`_reliable_after_gap`). Blocks after a
    finding are not judged (they may fail only because of it)."""
    total = len(blocks)
    gap = ""
    gap_names: set[str] = set()
    for run in outcomes:
        if run.ok:
            continue
        title, code, checked = blocks[run.index - 1]
        whose = verdict(run, product_modules)
        if (
            whose == "code"
            and checked
            and _used(code) & _shadowed(blocks, run.index)
            and run.error_type not in _SOURCE_ERRORS
        ):
            # It may fail only because it calls an illustration's version.
            whose = "unclear"
        if whose == "code" and not exact and run.error_type not in _SOURCE_ERRORS:
            # The packages are not at the docs' versions: an import, an
            # attribute or a signature may differ in the installed one.
            whose = "unclear"
        if (
            whose == "code"
            and run.error_type in ("NameError", "UnboundLocalError")
            and _missing_name(run) in _stored(blocks, run.index)
        ):
            # The code binds it (the static check found the binding): a
            # binding the run skipped (a try whose body failed, a branch not
            # taken) is the environment's as likely as the code's.
            whose = "unclear"
        # A gap matters to a block that uses what the gap's blocks bind
        # (an unrelated later block is judged as if there were none).
        related = bool(gap) and bool(_used(code) & gap_names)
        if (
            whose == "code"
            and checked
            and (not related or _reliable_after_gap(run, blocks))
        ):
            return ProgramRun(
                findings=[run_finding(title, code, run, after_gap=related)],
                blocks_run=run.index,
                blocks_total=total,
                limited=gap,
            )
        # What the failed block binds, and the objects earlier blocks made
        # that it uses (a method it called may have been meant to set their
        # state); not modules or builtins it uses.
        made = set().union(*(_bindings(c)[0] for _, c, _ in blocks[: run.index - 1]))
        gap_names |= _stored(blocks[run.index - 1 : run.index], 1) | (
            _used(code) & made
        )
        if not gap:
            reason = {
                "code": "fails, but is an excerpt or in a section not meant to run",
                "environment": "needs what the sandbox lacks",
                "unclear": "raised an error the check cannot attribute",
            }[whose]
            gap = (
                f"block {run.index} of {total} ({title}) {reason}: "
                f"{run.error_type}: {run.message[:160]}"
            )
    done = len(outcomes)
    if done < total:
        ended = f"the run ended after block {done} of {total} (killed or timed out)"
        gap = f"{gap}; {ended}" if gap else ended
    return ProgramRun(blocks_run=done, blocks_total=total, limited=gap)
