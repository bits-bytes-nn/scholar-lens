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

import json
import re
from collections.abc import Collection, Sequence
from typing import Literal

from pydantic import BaseModel, field_validator

from scholar_lens.domain.guide.evidence_guide import install_requirements
from scholar_lens.domain.program import _session_code, is_excerpt, python_blocks

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
        "SystemExit",
    }
)
# Words of a message that point at the environment, not the code.
_ENVIRONMENT_WORDS = (
    "credential",
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


def run_requirements(pins: Sequence[tuple[str, str]], markdown: str) -> list[str]:
    """What the sandbox installs before the post's code: the packages the
    docs install at the run's versions (``pins``) and what the post tells
    the reader to ``pip install``, with the extras the post names."""
    asked: dict[str, set[str]] = {}
    for name, extras in install_requirements(markdown):
        asked.setdefault(_distribution(name), set()).update(extras)
    def spec(name: str, extras: set[str]) -> str:
        return f"{name}[{','.join(sorted(extras))}]" if extras else name

    out = [
        f"{spec(name, asked.pop(_distribution(name), set()))}=={version}"
        for name, version in pins
    ]
    return out + [spec(name, extras) for name, extras in asked.items()]


def program_script(codes: Sequence[str], *, block_timeout_s: int) -> str:
    """One Python program that runs ``codes`` in order in one namespace (a
    reader's session: state carries over, ``__name__`` is ``"__main__"``),
    each under an alarm, printing each block's outcome as one JSON line after
    :data:`RESULT_MARK`. A failing block's later statements are skipped, the
    next block still runs (what it needs may be defined). A fresh
    interpreter, not a notebook kernel: ``asyncio.run`` works as pasted."""
    return f"""
import json as _sl_json, signal as _sl_signal, sys as _sl_sys
_SL_CODES = {list(codes)!r}
class BlockTimeout(Exception):
    pass
def _sl_alarm(signum, frame):
    raise BlockTimeout("the block ran over {block_timeout_s}s")
_sl_signal.signal(_sl_signal.SIGALRM, _sl_alarm)
_sl_ns = {{"__name__": "__main__"}}
for _sl_i, _sl_code in enumerate(_SL_CODES, 1):
    _sl_signal.alarm({block_timeout_s})
    try:
        exec(compile(_sl_code, f"<block {{_sl_i}}>", "exec"), _sl_ns)
        _sl_out = {{"index": _sl_i, "ok": True}}
    except BaseException as _sl_e:
        _sl_out = {{
            "index": _sl_i,
            "ok": False,
            "error_type": type(_sl_e).__name__,
            "error_module": type(_sl_e).__module__,
            "message": str(_sl_e)[:400],
        }}
    finally:
        _sl_signal.alarm(0)
    _sl_sys.stdout.flush()
    print("\\n" + {RESULT_MARK!r} + _sl_json.dumps(_sl_out), flush=True)
"""


def read_outcomes(stdout: str) -> list[BlockRun]:
    """The outcomes :func:`program_script` printed, in block order (fewer
    than its blocks when it was killed)."""
    out: list[BlockRun] = []
    for line in stdout.splitlines():
        _, mark, rest = line.partition(RESULT_MARK)
        if not mark:
            continue
        try:
            run = BlockRun.model_validate(json.loads(rest))
        except ValueError:
            continue
        if run.index == len(out) + 1:
            out.append(run)
    return out


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
        # A product module the post imports wrongly is the code's; a module
        # the sandbox lacks (one the post never installs) is the
        # environment's.
        named = message.split("'")[1] if message.count("'") >= 2 else ""
        top = named.split(".")[0]
        if top in product_modules or (
            run.error_type == "ImportError" and "cannot import name" in message
        ):
            return "code"
        return "environment"
    if run.error_type in ("TypeError", "AttributeError"):
        # A wrong call or attribute, unless a missing setting made a value
        # None.
        return "unclear" if "nonetype" in message else "code"
    module = run.error_module.split(".")[0]
    if (
        run.error_type in _ENVIRONMENT_ERRORS
        or module in _ENVIRONMENT_MODULES
        or any(word in message for word in _ENVIRONMENT_WORDS)
    ):
        return "environment"
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


def _reliable_after_gap(run: BlockRun) -> bool:
    """Whether a code failure after a block that could not run still says
    the code is wrong: an import, a module attribute or a call signature
    does not depend on what the failed block would have made (a name it
    would have bound, or an attribute of its object, may)."""
    message = run.message
    if run.error_type in ("ModuleNotFoundError", "ImportError"):
        return True
    if run.error_type == "AttributeError":
        return message.startswith("module ")
    if run.error_type == "TypeError":
        return any(
            phrase in message
            for phrase in (
                "unexpected keyword argument",
                "required positional argument",
                "required keyword-only argument",
                "positional argument but",
                "positional arguments but",
            )
        )
    return False


def judge_run(
    blocks: Sequence[tuple[str, str, bool]],
    outcomes: Sequence[BlockRun],
    product_modules: Collection[str],
) -> ProgramRun:
    """What running ``blocks`` (:func:`program_blocks`) showed: a finding
    for the first block of a runnable section that fails because of its
    code, and otherwise what limited the check.

    A block that fails for the environment (or a reason the check cannot
    attribute, or in a section not meant to run) leaves a gap: what it would
    have defined is missing later, so after it only failures that cannot
    come from the gap count (:func:`_reliable_after_gap`). Blocks after a
    finding are not judged (they may fail only because of it)."""
    total = len(blocks)
    gap = ""
    for run in outcomes:
        if run.ok:
            continue
        title, code, checked = blocks[run.index - 1]
        whose = verdict(run, product_modules)
        if whose == "code" and checked and (not gap or _reliable_after_gap(run)):
            return ProgramRun(
                findings=[run_finding(title, code, run, after_gap=bool(gap))],
                blocks_run=run.index,
                blocks_total=total,
                limited=gap,
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
    if done < total and not gap:
        gap = f"the run ended after block {done} of {total} (killed or timed out)"
    return ProgramRun(blocks_run=done, blocks_total=total, limited=gap)
