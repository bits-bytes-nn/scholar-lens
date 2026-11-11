"""The code sandbox on Amazon Bedrock AgentCore Code Interpreter.

Each run starts a session of the configured interpreter, installs the
requirements and runs the program in a fresh interpreter process (not the
session's notebook kernel, so ``asyncio.run`` works as pasted), and stops the
session. The interpreter runs without the job's credentials: the custom
interpreter the stack creates has no execution role, and its public network
mode reaches PyPI. Only the program's outcome lines come back, so a block that
prints a lot cannot crowd them out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    TokenRetrievalError,
)
from urllib3.exceptions import HTTPError

from scholar_lens.adapters.aws.session import AwsSession
from scholar_lens.application.ports import SandboxRun, SandboxUnavailable
from scholar_lens.domain.program_run import RESULT_MARK

logger = logging.getLogger(__name__)

# The interpreter AWS provides in every account (no network: nothing can be
# installed, so the run uses what it ships with).
SYSTEM_INTERPRETER = "aws.codeinterpreter.v1"
# The time pip may take before the program runs.
INSTALL_TIMEOUT_S = 300
_INSTALL_MARK = "@@SCHOLAR_LENS_INSTALL_FAILED@@"
_VERSIONS_MARK = "@@SCHOLAR_LENS_VERSIONS@@"
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# A session lives a little longer than the work it is started for.
_SESSION_MARGIN_S = 120
_MAX_SESSION_S = 28_800


def install_code(requirements: Sequence[str]) -> str:
    """The notebook cell that installs ``requirements`` one at a time and
    prints the ones that failed and the version each pin ended up at."""
    return f"""
import json, os, subprocess, sys, tempfile, time
_reqs = {list(requirements)!r}
_failed = []
_detail = []
_deadline = time.monotonic() + {INSTALL_TIMEOUT_S}
for _req in _reqs:
    # One at a time: a requirement that cannot be installed leaves the rest.
    _left = _deadline - time.monotonic()
    if _left <= 0:
        _failed.append(_req)
        _detail.append(_req + ": no time left")
        continue
    try:
        _p = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", _req],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_left,
        )
        if _p.returncode:
            _failed.append(_req)
            _detail.append(_req + ": " + (_p.stderr or _p.stdout)[-200:])
    except subprocess.TimeoutExpired:
        _failed.append(_req)
        _detail.append(_req + ": timed out")
if _failed:
    print({_INSTALL_MARK!r} + json.dumps([_failed, "; ".join(_detail)]))
# The version each pin ended up at (a later install may have moved it, a
# failed one leaves the preinstalled version).
import importlib.metadata as _md
_versions = {{}}
for _req in _reqs:
    if "==" in _req:
        _name = _req.split("==")[0].split("[")[0]
        try:
            _versions[_req] = _md.version(_name)
        except _md.PackageNotFoundError:
            _versions[_req] = None
print({_VERSIONS_MARK!r} + json.dumps(_versions))
"""


def run_code(program: str, timeout_s: int) -> str:
    """The notebook cell that runs ``program`` in a fresh interpreter and
    prints the program's outcome lines only."""
    return f"""
import os, subprocess, sys, tempfile
with tempfile.NamedTemporaryFile(
    "w", suffix=".py", delete=False, encoding="utf-8"
) as _f:
    _f.write({program!r})
try:
    _out = subprocess.run(
        [sys.executable, _f.name], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout={timeout_s},
        cwd=tempfile.mkdtemp(), env={{**os.environ, "PYTHONIOENCODING": "utf-8"}},
    ).stdout
except subprocess.TimeoutExpired as _e:
    _out = _e.stdout or ""
    if isinstance(_out, bytes):
        _out = _out.decode("utf-8", "replace")
print("\\n".join(l for l in _out.splitlines() if {RESULT_MARK!r} in l))
"""


def read_stream(response: dict[str, Any]) -> tuple[str, str, bool]:
    """(stdout, stderr, is_error) of an InvokeCodeInterpreter response.

    Raises:
        SandboxUnavailable: The stream carried an exception event.
    """
    stdout: list[str] = []
    stderr: list[str] = []
    is_error = False
    for event in response["stream"]:
        if "result" not in event:
            name, detail = next(iter(event.items()))
            raise SandboxUnavailable(f"{name}: {detail.get('message', detail)}")
        result = event["result"]
        content = result.get("structuredContent") or {}
        stdout.append(content.get("stdout") or "")
        stderr.append(content.get("stderr") or "")
        is_error = is_error or bool(result.get("isError"))
    return "".join(stdout), "".join(stderr), is_error


class AgentCoreCodeRunner:
    """:class:`~scholar_lens.application.ports.CodeRunner` on an AgentCore
    Code Interpreter (``interpreter_id``: the stack's custom interpreter, or
    :data:`SYSTEM_INTERPRETER`)."""

    def __init__(self, session: AwsSession, interpreter_id: str) -> None:
        self.session = session
        self.interpreter_id = interpreter_id

    @property
    def name(self) -> str:
        return f"agentcore:{self.interpreter_id}"

    async def run(
        self, program: str, *, requirements: Sequence[str], timeout_s: int
    ) -> SandboxRun:
        return await asyncio.to_thread(
            self._run, program, list(requirements), timeout_s
        )

    def _run(self, program: str, requirements: list[str], timeout_s: int) -> SandboxRun:
        skipped = ""
        if self.interpreter_id == SYSTEM_INTERPRETER and requirements:
            # Its preinstalled packages are of other versions: only what
            # cannot depend on a version is judged.
            skipped = (
                "the system interpreter has no network, so "
                f"{', '.join(requirements)[:200]} were not installed"
            )
            logger.info("Code sandbox: %s.", skipped)
            requirements = []
        work_s = timeout_s + (INSTALL_TIMEOUT_S if requirements else 0)
        client = self.session.client(
            "bedrock-agentcore",
            config=BotoConfig(
                retries={"mode": "standard", "max_attempts": 3},
                connect_timeout=10,
                read_timeout=60,
            ),
        )
        # The invocation is not retried: a retry would run the install and
        # the program again in a session that may expire meanwhile.
        runner = self.session.client(
            "bedrock-agentcore",
            config=BotoConfig(
                retries={"mode": "standard", "max_attempts": 1},
                connect_timeout=10,
                read_timeout=work_s + 60,
            ),
        )
        try:
            session_id = client.start_code_interpreter_session(
                codeInterpreterIdentifier=self.interpreter_id,
                name=f"scholar-lens-{uuid.uuid4().hex[:12]}",
                # A retried start reuses the session it may have created.
                clientToken=uuid.uuid4().hex + uuid.uuid4().hex[:8],
                sessionTimeoutSeconds=min(work_s + _SESSION_MARGIN_S, _MAX_SESSION_S),
            )["sessionId"]
        except (NoCredentialsError, TokenRetrievalError):
            raise
        except (ClientError, BotoCoreError) as e:
            raise SandboxUnavailable(f"could not start a session: {e}") from e
        try:
            # Two invocations, each well inside the 15-minute limit of a
            # synchronous call: the install, then the program.
            cells = [run_code(program, timeout_s)]
            if requirements:
                cells.insert(0, install_code(requirements))
            stdout = stderr = ""
            is_error = False
            for cell in cells:
                out, err, failed = read_stream(
                    runner.invoke_code_interpreter(
                        codeInterpreterIdentifier=self.interpreter_id,
                        sessionId=session_id,
                        name="executeCode",
                        arguments={"language": "python", "code": cell},
                    )
                )
                stdout, stderr = stdout + out + "\n", stderr + err
                if failed:
                    is_error = True
                    break
        except (NoCredentialsError, TokenRetrievalError):
            raise
        except (ClientError, BotoCoreError, HTTPError) as e:
            # HTTPError: the event stream reads urllib3 directly (a read
            # timeout or a dropped connection mid-stream).
            raise SandboxUnavailable(f"could not run the code: {e}") from e
        finally:
            try:
                client.stop_code_interpreter_session(
                    codeInterpreterIdentifier=self.interpreter_id,
                    sessionId=session_id,
                )
            except (ClientError, BotoCoreError) as e:
                # The session times out on its own.
                logger.warning("Could not stop code session %s: %s", session_id, e)
        if is_error:
            raise SandboxUnavailable(f"the driver failed: {stderr[-300:]}")
        warnings = [skipped] if skipped else []
        exact = not skipped
        for line in stdout.splitlines():
            # The driver's own lines (the program's lines carry the result
            # mark).
            if RESULT_MARK in line:
                continue
            if line.startswith(_INSTALL_MARK):
                try:
                    _failed, detail = json.loads(line[len(_INSTALL_MARK) :])
                except ValueError:
                    detail = line[len(_INSTALL_MARK) :]
                plain = " ".join(_ANSI.sub("", str(detail)).split())
                warnings.append(f"could not install {plain[:280]}")
            elif line.startswith(_VERSIONS_MARK):
                try:
                    versions = json.loads(line[len(_VERSIONS_MARK) :])
                except ValueError:
                    versions = {}
                off = [
                    f"{req} (found {found or 'none'})"
                    for req, found in versions.items()
                    if found != req.rpartition("==")[2]
                ]
                if off:
                    exact = False
                    warnings.append(
                        "not at the docs' versions: " + ", ".join(off)[:240]
                    )
        return SandboxRun(stdout=stdout, warning="; ".join(warnings), exact=exact)
