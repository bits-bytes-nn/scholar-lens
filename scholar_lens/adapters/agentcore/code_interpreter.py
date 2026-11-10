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
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# A session lives a little longer than the work it is started for.
_SESSION_MARGIN_S = 120
_MAX_SESSION_S = 28_800


def driver_code(program: str, requirements: Sequence[str], timeout_s: int) -> str:
    """The notebook cell that installs ``requirements``, runs ``program`` in
    a fresh interpreter (also when the install failed: the blocks before
    the first import of what is missing still run) and prints the install
    failure and the program's outcome lines only."""
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
        if self.interpreter_id == SYSTEM_INTERPRETER and requirements:
            logger.info(
                "The system interpreter has no network; running without "
                "installing %s.",
                ", ".join(requirements),
            )
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
            response = runner.invoke_code_interpreter(
                codeInterpreterIdentifier=self.interpreter_id,
                sessionId=session_id,
                name="executeCode",
                arguments={
                    "language": "python",
                    "code": driver_code(program, requirements, timeout_s),
                },
            )
            stdout, stderr, is_error = read_stream(response)
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
        warning = ""
        for line in stdout.splitlines():
            if line.startswith(_INSTALL_MARK):
                _failed, detail = json.loads(line[len(_INSTALL_MARK) :])
                plain = " ".join(_ANSI.sub("", str(detail)).split())
                warning = f"could not install {plain[:280]}"
        return SandboxRun(stdout=stdout, warning=warning)
