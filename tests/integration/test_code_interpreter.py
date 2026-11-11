"""The AgentCore code runner: the driver it runs, the stream it reads and
the session it always stops (no AWS call: a fake client)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Any

import pytest

from scholar_lens.adapters.agentcore.code_interpreter import (
    SYSTEM_INTERPRETER,
    AgentCoreCodeRunner,
    driver_code,
    read_stream,
)
from scholar_lens.application.ports import SandboxUnavailable
from scholar_lens.domain.program_run import program_script, read_outcomes


def _run(source: str, timeout: int = 60) -> str:
    return subprocess.run(
        [sys.executable, "-I", "-c", source],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    ).stdout


def test_the_driver_runs_the_program_and_prints_its_outcomes_only() -> None:
    script = program_script(["print('x' * 50)", "z"], block_timeout_s=5)
    out = _run(driver_code(script, [], 30))
    assert "xxxx" not in out
    assert [(r.index, r.ok) for r in read_outcomes(out)] == [(1, True), (2, False)]


def _result(stdout: str, *, error: bool = False, stderr: str = "") -> dict[str, Any]:
    return {
        "result": {
            "structuredContent": {"stdout": stdout, "stderr": stderr},
            "isError": error,
        }
    }


class _Client:
    def __init__(self, events: list[dict[str, Any]] | Exception) -> None:
        self.events = events
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def start_code_interpreter_session(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("start", kw))
        return {"sessionId": "s1"}

    def invoke_code_interpreter(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("invoke", kw))
        if isinstance(self.events, Exception):
            raise self.events
        return {
            "stream": (
                self.events if hasattr(self.events, "__next__") else iter(self.events)
            )
        }

    def stop_code_interpreter_session(self, **kw: Any) -> None:
        self.calls.append(("stop", kw))


class _Session:
    def __init__(self, client: _Client) -> None:
        self._client = client

    def client(self, name: str, **_: Any) -> _Client:
        assert name == "bedrock-agentcore"
        return self._client


def _runner(client: _Client, interpreter: str = "custom-1") -> AgentCoreCodeRunner:
    return AgentCoreCodeRunner(_Session(client), interpreter)  # type: ignore[arg-type]


def test_the_runner_returns_the_output_and_stops_the_session() -> None:
    client = _Client(
        [_result('out\n@@SCHOLAR_LENS_INSTALL_FAILED@@[["pkg==1"], "pkg==1: no"]')]
    )
    run = asyncio.run(_runner(client).run("p", requirements=["pkg==1"], timeout_s=30))
    assert "out" in run.stdout and run.failure == ""
    assert "pkg==1" in run.warning
    assert [c for c, _ in client.calls] == ["start", "invoke", "stop"]
    assert len(client.calls[0][1]["clientToken"]) >= 33
    assert "pkg==1" in client.calls[1][1]["arguments"]["code"]


def test_the_system_interpreter_installs_nothing() -> None:
    client = _Client([_result("")])
    runner = _runner(client, SYSTEM_INTERPRETER)
    asyncio.run(runner.run("p", requirements=["pkg==1"], timeout_s=30))
    assert "pkg==1" not in client.calls[1][1]["arguments"]["code"]


@pytest.mark.parametrize(
    "events",
    [
        [{"throttlingException": {"message": "slow down"}}],
        [_result("", error=True, stderr="kernel died")],
    ],
    ids=["exception-event", "driver-error"],
)
def test_a_sandbox_failure_is_unavailable_and_the_session_stops(
    events: list[dict[str, Any]],
) -> None:
    client = _Client(events)
    with pytest.raises(SandboxUnavailable):
        asyncio.run(_runner(client).run("p", requirements=[], timeout_s=30))
    assert client.calls[-1][0] == "stop"


def test_read_stream_joins_the_chunks() -> None:
    out, err, is_error = read_stream({"stream": [_result("a"), _result("b")]})
    assert (out, err, is_error) == ("ab", "", False)


def test_a_dropped_stream_is_unavailable() -> None:
    from urllib3.exceptions import ProtocolError

    def stream():  # type: ignore[no-untyped-def]
        yield _result("a")
        raise ProtocolError("connection broken")

    client = _Client([])
    client.events = stream()  # type: ignore[assignment]
    with pytest.raises(SandboxUnavailable):
        asyncio.run(_runner(client).run("p", requirements=[], timeout_s=30))
    assert client.calls[-1][0] == "stop"


def test_a_forged_install_line_from_the_program_is_ignored() -> None:
    forged = "@@SCHOLAR_LENS_INSTALL_FAILED@@{@@SCHOLAR_LENS_RESULT@@"
    client = _Client([_result(f"{forged}\n")])
    run = asyncio.run(_runner(client).run("p", requirements=[], timeout_s=30))
    assert run.warning == ""
