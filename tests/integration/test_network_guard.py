"""The suite's network guard (tests/conftest.py) catches a leak that the code
under test swallows, through every HTTP client the code uses."""

from __future__ import annotations

import urllib.request

import boto3
import httpx
import pytest
import requests


def _swallowed(call) -> None:  # type: ignore[no-untyped-def]
    try:
        call()
    except Exception:  # noqa: BLE001 - the code under test would degrade here
        pass


@pytest.mark.parametrize(
    "client",
    ["requests", "httpx", "urllib", "boto3"],
)
def test_an_unmocked_call_is_recorded_even_when_swallowed(
    client: str, request: pytest.FixtureRequest
) -> None:
    calls = {
        "requests": lambda: requests.get("http://example.com/x", timeout=2),
        "httpx": lambda: httpx.get("http://example.com/x", timeout=2),
        "urllib": lambda: urllib.request.urlopen("http://example.com/x", timeout=2),
        "boto3": lambda: boto3.client(
            "sts", region_name="us-east-1", endpoint_url="http://example.com"
        ).get_caller_identity(),
    }
    _swallowed(calls[client])
    blocked = request.node.blocked_network
    assert blocked, f"{client} reached no guard: a leak would pass silently"
    blocked.clear()  # this test's leak is deliberate


@pytest.mark.parametrize("address", [("93.184.216.34", 80), ("127.0.0.1", 9)])
def test_connect_ex_to_the_network_or_the_fence_is_blocked(
    address: tuple[str, int], request: pytest.FixtureRequest
) -> None:
    import socket

    with socket.socket() as sock, pytest.raises(RuntimeError, match="Blocked"):
        sock.connect_ex(address)
    blocked = request.node.blocked_network
    assert blocked == [address]
    blocked.clear()


def test_connect_ex_to_a_closed_local_port_returns_its_errno() -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0
