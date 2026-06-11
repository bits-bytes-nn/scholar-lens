"""Tests for the Slack Events API receiver Lambda (no real AWS/Slack).

The receiver verifies the request signature, answers the URL-verification
handshake, drops Slack retries, and async-invokes the worker. All AWS access is
patched; signatures are computed with a known secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from unittest.mock import MagicMock

import pytest

from scholar_lens.slack import lambda_receiver

_SECRET = "test-signing-secret"


def _sign(body: str, *, ts: str = "1700000000", secret: str = _SECRET) -> str:
    basestring = f"v0:{ts}:{body}".encode()
    return "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()


def _event(
    body: str,
    *,
    ts: str = "1700000000",
    signature: str | None = None,
    extra_headers: dict[str, str] | None = None,
    is_base64: bool = False,
) -> dict:
    headers = {
        "x-slack-request-timestamp": ts,
        "x-slack-signature": signature if signature is not None else _sign(body, ts=ts),
    }
    if extra_headers:
        headers.update(extra_headers)
    return {"headers": headers, "body": body, "isBase64Encoded": is_base64}


@pytest.fixture(autouse=True)
def _patch_secret_and_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stable signing secret + clock so signatures verify deterministically.
    monkeypatch.setattr(lambda_receiver, "_signing_secret", _SECRET)
    monkeypatch.setattr(lambda_receiver, "_get_signing_secret", lambda: _SECRET)
    monkeypatch.setattr(lambda_receiver.time, "time", lambda: 1700000100.0)
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "worker-fn")


@pytest.fixture
def mock_lambda(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch boto3.client so _invoke_worker records the async invoke."""
    client = MagicMock()
    monkeypatch.setattr(lambda_receiver.boto3, "client", lambda service: client)
    return client


class TestSignatureVerification:
    def test_valid_signature_accepts_and_invokes(self, mock_lambda: MagicMock) -> None:
        body = json.dumps({"event": {"type": "app_mention", "text": "hi"}})
        resp = lambda_receiver.handler(_event(body), None)
        assert resp["statusCode"] == 200
        mock_lambda.invoke.assert_called_once()
        kwargs = mock_lambda.invoke.call_args.kwargs
        assert kwargs["FunctionName"] == "worker-fn"
        assert kwargs["InvocationType"] == "Event"
        payload = json.loads(kwargs["Payload"])
        assert payload["slack_event"]["type"] == "app_mention"

    def test_invalid_signature_rejected(self, mock_lambda: MagicMock) -> None:
        body = json.dumps({"event": {"type": "app_mention"}})
        resp = lambda_receiver.handler(_event(body, signature="v0=deadbeef"), None)
        assert resp["statusCode"] == 401
        mock_lambda.invoke.assert_not_called()

    def test_stale_timestamp_rejected(self, mock_lambda: MagicMock) -> None:
        body = json.dumps({"event": {"type": "app_mention"}})
        # ts is ~ now-100s in the fixture; push it well beyond the 5-min window.
        resp = lambda_receiver.handler(_event(body, ts="1699990000"), None)
        assert resp["statusCode"] == 401
        mock_lambda.invoke.assert_not_called()

    def test_missing_headers_rejected(self, mock_lambda: MagicMock) -> None:
        resp = lambda_receiver.handler({"headers": {}, "body": "{}"}, None)
        assert resp["statusCode"] == 401
        mock_lambda.invoke.assert_not_called()


class TestUrlVerification:
    def test_challenge_echoed(self, mock_lambda: MagicMock) -> None:
        body = json.dumps({"type": "url_verification", "challenge": "abc123"})
        resp = lambda_receiver.handler(_event(body), None)
        assert resp["statusCode"] == 200
        assert resp["body"] == "abc123"
        mock_lambda.invoke.assert_not_called()


class TestRetrySuppression:
    def test_retry_header_suppressed(self, mock_lambda: MagicMock) -> None:
        body = json.dumps({"event": {"type": "app_mention"}})
        resp = lambda_receiver.handler(
            _event(body, extra_headers={"x-slack-retry-num": "1"}), None
        )
        assert resp["statusCode"] == 200
        mock_lambda.invoke.assert_not_called()


class TestBase64Body:
    def test_base64_body_decoded_and_verified(self, mock_lambda: MagicMock) -> None:
        raw = json.dumps({"event": {"type": "app_mention", "text": "x"}})
        encoded = base64.b64encode(raw.encode()).decode()
        # Signature must be over the RAW (decoded) body, not the base64 form.
        resp = lambda_receiver.handler(
            _event(encoded, signature=_sign(raw), is_base64=True), None
        )
        assert resp["statusCode"] == 200
        mock_lambda.invoke.assert_called_once()


class TestEventFiltering:
    def test_bot_message_ignored(self, mock_lambda: MagicMock) -> None:
        body = json.dumps(
            {"event": {"type": "message", "channel_type": "im", "bot_id": "B1"}}
        )
        resp = lambda_receiver.handler(_event(body), None)
        assert resp["statusCode"] == 200
        mock_lambda.invoke.assert_not_called()

    def test_im_message_invokes(self, mock_lambda: MagicMock) -> None:
        body = json.dumps(
            {"event": {"type": "message", "channel_type": "im", "text": "review x"}}
        )
        lambda_receiver.handler(_event(body), None)
        mock_lambda.invoke.assert_called_once()

    def test_channel_message_ignored(self, mock_lambda: MagicMock) -> None:
        # A plain channel message (not a mention, not a DM) is not actionable.
        body = json.dumps({"event": {"type": "message", "channel_type": "channel"}})
        resp = lambda_receiver.handler(_event(body), None)
        assert resp["statusCode"] == 200
        mock_lambda.invoke.assert_not_called()
