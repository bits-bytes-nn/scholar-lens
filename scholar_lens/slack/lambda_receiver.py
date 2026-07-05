"""Slack Events API receiver (AWS Lambda behind an HTTP API Gateway).

This is the fast inbound edge of Paper Bot. Slack requires an HTTP 200 within
three seconds; intent parsing is a Bedrock LLM call that can exceed that on a
cold start. So this receiver does only cheap, bounded work — verify the request
signature, answer the URL-verification handshake, drop Slack retries, and
async-invoke the heavy worker Lambda — then returns 200 immediately. The worker
(``lambda_worker.py``) does the intent parse + Batch dispatch off the 3s clock.

It sits behind an API Gateway HTTP API (payload format v2.0), not a Lambda
Function URL: this account's SCP guardrails block anonymous Function-URL invokes.
The gateway leaves the route unauthorized and security is the HMAC check below.

IMPORTANT: this module is packaged as a standalone Lambda zip and must NOT import
the ``scholar_lens`` package — importing it would drag in langchain/boto-heavy
modules and blow the cold-start budget on the ack path. It therefore uses only
the standard library plus ``boto3`` (provided by the Lambda runtime), and
duplicates the handful of string constants it needs. Their source of truth is
``scholar_lens/src/constants.py`` (kept in sync by hand):

    EnvVars.WORKER_FUNCTION_NAME      -> "WORKER_FUNCTION_NAME"
    SSMParams.SLACK_SIGNING_SECRET    -> "slack-signing-secret"
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any

import boto3

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# Mirror of constants.py (see module docstring) — duplicated to keep this zip
# free of the scholar_lens package import.
_WORKER_FUNCTION_NAME_ENV = "WORKER_FUNCTION_NAME"
_SIGNING_SECRET_SSM_SUFFIX = "slack-signing-secret"
_PROJECT_ENV = "PROJECT"
_STAGE_ENV = "STAGE"

# Reject requests whose timestamp is older than this (seconds) to thwart replay.
_MAX_REQUEST_AGE_SECONDS = 60 * 5

# Cache the signing secret across warm invocations (fetched once from SSM).
_signing_secret: str | None = None

_OK = {"statusCode": 200, "body": ""}


def _get_signing_secret() -> str:
    """Fetch and cache the Slack signing secret from SSM (SecureString)."""
    global _signing_secret
    if _signing_secret is None:
        project = os.environ[_PROJECT_ENV]
        stage = os.environ[_STAGE_ENV]
        name = f"/{project}/{stage}/{_SIGNING_SECRET_SSM_SUFFIX}"
        ssm = boto3.client("ssm")
        resp = ssm.get_parameter(Name=name, WithDecryption=True)
        _signing_secret = resp["Parameter"]["Value"]
    return _signing_secret


def _lower_headers(headers: dict[str, str] | None) -> dict[str, str]:
    """Slack header casing varies; normalise to lowercase keys."""
    return {k.lower(): v for k, v in (headers or {}).items()}


def _raw_body(event: dict[str, Any]) -> str:
    """Return the exact raw request body string.

    The HMAC must be computed over the bytes Slack sent, so we base64-decode when
    the gateway flags the body as encoded and never re-serialize the JSON.
    """
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        return base64.b64decode(body).decode("utf-8")
    return body


def verify_signature(raw_body: str, headers: dict[str, str], secret: str) -> bool:
    """Verify Slack's v0 request signature (constant-time).

    basestring = ``v0:{timestamp}:{raw_body}``; the signature is
    ``v0=`` + HMAC-SHA256(secret, basestring). Rejects stale timestamps to guard
    against replay.
    """
    timestamp = headers.get("x-slack-request-timestamp", "")
    received = headers.get("x-slack-signature", "")
    if not timestamp or not received:
        return False
    try:
        if abs(time.time() - int(timestamp)) > _MAX_REQUEST_AGE_SECONDS:
            logger.warning("Rejecting Slack request with stale timestamp.")
            return False
    except ValueError:
        return False
    basestring = f"v0:{timestamp}:{raw_body}".encode()
    computed = "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, received)


def _is_actionable_event(inner: dict[str, Any]) -> bool:
    """Match the same events the Socket Mode bot handled: app mentions and DMs.

    DMs arrive as ``message`` events with ``channel_type == "im"``; we ignore the
    bot's own messages (``bot_id`` set) to avoid loops.
    """
    event_type = inner.get("type")
    if event_type == "app_mention":
        return True
    if event_type == "message":
        return inner.get("channel_type") == "im" and not inner.get("bot_id")
    return False


def _invoke_worker(inner_event: dict[str, Any]) -> None:
    """Async-invoke the worker Lambda with the inner Slack event."""
    function_name = os.environ[_WORKER_FUNCTION_NAME_ENV]
    boto3.client("lambda").invoke(
        FunctionName=function_name,
        InvocationType="Event",  # async; receiver returns 200 without waiting
        Payload=json.dumps({"slack_event": inner_event}).encode("utf-8"),
    )


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """API Gateway (HTTP API v2.0) handler for Slack Events API requests."""
    headers = _lower_headers(event.get("headers"))
    raw_body = _raw_body(event)

    # Secret fetch can fail transiently (SSM throttle / IAM drift / cold start).
    # We haven't invoked the worker yet, so returning 500 (letting Slack retry) is
    # safe — no double-dispatch — and far better than raising an unhandled 502 that
    # gets silently dropped once the secret later caches.
    try:
        signing_secret = _get_signing_secret()
    except Exception:
        logger.exception("Could not load the Slack signing secret from SSM.")
        return {"statusCode": 500, "body": "configuration error"}

    if not verify_signature(raw_body, headers, signing_secret):
        # Metadata-only diagnostics (never the body or secret): a mismatch almost
        # always means the raw body we hash differs from what Slack signed
        # (encoding, header casing) or the secret in SSM is stale.
        logger.warning(
            "Slack signature verification failed. "
            "is_b64=%s body_len=%d has_ts=%s has_sig=%s content_type=%r",
            event.get("isBase64Encoded"),
            len(raw_body),
            bool(headers.get("x-slack-request-timestamp")),
            bool(headers.get("x-slack-signature")),
            headers.get("content-type"),
        )
        return {"statusCode": 401, "body": "invalid signature"}

    try:
        payload = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError:
        logger.warning("Could not parse Slack request body as JSON.")
        return {"statusCode": 400, "body": "bad request"}

    # URL-verification handshake (run after the signature check — Slack signs it).
    if payload.get("type") == "url_verification":
        return {
            "statusCode": 200,
            "headers": {"content-type": "text/plain"},
            "body": payload.get("challenge", ""),
        }

    # Drop Slack retries: the receiver already 200s in <1s, so a retry only means
    # the worker leg failed downstream — re-dispatching would double-submit.
    if "x-slack-retry-num" in headers:
        logger.info(
            "Ignoring Slack retry (num=%s, reason=%s).",
            headers.get("x-slack-retry-num"),
            headers.get("x-slack-retry-reason"),
        )
        return _OK

    inner = payload.get("event") or {}
    if not _is_actionable_event(inner):
        logger.info("Ignoring non-actionable Slack event '%s'.", inner.get("type"))
        return _OK

    try:
        _invoke_worker(inner)
    except Exception:
        # The async invoke itself failed (throttle/permission). Return 500 so Slack
        # retries rather than silently losing the request; since the worker was
        # never invoked, the retry is the first real dispatch, not a duplicate.
        logger.exception("Failed to invoke the worker Lambda.")
        return {"statusCode": 500, "body": "dispatch error"}
    logger.info("Dispatched Slack event '%s' to worker.", inner.get("type"))
    return _OK
