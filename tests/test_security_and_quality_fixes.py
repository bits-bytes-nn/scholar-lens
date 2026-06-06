"""Tests for the P1 security + quality fixes (no real network).

Covers: the SSRF url_guard, HTMLTagOutputParser entity-unescape + trailing-tag
strip, the code_retriever trailing-slash repo-name fix, and the git token
redaction helper in the publisher.
"""

from __future__ import annotations

import pytest

from scholar_lens.src.code_retriever import CodeRetriever
from scholar_lens.src.publisher import Publisher
from scholar_lens.src.url_guard import (
    UnsafeUrlError,
    assert_url_is_public,
    is_url_public,
)
from scholar_lens.src.utils.parsers import HTMLTagOutputParser

# --------------------------------------------------------------------------- #
# SSRF url_guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1:8080/",  # loopback
        "http://localhost/x",  # loopback by name
        "http://10.0.0.5/internal",  # RFC1918
        "http://192.168.1.1/admin",  # RFC1918
        "http://[::1]/x",  # IPv6 loopback
        "ftp://example.com/x",  # disallowed scheme
        "file:///etc/passwd",  # disallowed scheme
        "http://0.0.0.0/",  # unspecified
    ],
)
def test_blocks_unsafe_urls(url: str) -> None:
    assert is_url_public(url) is False
    with pytest.raises(UnsafeUrlError):
        assert_url_is_public(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://1.1.1.1/x",  # public IP literal — no DNS needed
        "http://8.8.8.8/paper.pdf",
        "https://[2606:4700:4700::1111]/x",  # public IPv6 literal
    ],
)
def test_allows_public_ip_literals(url: str) -> None:
    # IP literals avoid DNS so the test stays hermetic under the network block.
    assert is_url_public(url) is True


def test_allows_public_hostname_via_mocked_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock DNS so we exercise the hostname path without real network.
    import scholar_lens.src.url_guard as guard

    monkeypatch.setattr(
        guard.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    assert is_url_public("https://example.com/paper.pdf") is True


def test_ipv4_mapped_ipv6_loopback_blocked() -> None:
    # ::ffff:127.0.0.1 must be unwrapped and rejected.
    assert is_url_public("http://[::ffff:127.0.0.1]/") is False


# --------------------------------------------------------------------------- #
# HTMLTagOutputParser
# --------------------------------------------------------------------------- #


def test_parser_unescapes_entities() -> None:
    p = HTMLTagOutputParser(tag_names="summary")
    assert p.parse("<summary>A &amp; B &gt; C &lt; D</summary>") == "A & B > C < D"


def test_parser_strips_trailing_orphan_tags() -> None:
    p = HTMLTagOutputParser(tag_names="summary")
    assert p.parse("<summary>real body</path></name></summary>") == "real body"


def test_parser_keeps_inline_markup_but_unescapes() -> None:
    p = HTMLTagOutputParser(tag_names="guide")
    # A markdown table header with an escaped ampersand must come back literal.
    out = p.parse("<guide>| Model &amp; Method | x |</guide>")
    assert out == "| Model & Method | x |"


def test_parser_dict_mode_multiple_tags() -> None:
    p = HTMLTagOutputParser(tag_names=["summary", "tags"])
    out = p.parse("<summary>body &amp; more</summary><tags>A, B</tags>")
    assert out == {"summary": "body & more", "tags": "A, B"}


# --------------------------------------------------------------------------- #
# code_retriever repo-name derivation (blocker fix)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/microsoft/LoRA", "LoRA"),
        ("https://github.com/microsoft/LoRA/", "LoRA"),  # trailing slash
        ("https://github.com/org/repo.git", "repo"),
        ("https://github.com/org/repo.git/", "repo"),
    ],
)
def test_repo_dir_name(url: str, expected: str) -> None:
    assert CodeRetriever._repo_dir_name(url) == expected


@pytest.mark.parametrize(
    "url",
    ["https://github.com/", "https://github.com", "/", ""],
)
def test_repo_dir_name_never_empty(url: str) -> None:
    # The blocker invariant: never collapse onto the cache root (which a later
    # rmtree would wipe). Any non-empty name is acceptable.
    assert CodeRetriever._repo_dir_name(url) != ""


# --------------------------------------------------------------------------- #
# publisher git token redaction
# --------------------------------------------------------------------------- #


def test_redact_token_scrubs_secret() -> None:
    err = RuntimeError("clone failed for https://oauth2:ghp_SECRET123@github.com/x")
    scrubbed = Publisher._redact_token(err, "ghp_SECRET123")
    assert "ghp_SECRET123" not in str(scrubbed)
    assert "***" in str(scrubbed)


def test_redact_token_noop_when_no_token() -> None:
    err = RuntimeError("some error")
    assert str(Publisher._redact_token(err, None)) == "some error"
