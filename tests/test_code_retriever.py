"""Tests for CodeRetriever's clone-URL safety guard (no network)."""

from __future__ import annotations

import pytest

from scholar_lens.src.code_retriever import CodeRetriever
from scholar_lens.src.url_guard import UnsafeUrlError


class TestAssertClonable:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ssh://git@github.com/org/repo.git",
            "git://github.com/org/repo.git",
            "http://github.com/org/repo.git",  # non-https downgrade
            "http://169.254.169.254/repo.git",  # internal literal
        ],
    )
    def test_rejects_unsafe_urls(self, url: str) -> None:
        with pytest.raises(UnsafeUrlError):
            CodeRetriever._assert_clonable(url)

    def test_accepts_public_https_forge_url(self) -> None:
        # A normal public https forge URL passes (github.com resolves publicly).
        CodeRetriever._assert_clonable("https://github.com/org/repo.git")
