"""Loading the blog's conventions and existing tags (no network)."""

from __future__ import annotations

from typing import Any

import pytest
import requests
import yaml

from scholar_lens.src import post_conventions
from scholar_lens.src.post_conventions import load_conventions, load_local_conventions


class _Response:
    def __init__(self, payload: Any = None, text: str = "", status: int = 200) -> None:
        self._payload, self.text, self.status_code = payload, text, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self) -> Any:
        return self._payload


class TestFetchBlogTags:
    def test_counts_tags_across_posts_and_skips_unreadable_ones(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        posts = {
            "a": '---\ntitle: "A"\ntags: ["GraphRAG", "Knowledge-Graph"]\n---\nbody',
            "b": '---\ntitle: "B"\ntags: ["GraphRAG"]\n---\nbody',
        }
        tree = [
            {"type": "blob", "path": "_posts/2025-01-01-a.md", "sha": "a"},
            {"type": "blob", "path": "_posts/2024/2024-01-01-b.md", "sha": "b"},
            {"type": "blob", "path": "_posts/2024-01-02-gone.md", "sha": "gone"},
            {"type": "blob", "path": "README.md", "sha": "r"},
            {"type": "tree", "path": "_posts/2024", "sha": "t"},
        ]
        requested: list[str] = []

        def fake_get(url: str, **kwargs: Any) -> _Response:
            requested.append(url)
            if "/git/trees/" in url:
                assert kwargs["params"] == {"recursive": "1"}
                return _Response({"tree": tree, "truncated": False})
            sha = url.rsplit("/", 1)[1]
            return _Response(text=posts[sha]) if sha in posts else _Response(status=500)

        monkeypatch.setattr(post_conventions.requests, "get", fake_get)
        tags = post_conventions.fetch_blog_tags("o/r", "main", None)
        assert list(tags.items()) == [("GraphRAG", 2), ("Knowledge-Graph", 1)]
        assert not any(url.endswith("/r") for url in requested)


class TestLoadConventions:
    @pytest.fixture
    def local(self) -> Any:
        return load_local_conventions()

    def _fetch(self, monkeypatch: pytest.MonkeyPatch, outcome: Any) -> None:
        def fetch(repo_name: str, branch: str, token: str | None) -> str | None:
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(post_conventions, "fetch_blog_conventions", fetch)

    def test_blog_unused_returns_the_bundled_copy(self, local: Any) -> None:
        assert (
            load_conventions(repo_name="o/r", branch="main", token=None, use_blog=False)
            == local
        )

    @pytest.mark.parametrize(
        "outcome", [requests.ConnectionError("down"), None], ids=["network", "404"]
    )
    def test_unreachable_blog_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, local: Any, outcome: Any
    ) -> None:
        self._fetch(monkeypatch, outcome)
        assert (
            load_conventions(repo_name="o/r", branch="main", token=None, use_blog=True)
            == local
        )

    def test_blog_rules_win_when_they_differ(
        self, monkeypatch: pytest.MonkeyPatch, local: Any
    ) -> None:
        text = post_conventions.LOCAL_CONVENTIONS_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        retired = data["tags"]["controlled"].pop()
        self._fetch(monkeypatch, yaml.safe_dump(data))
        blog = load_conventions(
            repo_name="o/r", branch="main", token=None, use_blog=True
        )
        assert blog != local and retired not in blog.tags.controlled


def test_fetch_blog_tags_fails_when_most_posts_are_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = [
        {"type": "blob", "path": f"_posts/2025-01-0{i}-p.md", "sha": str(i)}
        for i in range(1, 4)
    ]
    requested: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> _Response:
        requested.append(url)
        if "/git/trees/" in url:
            return _Response({"tree": tree})
        if url.endswith("/1"):
            return _Response(text='---\ntags: ["A"]\n---\n')
        return _Response(status=403)

    monkeypatch.setattr(post_conventions.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="2 of 3"):
        post_conventions.fetch_blog_tags("o/r", "feat/x", None)
    assert requested[0].endswith("/git/trees/feat%2Fx")
