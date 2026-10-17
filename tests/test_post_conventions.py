"""Loading the blog's conventions and existing tags (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
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


class TestFetchBlogPosts:
    def test_reads_front_matter_and_skips_unreadable_posts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        posts = {
            "a": (
                '---\ntitle: "A"\ntags: ["GraphRAG", "Knowledge-Graph"]\n'
                "description: >-\n  Graph RAG는 전역 질문에 답합니다.\n"
                "lang: ko\ntranslation_id: a\n---\nbody"
            ),
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
        posts_read = post_conventions.fetch_blog_posts("o/r", "main", None)
        assert [p.path for p in posts_read] == [
            "_posts/2025-01-01-a.md",
            "_posts/2024/2024-01-01-b.md",
        ]
        first = posts_read[0]
        assert first.description == "Graph RAG는 전역 질문에 답합니다."
        assert (first.lang, first.translation_id) == ("ko", "a")
        assert posts_read[1].description is None
        assert list(post_conventions.tag_counts(posts_read).items()) == [
            ("GraphRAG", 2),
            ("Knowledge-Graph", 1),
        ]
        assert not any(url.endswith("/r") for url in requested)


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2026-01-05 03:00:00", datetime(2026, 1, 5, 3, tzinfo=UTC)),  # UTC
        ("2026-01-05 12:00:00 +0900", datetime(2026, 1, 5, 3, tzinfo=UTC)),
        ('"2026-01-05T12:00:00+09:00"', datetime(2026, 1, 5, 3, tzinfo=UTC)),
        ("2026-01-05", datetime(2026, 1, 5, tzinfo=UTC)),
        ("someday", None),
    ],
)
def test_a_post_date_and_categories_are_read(
    date: str, expected: datetime | None
) -> None:
    # What a regenerated post keeps: the instant Jekyll dates the post at
    # (the blog builds in UTC) and its categories.
    text = f'---\ndate: {date}\ncategories: ["Tech Guides", "Language-Models"]\n---\n'
    post = post_conventions._blog_post("_posts/2026-01-05-x.md", text)
    assert post.date == expected
    assert post.categories == ["Tech Guides", "Language-Models"]
    assert post.topic == "Language-Models"


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

    def test_without_fallback_an_unreachable_blog_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The re-read before publishing must not mistake the bundled copy
        # for a blog change.
        self._fetch(monkeypatch, requests.ConnectionError("down"))
        with pytest.raises(requests.ConnectionError):
            load_conventions(
                repo_name="o/r",
                branch="main",
                token=None,
                use_blog=True,
                fallback=False,
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


def test_fetch_blog_posts_fails_when_most_posts_are_unreadable(
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
        post_conventions.fetch_blog_posts("o/r", "feat/x", None)
    assert requested[0].endswith("/git/trees/feat%2Fx")
