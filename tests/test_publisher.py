"""Tests for the artifact-agnostic Publisher (S3 mocked, git stubbed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from scholar_lens.configs import Github
from scholar_lens.src.aws_helpers import S3Handler
from scholar_lens.src.publisher import Publisher, PublishRequest, slugify

BUCKET = "scholar-lens-test"


class TestSlugify:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Getting Started with FastAPI", "getting-started-with-fastapi"),
            ("Foo/Bar: Guide?", "foo-bar-guide"),
            ("///", "untitled"),
        ],
    )
    def test_slugify(self, title: str, expected: str) -> None:
        assert slugify(title) == expected


def _request(work_dir: Path, **overrides) -> PublishRequest:
    base = dict(
        title="My Guide",
        markdown="# Title\n\n![fig](figures/diagram.png)\n",
        work_dir=work_dir,
        branch_id="my-guide",
        pr_title="Tech Guide: My Guide",
        pr_body="body",
        commit_message="feat: add guide",
    )
    base.update(overrides)
    return PublishRequest(**base)  # type: ignore[arg-type]


class TestPublishLocal:
    async def test_publish_without_s3_writes_local_file(self, tmp_path: Path) -> None:
        pub = Publisher(Github(), root_dir=tmp_path)
        request = _request(tmp_path, rewrite_local_images=False)
        s3_url, md_path = await pub.publish(request)
        assert s3_url is None
        assert md_path.exists()
        assert md_path.read_text(encoding="utf-8").startswith("# Title")

    async def test_local_images_rewritten_to_assets_path(self, tmp_path: Path) -> None:
        pub = Publisher(Github(), root_dir=tmp_path)
        request = _request(tmp_path, rewrite_local_images=True)
        _, md_path = await pub.publish(request)
        content = md_path.read_text(encoding="utf-8")
        # Local "figures/diagram.png" becomes "/assets/<file>/diagram.png".
        assert "/assets/" in content and "diagram.png" in content
        assert "figures/diagram.png" not in content

    async def test_http_images_untouched(self, tmp_path: Path) -> None:
        pub = Publisher(Github(), root_dir=tmp_path)
        request = _request(
            tmp_path,
            markdown="![x](https://cdn.example.com/a.png)",
            rewrite_local_images=True,
        )
        _, md_path = await pub.publish(request)
        assert "https://cdn.example.com/a.png" in md_path.read_text(encoding="utf-8")


class TestPublishS3:
    async def test_publish_uploads_post_and_returns_s3_url(
        self, tmp_path: Path
    ) -> None:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            handler = S3Handler(boto3.Session(region_name="us-east-1"), BUCKET)

            pub = Publisher(
                Github(),
                root_dir=tmp_path,
                s3_handler=handler,
                s3_bucket_name=BUCKET,
                s3_prefix="scholar-lens",
            )
            request = _request(tmp_path, rewrite_local_images=False)
            s3_url, _ = await pub.publish(request)
            assert s3_url is not None
            assert s3_url.startswith(f"s3://{BUCKET}/scholar-lens/posts/")
            # The post object exists in the bucket.
            keys = [
                o["Key"]
                for o in client.list_objects_v2(Bucket=BUCKET).get("Contents", [])
            ]
            assert any(k.endswith(".md") for k in keys)


class TestCreatePullRequest:
    async def test_skips_when_repo_not_configured(self, tmp_path: Path) -> None:
        pub = Publisher(Github(repo_name=None), root_dir=tmp_path)
        request = _request(tmp_path)
        # Should return without raising (no repo configured).
        await pub.create_pull_request(request, tmp_path / "x.md")

    async def test_skips_when_token_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        pub = Publisher(Github(repo_name="owner/blog"), root_dir=tmp_path)
        request = _request(tmp_path)
        await pub.create_pull_request(request, tmp_path / "x.md")  # no raise

    async def test_git_and_pr_invoked_when_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
        pub = Publisher(Github(repo_name="owner/blog"), root_dir=tmp_path)

        git_calls = MagicMock()
        monkeypatch.setattr(pub, "_git_operations", git_calls)

        created = {}

        class FakeRepo:
            def create_pull(self, **kwargs):  # type: ignore[no-untyped-def]
                created.update(kwargs)

        class FakeGithub:
            def __init__(self, *a, **k) -> None:
                pass

            def get_repo(self, name):  # type: ignore[no-untyped-def]
                return FakeRepo()

        monkeypatch.setattr("scholar_lens.src.publisher.Github", FakeGithub)
        request = _request(tmp_path)
        await pub.create_pull_request(request, tmp_path / "x.md")

        git_calls.assert_called_once()
        assert created["title"] == "Tech Guide: My Guide"
        assert created["base"] == "main"
