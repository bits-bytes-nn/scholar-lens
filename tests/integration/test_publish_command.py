"""``scholar-lens publish`` (docs/engine.md, 4.9): every run saves its
publish request next to the saved or rejected post; ``publish`` checks the
(edited) post again and opens its PR on the same branch, with a run's exit
codes. The request JSON fails loudly on unknown fields and defaults the
missing optional ones."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from scholar_lens import cli
from scholar_lens.adapters.aws.storage import S3Handler
from scholar_lens.adapters.files import LocalOrS3Files
from scholar_lens.adapters.github.publisher import ImageLocalizer, Publisher
from scholar_lens.application.run import publish_artifact
from scholar_lens.application.use_cases import publish
from scholar_lens.configs import Github
from scholar_lens.domain.publishing import (
    PublishRequest,
    SavedPublishRequest,
    request_name,
    without_violations,
)
from scholar_lens.domain.validation import PostValidationError
from scholar_lens.entrypoints import runtime as saved_post
from tests.integration.test_publisher import PNG, VALID_POST, FakeGithub, FakeRepo

BROKEN = VALID_POST.replace('paper_author: "Microsoft"', 'author: "Edward Hu"')
_CONFIG = Github(
    enabled=True,
    repo_name="owner/blog",
    branch_prefix="paper-reviews",
    author_name="Bot",
    author_email="bot@example.com",
)
BUCKET = "scholar-lens-test"


def _request(work_dir: Path, markdown: str = BROKEN) -> PublishRequest:
    return PublishRequest(
        title="My Guide",
        markdown=markdown,
        work_dir=work_dir,
        branch_id="my-guide",
        pr_title="Tech Guide: My Guide",
        pr_body="body",
        commit_message="feat: add guide",
        review_notes=["a reviewer note"],
        blocking=["the claim check did not run"],
    )


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> FakeRepo:
    repo = FakeRepo({"README.md": b"blog"})
    monkeypatch.setattr(FakeGithub, "repo", repo)
    monkeypatch.setattr("scholar_lens.adapters.github.publisher.Github", FakeGithub)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    return repo


def _context(tmp_path: Path, s3_handler: Any = None, prefix: str = "") -> Any:
    context = MagicMock()
    context.config.resources.github = _CONFIG
    context.config.resources.s3_prefix = prefix
    context.s3_handler = s3_handler
    context.files = LocalOrS3Files(s3_handler)
    context.papers_dir = tmp_path / "papers"
    context.conventions = None
    context.publisher_factory = lambda ctx: Publisher(
        _CONFIG,
        root_dir=tmp_path,
        s3_handler=s3_handler,
        s3_bucket_name=BUCKET if s3_handler else None,
        s3_prefix=prefix,
    )
    return context


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    context = _context(tmp_path)
    monkeypatch.setattr(saved_post.Config, "load", classmethod(lambda cls: None))
    monkeypatch.setattr(saved_post, "build_context", lambda config: context)
    monkeypatch.setattr(saved_post, "load_secrets_from_ssm", lambda *a: None)
    return context


def _rejected_run(context: Any, work_dir: Path) -> Path:
    """A run whose post the blog rules rejected: its rejected copy."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PostValidationError) as info:
        asyncio.run(publish_artifact(context, _request(work_dir), create_pr=True))
    assert info.value.rejected_path is not None
    return Path(info.value.rejected_path)


# ---- the round trip -------------------------------------------------------------


def test_a_rejected_post_edited_is_published_on_the_runs_branch(
    context: Any, repo: FakeRepo, tmp_path: Path
) -> None:
    rejected = _rejected_run(context, tmp_path / "guide-my-guide")
    saved = rejected.with_name(request_name(rejected.name))
    assert saved.name == "my-guide.rejected.request.json"
    request = SavedPublishRequest.model_validate_json(saved.read_text())
    assert request.branch_id == "my-guide" and request.post_name
    assert request.blocking == ["the claim check did not run"]

    # Unedited, the post still breaks the rule: failed, exit 1.
    assert cli.main(["publish", str(rejected)]) == cli.EXIT_FAILED
    assert not repo.pulls

    rejected.write_text(
        rejected.read_text().replace('author: "Edward Hu"', 'paper_author: "Microsoft"')
    )
    assert cli.main(["publish", str(rejected)]) == cli.EXIT_OK
    (pull,) = repo.pulls
    assert pull.head == "paper-reviews/my-guide"
    # The run's blocking problems and notes reach the PR as the run's would.
    assert "the claim check did not run" in pull.body
    assert "a reviewer note" in pull.body
    posts = [p for p in repo.files_at("heads/paper-reviews/my-guide") if "_posts/" in p]
    assert posts == [f"_posts/{request.post_name}.md"]
    published = repo.files_at("heads/paper-reviews/my-guide")[posts[0]].decode()
    assert "violations" not in published and 'paper_author: "Microsoft"' in published

    # The saved post, published again, updates the same PR.
    saved_post_path = rejected.parent / f"{request.post_name}.md"
    assert saved_post_path.with_name(request_name(saved_post_path.name)).is_file()
    assert cli.main(["publish", str(saved_post_path)]) == cli.EXIT_OK
    assert len(repo.pulls) == 1


def test_a_no_pr_run_is_published_later(
    context: Any, repo: FakeRepo, tmp_path: Path
) -> None:
    work_dir = tmp_path / "guide-my-guide"
    work_dir.mkdir()
    outcome = asyncio.run(
        publish_artifact(context, _request(work_dir, VALID_POST), create_pr=False)
    )
    assert outcome.markdown_path is not None and not repo.pulls
    assert cli.main(["publish", str(outcome.markdown_path), "--no-pr"]) == 0
    assert not repo.pulls
    assert cli.main(["publish", str(outcome.markdown_path)]) == cli.EXIT_OK
    assert len(repo.pulls) == 1


# ---- exit codes -----------------------------------------------------------------


def test_publish_exit_codes(
    context: Any,
    repo: FakeRepo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No such post, or a post no run saved: rejected input.
    assert cli.main(["publish", str(tmp_path / "missing.md")]) == cli.EXIT_REJECTED
    capsys.readouterr()
    stray = tmp_path / "stray.md"
    stray.write_text(VALID_POST)
    assert cli.main(["publish", str(stray)]) == cli.EXIT_REJECTED
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "rejected"
    assert "No publish request next to" in result["error"]

    work_dir = tmp_path / "guide-my-guide"
    work_dir.mkdir()
    outcome = asyncio.run(
        publish_artifact(context, _request(work_dir, VALID_POST), create_pr=False)
    )

    class DownGithub(FakeGithub):
        def get_repo(self, name: str) -> Any:
            raise RuntimeError("GitHub is down")

    monkeypatch.setattr("scholar_lens.adapters.github.publisher.Github", DownGithub)
    capsys.readouterr()
    assert cli.main(["publish", str(outcome.markdown_path)]) == cli.EXIT_PR_FAILED
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed" and "GitHub is down" in result["pr_error"]
    assert result["markdown_path"] == str(outcome.markdown_path)
    assert result["kind"] == "review"

    # No token for a wanted PR: failed before anything is written.
    monkeypatch.delenv("GITHUB_TOKEN")
    assert cli.main(["publish", str(outcome.markdown_path)]) == cli.EXIT_FAILED


# ---- the request JSON -------------------------------------------------------------


def _write_request(path: Path, data: dict[str, Any]) -> Path:
    post = path / "2026-11-08-my-guide.md"
    post.write_text(VALID_POST)
    post.with_name(request_name(post.name)).write_text(json.dumps(data))
    return post


_REQUIRED = {
    "title": "My Guide",
    "work_dir": "/tmp/papers/guide-my-guide",
    "branch_id": "my-guide",
    "pr_title": "Tech Guide: My Guide",
    "pr_body": "body",
    "commit_message": "feat: add guide",
}


def test_an_unknown_request_field_fails_loudly(context: Any, tmp_path: Path) -> None:
    post = _write_request(tmp_path, {**_REQUIRED, "surprise": 1})
    with pytest.raises(publish.SavedPostError, match="not one this version"):
        publish.load_saved_post(context, str(post))
    result = saved_post.run_publish(str(post), create_pr=False)
    assert result.status == "rejected" and "surprise" in (result.error or "")
    # A request of a later format version is refused too.
    with pytest.raises(publish.SavedPostError):
        publish.load_saved_post(
            context, str(_write_request(tmp_path, {**_REQUIRED, "version": 2}))
        )


def test_missing_optional_request_fields_take_their_defaults(
    context: Any, tmp_path: Path
) -> None:
    post = _write_request(tmp_path, _REQUIRED)
    request = publish.load_saved_post(context, str(post))
    assert request.kind == "review" and request.blocking == []
    assert request.review_notes == [] and request.published_path is None
    assert request.post_name is None
    # A local post's files are where the post is, not the run's work dir.
    assert request.work_dir == tmp_path.resolve()
    assert request.markdown == VALID_POST


def test_a_saved_request_round_trips(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request.published_path = "_posts/2026-01-01-my-guide.md"
    request.post_name = "2026-01-01-my-guide"
    saved = SavedPublishRequest.model_validate_json(
        SavedPublishRequest.of(request).model_dump_json()
    )
    back = saved.request(request.markdown, tmp_path)
    assert back == request
    assert "markdown" not in SavedPublishRequest.model_fields


def test_the_violations_comment_is_removed() -> None:
    kept = "## A\n\ntext <!-- a comment -->\n"
    assert without_violations(kept) == kept
    assert (
        without_violations("## A\n\ntext\n\n<!-- violations\n- x\n-->\n")
        == "## A\n\ntext\n"
    )


# ---- a Batch run's post, from S3 ----------------------------------------------------


def test_a_rejected_post_in_s3_brings_its_images(tmp_path: Path) -> None:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        handler = S3Handler(boto3.Session(region_name="us-east-1"), BUCKET)
        context = _context(tmp_path, handler, prefix="scholar-lens")
        markdown = VALID_POST + "\n![그림](/assets/images/my-guide-1.png)\n"
        request = _request(Path("/tmp/papers/guide-my-guide"), markdown)
        request.post_name = "2026-11-08-my-guide"
        handler.put_text(
            "scholar-lens/rejected/my-guide.rejected.md",
            markdown + "\n\n<!-- violations\n- x\n-->\n",
        )
        handler.put_text(
            "scholar-lens/rejected/my-guide.rejected.request.json",
            SavedPublishRequest.of(request).model_dump_json(),
        )
        handler._s3_client.put_object(
            Bucket=BUCKET,
            Key="scholar-lens/rejected/blog-images/my-guide/my-guide-1.png",
            Body=PNG,
        )
        loaded = publish.load_saved_post(
            context, f"s3://{BUCKET}/scholar-lens/rejected/my-guide.rejected.md"
        )
        with pytest.raises(publish.SavedPostError, match="not under"):
            publish.load_saved_post(context, "s3://other/scholar-lens/x.md")
    assert loaded.work_dir == tmp_path / "papers" / "guide-my-guide"
    assert loaded.markdown == markdown
    assert (loaded.staged_images_dir / "my-guide-1.png").read_bytes() == PNG
    assert loaded.post_name == "2026-11-08-my-guide"


def test_a_republished_post_keeps_the_images_it_shows(tmp_path: Path) -> None:
    staging = tmp_path / "blog-images" / "my-guide"
    staging.mkdir(parents=True)
    (staging / "my-guide-1.png").write_bytes(PNG)
    (staging / "my-guide-2.png").write_bytes(PNG)  # no longer shown
    localizer = ImageLocalizer(
        slug="my-guide",
        figures_dir=tmp_path / "figures",
        staging_dir=staging,
        fetch=lambda url: PNG,
    )
    out = localizer.localize(
        "![a](/assets/images/my-guide-1.png)\n\n"
        "![b](https://arxiv.org/html/2106.09685v2/x1.png)\n"
    )
    assert "/assets/images/my-guide-1.png" in out
    # The new figure is numbered past the kept one.
    assert "/assets/images/my-guide-2.png" in out
    assert sorted(p.name for p in staging.iterdir()) == [
        "my-guide-1.png",
        "my-guide-2.png",
    ]


def test_blocking_problems_are_their_own_result_field_and_resolvable(
    context: Any,
    repo: FakeRepo,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    work_dir = tmp_path / "guide-my-guide"
    work_dir.mkdir()
    outcome = asyncio.run(
        publish_artifact(context, _request(work_dir, VALID_POST), create_pr=False)
    )
    assert outcome.blocking == ["the claim check did not run"]
    post = str(outcome.markdown_path)

    # Published as is, the run's blocking problems still block the PR.
    capsys.readouterr()
    assert cli.main(["publish", post, "--no-pr"]) == cli.EXIT_OK
    result = json.loads(capsys.readouterr().out)
    assert result["blocking"] == ["the claim check did not run"]

    # The editor states they are fixed: no longer blocking, still to check.
    assert cli.main(["publish", post, "--resolved"]) == cli.EXIT_OK
    result = json.loads(capsys.readouterr().out)
    assert result["blocking"] == []
    (pull,) = repo.pulls
    assert "### Blocking" not in pull.body
    assert f"{publish.RESOLVED_NOTE}the claim check did not run" in pull.body
