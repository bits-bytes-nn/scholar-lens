"""Contracts between the CDK stack, the CLI worker, and helper scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scholar_lens.cli import build_parser
from scholar_lens.src.jobs import JobRequest
from scripts import check_post_conventions as conventions
from scripts.ci_synth_check import check_template, resources_of_type, synth_template
from scripts.deploy_infra import JOB_COMMAND


@pytest.fixture(scope="module")
def template() -> dict[str, Any]:
    return synth_template()


class TestJobCommandContract:
    def test_job_command_parses_as_worker_request(self) -> None:
        request = JobRequest(
            kind="guide",
            urls=["https://docs.x.dev"],
            search_queries=["multi word query"],
        )
        argv = [
            request.model_dump_json() if arg == "Ref::request" else arg
            for arg in JOB_COMMAND
        ]
        assert argv[:3] == ["python3", "-m", "scholar_lens"]
        args = build_parser().parse_args(argv[3:])
        assert args.command == "worker"
        assert JobRequest.model_validate_json(args.request) == request

    def test_synthesized_template_honours_contract(
        self, template: dict[str, Any]
    ) -> None:
        check_template(template)

    def test_retry_only_on_infrastructure_failures(
        self, template: dict[str, Any]
    ) -> None:
        (job_def,) = resources_of_type(template, "AWS::Batch::JobDefinition")
        rules = job_def["Properties"]["RetryStrategy"]["EvaluateOnExit"]
        assert rules[-1] == {"Action": "EXIT", "OnReason": "*"}
        assert all(r["Action"] == "RETRY" for r in rules[:-1])


class TestStackWiring:
    def test_cloudwatch_may_publish_to_topic(self, template: dict[str, Any]) -> None:
        policies = resources_of_type(template, "AWS::SNS::TopicPolicy")
        statements = [
            s for p in policies for s in p["Properties"]["PolicyDocument"]["Statement"]
        ]
        assert any(
            isinstance(s.get("Principal"), dict)
            and s["Principal"].get("Service") == "cloudwatch.amazonaws.com"
            and s.get("Action") == "sns:Publish"
            for s in statements
        )

    def test_job_role_is_least_privilege(self, template: dict[str, Any]) -> None:
        actions: set[str] = set()
        for policy in resources_of_type(template, "AWS::IAM::Policy"):
            for statement in policy["Properties"]["PolicyDocument"]["Statement"]:
                value = statement["Action"]
                actions.update([value] if isinstance(value, str) else value)
        assert not {a for a in actions if a.startswith("batch:")}
        assert "s3:DeleteObject" not in actions
        assert "s3:PutObjectAcl" not in actions  # public_assets is off

    def test_single_error_alarm(self, template: dict[str, Any]) -> None:
        alarms = resources_of_type(template, "AWS::CloudWatch::Alarm")
        names = sorted(a["Properties"]["AlarmName"] for a in alarms)
        assert names == [
            "scholar-lens-dev-estimated-cost",
            "scholar-lens-dev-log-errors",
        ]


class TestPostConventionsCheck:
    def _local(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "post_conventions.yml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_equal_yaml_passes_despite_formatting(self, tmp_path: Path) -> None:
        local = self._local(tmp_path, "tags: [a, b]\nmax_tags: 5\n")
        assert (
            conventions.check(local, lambda: "max_tags: 5\ntags:\n  - a\n  - b\n") == 0
        )

    def test_drift_fails(self, tmp_path: Path) -> None:
        local = self._local(tmp_path, "max_tags: 5\n")
        assert (
            conventions.check(local, lambda: "max_tags: 3\n", out=lambda _m: None) == 1
        )

    def test_missing_canonical_skips(self, tmp_path: Path) -> None:
        local = self._local(tmp_path, "max_tags: 5\n")
        assert conventions.check(local, lambda: None, out=lambda _m: None) == 0

    def test_no_repo_configured_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BLOG_REPO", raising=False)
        assert conventions.main([]) == 0
