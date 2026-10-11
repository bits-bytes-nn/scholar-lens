"""Contracts between the CDK stack, the CLI worker, and helper scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scholar_lens.cli import APP_REPORTED_EXIT_CODES, build_parser
from scholar_lens.src.jobs import JobRequest
from scripts import check_post_conventions as conventions
from scripts.ci_synth_check import check_template, resources_of_type, synth_template
from scripts.deploy_infra import JOB_COMMAND, check_deployable


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

    def test_only_the_cost_alarm(self, template: dict[str, Any]) -> None:
        # Run outcomes are reported by the app and the failed-job rule; a log
        # alarm would repeat them.
        alarms = resources_of_type(template, "AWS::CloudWatch::Alarm")
        names = sorted(a["Properties"]["AlarmName"] for a in alarms)
        assert names == ["scholar-lens-dev-estimated-cost"]

    def test_failed_job_rule_skips_only_app_reported_exits(
        self, template: dict[str, Any]
    ) -> None:
        (rule,) = resources_of_type(template, "AWS::Events::Rule")
        detail = rule["Properties"]["EventPattern"]["detail"]
        assert detail["status"] == ["FAILED"]
        assert detail["$or"] == [
            {"container": {"exitCode": [{"anything-but": [10, 11]}]}},
            # A lost host or a job that never started has no exit code.
            {"container": {"exitCode": [{"exists": False}]}},
        ]
        # Not 1/2: those are also a Python crash / argparse error.
        assert list(APP_REPORTED_EXIT_CODES) == [10, 11]


class TestDeployPreconditions:
    def test_missing_config_file_stops_the_deploy(self, tmp_path: Path) -> None:
        from scholar_lens.configs import Config

        with pytest.raises(SystemExit, match="missing"):
            check_deployable(tmp_path / "config.yaml", Config())

    def test_bucket_is_required(self, tmp_path: Path) -> None:
        from scholar_lens.configs import Config

        path = tmp_path / "config.yaml"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="s3_bucket_name"):
            check_deployable(path, Config())


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

    def test_missing_canonical_skips_without_a_token(self, tmp_path: Path) -> None:
        local = self._local(tmp_path, "max_tags: 5\n")
        assert conventions.check(local, lambda: None, out=lambda _m: None) == 0

    def test_missing_canonical_fails_with_a_token(self, tmp_path: Path) -> None:
        local = self._local(tmp_path, "max_tags: 5\n")
        assert (
            conventions.check(
                local, lambda: None, out=lambda _m: None, authenticated=True
            )
            == 2  # cannot verify (missing file or no access), not drift
        )

    def test_no_repo_configured_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BLOG_REPO", raising=False)
        assert conventions.main([]) == 0
