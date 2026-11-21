"""Contracts between the CDK stack, the CLI worker, and helper scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scholar_lens.cli import APP_REPORTED_EXIT_CODES, build_parser
from scholar_lens.domain.jobs import JobRequest
from scripts import check_post_conventions as conventions
from scripts.ci_synth_check import check_template, resources_of_type, synth_template
from scripts.deploy_infra import JOB_COMMAND, check_deployable, interpreter_name


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
        assert "s3:PutObjectAcl" not in actions  # uploads are always private

    @staticmethod
    def _statements(template: dict[str, Any], sid: str) -> list[dict[str, Any]]:
        return [
            s
            for p in resources_of_type(template, "AWS::IAM::Policy")
            for s in p["Properties"]["PolicyDocument"]["Statement"]
            if s.get("Sid") == sid
        ]

    def test_guide_code_runs_without_credentials_in_its_own_sandbox(
        self, template: dict[str, Any]
    ) -> None:
        (interpreter,) = resources_of_type(
            template, "AWS::BedrockAgentCore::CodeInterpreterCustom"
        )
        props = interpreter["Properties"]
        assert props["NetworkConfiguration"]["NetworkMode"] == "PUBLIC"
        assert "ExecutionRoleArn" not in props
        (statement,) = self._statements(template, "GuideCodeSessions")
        assert all(a.startswith("bedrock-agentcore:") for a in statement["Action"])
        assert "*" not in str(statement["Resource"])
        (job,) = resources_of_type(template, "AWS::Batch::JobDefinition")
        env = {
            e["Name"] for e in job["Properties"]["ContainerProperties"]["Environment"]
        }
        assert "SCHOLAR_LENS_CODE_INTERPRETER_ID" in env

    def test_guide_web_search_is_the_managed_tool_in_its_region(
        self, template: dict[str, Any]
    ) -> None:
        from scholar_lens.configs import Config

        (statement,) = self._statements(template, "GuideWebSearch")
        assert statement["Action"] == "bedrock-agentcore:InvokeWebSearch"
        region = Config().tech_guide.web_search_region
        prefix, _partition, rest = statement["Resource"]["Fn::Join"][1]
        assert (prefix, rest) == (
            "arn:",
            f":bedrock-agentcore:{region}:aws:tool/web-search.v1",
        )

    def test_no_web_search_grant_without_agentcore(self) -> None:
        from scholar_lens.configs import Config
        from scripts.deploy_infra import web_search_region

        config = Config()
        config.tech_guide.web_search = ["tavily", "brave"]
        assert web_search_region(config) is None
        assert web_search_region(Config()) == "ap-northeast-1"

    def test_bedrock_invoke_is_limited_to_the_app_vendors(
        self, template: dict[str, Any]
    ) -> None:
        (invoke,) = self._statements(template, "BedrockInvoke")
        arns = [r["Fn::Join"][1][-1] for r in invoke["Resource"]]
        models = [a for a in arns if "foundation-model/" in a]
        assert models and all(not a.endswith("foundation-model/*") for a in models)
        assert {a.rsplit("/", 1)[1] for a in models} == {
            "amazon.*",
            "anthropic.*",
            "cohere.*",
        }

    def test_bedrock_invoke_covers_the_kind_profiles(
        self, template: dict[str, Any]
    ) -> None:
        (invoke,) = self._statements(template, "BedrockInvoke")
        arns = ["".join(map(str, r["Fn::Join"][1])) for r in invoke["Resource"]]
        assert any(a.endswith(":application-inference-profile/*") for a in arns)

    def test_bedrock_discovery_is_only_what_the_app_calls(
        self, template: dict[str, Any]
    ) -> None:
        # ListInferenceProfiles: utils/factories; GetInferenceProfile: langchain_aws.
        (discovery,) = self._statements(template, "BedrockDiscovery")
        assert sorted(discovery["Action"]) == [
            "bedrock:GetInferenceProfile",
            "bedrock:ListInferenceProfiles",
        ]
        assert "aws:RequestedRegion" in discovery["Condition"]["StringEquals"]

    def test_list_bucket_has_no_prefix_condition(
        self, template: dict[str, Any]
    ) -> None:
        # A GetObject on a missing key is 404 only with an unconditioned
        # ListBucket: S3 does not evaluate s3:prefix for that check.
        (listing,) = self._statements(template, "S3ListBucket")
        assert listing["Action"] == "s3:ListBucket"
        assert "Condition" not in listing

    def test_execution_role_has_only_its_own_grants(
        self, template: dict[str, Any]
    ) -> None:
        roles = resources_of_type(template, "AWS::IAM::Role")
        execution = [
            r
            for r in roles
            if r["Properties"]["RoleName"].endswith("paper-review-exec")
        ]
        assert execution and "ManagedPolicyArns" not in execution[0]["Properties"]

    def test_only_the_failure_rule_may_publish(self, template: dict[str, Any]) -> None:
        statements = [
            s
            for p in resources_of_type(template, "AWS::SNS::TopicPolicy")
            for s in p["Properties"]["PolicyDocument"]["Statement"]
        ]
        (deny,) = [s for s in statements if s.get("Sid") == "OnlyTheFailureRule"]
        assert deny["Effect"] == "Deny"
        assert "aws:SourceArn" in deny["Condition"]["ArnNotEquals"]
        alarm = next(s for s in statements if s.get("Sid") == "AllowCloudWatchAlarms")
        assert "aws:SourceArn" in alarm["Condition"]["ArnLike"]

    def test_on_demand_compute_only(self, template: dict[str, Any]) -> None:
        (env,) = resources_of_type(template, "AWS::Batch::ComputeEnvironment")
        assert env["Properties"]["ComputeResources"]["Type"] == "EC2"

    def test_compute_follows_the_latest_ami(self, template: dict[str, Any]) -> None:
        (env,) = resources_of_type(template, "AWS::Batch::ComputeEnvironment")
        # An in-place infrastructure update (no replacement) moves to the AMI.
        assert env["Properties"]["ReplaceComputeEnvironment"] is False
        assert (
            env["Properties"]["ComputeResources"]["UpdateToLatestImageVersion"] is True
        )

    def test_root_volume_is_encrypted(self, template: dict[str, Any]) -> None:
        (lt,) = resources_of_type(template, "AWS::EC2::LaunchTemplate")
        (device,) = lt["Properties"]["LaunchTemplateData"]["BlockDeviceMappings"]
        assert device["DeviceName"] == "/dev/xvda"
        ebs = device["Ebs"]
        assert ebs["Encrypted"] is True
        assert ebs["VolumeType"] == "gp3"
        assert ebs["VolumeSize"] >= 30
        assert "KmsKeyId" not in ebs  # the AWS-managed EBS key

    def test_compute_tracks_the_launch_template_version(
        self, template: dict[str, Any]
    ) -> None:
        (env,) = resources_of_type(template, "AWS::Batch::ComputeEnvironment")
        spec = env["Properties"]["ComputeResources"]["LaunchTemplate"]
        lt_id = spec["LaunchTemplateId"]["Ref"]
        assert spec["Version"] == {"Fn::GetAtt": [lt_id, "LatestVersionNumber"]}

    def test_compute_uses_the_optimal_bundle_once(
        self, template: dict[str, Any]
    ) -> None:
        # CDK adds "optimal" itself; newer releases reject it as an explicit
        # ec2.InstanceType.
        (env,) = resources_of_type(template, "AWS::Batch::ComputeEnvironment")
        assert env["Properties"]["ComputeResources"]["InstanceTypes"] == ["optimal"]

    def test_only_cost_and_failure_delivery_alarms(
        self, template: dict[str, Any]
    ) -> None:
        # Run outcomes are reported by the app and the failed-job rule; a log
        # alarm would repeat them.
        alarms = resources_of_type(template, "AWS::CloudWatch::Alarm")
        names = sorted(a["Properties"]["AlarmName"] for a in alarms)
        assert names == [
            "scholar-lens-dev-batch-failed-delivery",
            "scholar-lens-dev-estimated-cost",
        ]

    def test_failure_rule_delivery_alarm_watches_the_rule(
        self, template: dict[str, Any]
    ) -> None:
        (rule_id,) = [
            logical_id
            for logical_id, r in template["Resources"].items()
            if r["Type"] == "AWS::Events::Rule"
        ]
        (alarm,) = [
            a
            for a in resources_of_type(template, "AWS::CloudWatch::Alarm")
            if a["Properties"]["MetricName"] == "FailedInvocations"
        ]
        props = alarm["Properties"]
        assert props["Namespace"] == "AWS/Events"
        assert props["Dimensions"] == [{"Name": "RuleName", "Value": {"Ref": rule_id}}]
        assert props["AlarmActions"]

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
        # A job the user cancelled is no alarm; one without a reason is.
        from scholar_lens.domain.jobs import CANCEL_REASON

        assert detail["statusReason"] == [
            {"anything-but": {"prefix": CANCEL_REASON}},
            {"exists": False},
        ]


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

    def test_scheduled_run_needs_a_repo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BLOG_REPO", raising=False)
        assert conventions.main(["--require-repo"]) == 2


@pytest.mark.parametrize(
    ("project", "stage", "expected"),
    [
        ("scholar-lens", "dev", "scholar_lens_dev_guide_code"),
        ("1x.y", "prod", "sl_1x_y_prod_guide_code"),
        ("p" * 60, "dev", "p" * 48),
    ],
)
def test_the_interpreter_name_fits_agentcore(
    project: str, stage: str, expected: str
) -> None:
    assert interpreter_name(project, stage) == expected


def test_the_cost_alarm_follows_the_run_ceilings() -> None:
    from scholar_lens.configs.config import Config
    from scripts.deploy_infra import cost_alarm_usd

    # Defaults: review 40, summary 12, guide 60, survey 60.
    assert cost_alarm_usd(Config()) == 120.0
    config = Config()
    config.tech_guide.max_cost_usd = 90.0
    assert cost_alarm_usd(config) == 180.0


class TestSourceRevision:
    @staticmethod
    def _git_env() -> dict[str, str]:
        import os

        # Never the developer's git config (signing, hooks).
        return {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        }

    def _with_origin(self, tmp_path: Path) -> Path:
        import subprocess

        repo = self._repo(tmp_path / "work")
        remote = tmp_path / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
        for args in (
            ["remote", "add", "origin", str(remote)],
            ["push", "-q", "origin", "main"],
        ):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
        return repo

    @staticmethod
    def _repo(tmp_path: Path) -> Path:
        import subprocess

        def git(*args: str) -> None:
            subprocess.run(
                ["git", *args],
                cwd=tmp_path,
                check=True,
                capture_output=True,
                env=TestSourceRevision._git_env(),
            )

        tmp_path.mkdir(parents=True, exist_ok=True)
        git("init", "-q", "-b", "main")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (tmp_path / "scholar_lens").mkdir()
        (tmp_path / "scholar_lens" / "a.py").write_text("x = 1\n")
        (tmp_path / "notes.txt").write_text("not shipped\n")
        git("add", "scholar_lens")
        git("commit", "-q", "-m", "init")
        return tmp_path

    def test_a_clean_checkout_gives_its_commit(self, tmp_path: Path) -> None:
        from scripts.deploy_infra import source_revision

        repo = self._repo(tmp_path)
        # An untracked file outside the image sources does not count.
        revision = source_revision(repo, "dev")
        assert len(revision) == 12 and "+" not in revision

    @pytest.mark.parametrize("change", ["edit", "new_file"])
    def test_uncommitted_image_sources_are_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
    ) -> None:
        from scripts.deploy_infra import ALLOW_DIRTY_ENV, source_revision

        repo = self._repo(tmp_path)
        target = repo / "scholar_lens" / ("a.py" if change == "edit" else "b.py")
        target.write_text("x = 2\n")
        monkeypatch.delenv(ALLOW_DIRTY_ENV, raising=False)
        with pytest.raises(SystemExit, match="Uncommitted changes"):
            source_revision(repo, "dev")
        monkeypatch.setenv(ALLOW_DIRTY_ENV, "1")
        assert source_revision(repo, "dev").endswith("+dirty")
        # "0" or anything but "1" does not allow.
        monkeypatch.setenv(ALLOW_DIRTY_ENV, "0")
        with pytest.raises(SystemExit, match="Uncommitted changes"):
            source_revision(repo, "dev")

    def test_prod_refuses_uncommitted_changes_even_when_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from scripts.deploy_infra import ALLOW_DIRTY_ENV, source_revision

        repo = self._with_origin(tmp_path)
        (repo / "scholar_lens" / "a.py").write_text("x = 9\n")
        monkeypatch.setenv(ALLOW_DIRTY_ENV, "1")
        # HEAD is the remote's main: only the dirty tree refuses it.
        with pytest.raises(SystemExit, match="uncommitted changes"):
            source_revision(repo, "prod")

    def test_a_repo_whose_image_sources_were_never_committed_is_refused(
        self, tmp_path: Path
    ) -> None:
        import subprocess

        from scripts.deploy_infra import source_revision

        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
        (tmp_path / "notes.txt").write_text("docs only\n")
        for args in (["add", "notes.txt"], ["commit", "-q", "-m", "docs"]):
            subprocess.run(
                ["git", *args],
                cwd=tmp_path,
                check=True,
                capture_output=True,
                env=self._git_env(),
            )
        with pytest.raises(SystemExit, match="No commit holds the image sources"):
            source_revision(tmp_path, "dev")

    def test_git_never_waits_at_a_prompt_and_times_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        import scripts.deploy_infra as deploy

        seen: dict[str, object] = {}

        def hang(*args: object, **kwargs: object) -> object:
            seen.update(kwargs)
            raise subprocess.TimeoutExpired(cmd="git", timeout=30)

        monkeypatch.setattr(deploy.subprocess, "run", hang)
        with pytest.raises(SystemExit, match="timed out"):
            deploy._git(tmp_path, "ls-remote", "origin")
        assert seen["timeout"] == 30
        assert seen["env"]["GIT_TERMINAL_PROMPT"] == "0"  # type: ignore[index]

    def test_a_commit_outside_the_image_keeps_the_revision(
        self, tmp_path: Path
    ) -> None:
        import subprocess

        from scripts.deploy_infra import source_revision

        repo = self._repo(tmp_path)
        before = source_revision(repo, "dev")
        (repo / "notes.txt").write_text("docs only\n")
        subprocess.run(["git", "add", "notes.txt"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "docs"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        # No image source changed: the same image, no new job definition.
        assert source_revision(repo, "dev") == before

    def test_a_shallow_clone_is_refused(self, tmp_path: Path) -> None:
        import subprocess

        from scripts.deploy_infra import source_revision

        repo = self._repo(tmp_path / "full")
        (repo / "scholar_lens" / "a.py").write_text("x = 4\n")
        subprocess.run(
            ["git", "commit", "-qam", "two"], cwd=repo, check=True, capture_output=True
        )
        shallow = tmp_path / "shallow"
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)],
            check=True,
            capture_output=True,
        )
        with pytest.raises(SystemExit, match="full clone"):
            source_revision(shallow, "dev")

    def test_prod_ships_only_the_remotes_main(self, tmp_path: Path) -> None:
        import subprocess

        from scripts.deploy_infra import source_revision

        repo = self._repo(tmp_path / "work")
        remote = tmp_path / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
        for args in (
            ["remote", "add", "origin", str(remote)],
            ["push", "-q", "origin", "main"],
        ):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
        assert source_revision(repo, "prod")
        # A local commit main does not have yet is refused.
        (repo / "scholar_lens" / "a.py").write_text("x = 3\n")
        subprocess.run(
            ["git", "commit", "-qam", "ahead"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        with pytest.raises(SystemExit, match="remote's main"):
            source_revision(repo, "prod")

    def test_the_guard_watches_everything_the_image_ships(self) -> None:
        from scripts.deploy_infra import IMAGE_SOURCES

        root = Path(__file__).parents[2]
        shipped = [
            line[1:].strip().strip("/")
            for line in (root / ".dockerignore").read_text().splitlines()
            if line.startswith("!")
        ]
        assert shipped and all(
            any(path == src or path.startswith(src + "/") for src in IMAGE_SOURCES)
            for path in shipped
        ), shipped

    def test_the_image_carries_the_revision_in_its_last_layers(self) -> None:
        dockerfile = (Path(__file__).parents[2] / "Dockerfile").read_text()
        # After every RUN: a new revision rebuilds no layer.
        last_run = dockerfile.rindex("\nRUN ")
        for line in (
            "ARG SOURCE_REVISION",
            "LABEL org.opencontainers",
            "ENV SCHOLAR_LENS_REVISION",
        ):
            assert dockerfile.index(line) > last_run, line
        assert "ENV SCHOLAR_LENS_REVISION=${SOURCE_REVISION}" in dockerfile
