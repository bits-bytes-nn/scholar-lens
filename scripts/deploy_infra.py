"""CDK app: the AWS Batch stack that runs Scholar-Lens jobs.

``cdk synth|diff|deploy`` executes this file (see ``cdk.json``). It only builds
the CloudFormation template; secrets are written to SSM separately by
``scripts/put_secrets.py`` so a read-only synth/diff never touches them.
"""

import os
import sys
from pathlib import Path
from typing import Any

import aws_cdk as core
import boto3
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import (
    aws_batch as batch,
)
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
)
from aws_cdk import (
    aws_cloudwatch_actions as cw_actions,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_ecs as ecs,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as targets,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_sns as sns,
)
from aws_cdk import (
    aws_sns_subscriptions as subscriptions,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from aws_cdk.aws_ecr_assets import DockerImageAsset, Platform
from constructs import Construct

sys.path.append(str(Path(__file__).parent.parent))
from scholar_lens.cli import APP_REPORTED_EXIT_CODES
from scholar_lens.configs import Config
from scholar_lens.configs.config import CONFIG_PATH
from scholar_lens.src.aws_helpers import get_account_id
from scholar_lens.src.constants import (
    RUN_SECRETS,
    EmbeddingModelId,
    EnvVars,
    LanguageModelId,
    SSMParams,
)
from scholar_lens.src.jobs import ssm_base
from scholar_lens.src.logger import logger
from scholar_lens.src.metrics import COST_METRIC_NAME, METRICS_NAMESPACE

# The single job definition runs every artifact kind; the CLI serialises a
# JobRequest into the ``request`` parameter (scholar_lens/src/jobs.py).
JOB_COMMAND = ["python3", "-m", "scholar_lens", "worker", "--request", "Ref::request"]

# Bedrock model vendors the app can invoke ("anthropic", "amazon", "cohere"),
# from the model registries: the job role may call only their models.
BEDROCK_VENDORS = sorted(
    {m.value.split(".")[0] for m in (*LanguageModelId, *EmbeddingModelId)}
)


class PaperReviewStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        project_name: str,
        stage: str = "dev",
        vpc_id: str | None = None,
        subnet_ids: list[str] | None = None,
        email_address: str | None = None,
        s3_bucket_name: str | None = None,
        s3_prefix: str = "",
        bedrock_region_name: str | None = None,
        public_assets: bool = False,
        environment_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.stage = stage
        self.s3_bucket_name = s3_bucket_name
        self.s3_prefix = s3_prefix
        self.bedrock_region_name = bedrock_region_name
        self.public_assets = public_assets
        # Keep encryption keys and logs when a prod stack is deleted.
        self.removal_policy = (
            RemovalPolicy.RETAIN if stage == "prod" else RemovalPolicy.DESTROY
        )

        self._add_tags()
        self._configure_vpc(vpc_id, subnet_ids)
        self.security_group = self._create_security_group()

        self.topic = self._create_sns_topic(email_address)
        self.instance_role = self._create_instance_role()
        self.job_role = self._create_job_role()
        self.execution_role = self._create_execution_role()

        self.job_definition = self._create_job_definition(environment_vars or {})
        self.job_queue = self._create_job_queue()
        self._store_ssm_parameters()
        self._create_observability()

    def _get_resource_name(self, suffix: str) -> str:
        return f"{self.project_name}-{self.stage}-{suffix}"

    def _add_tags(self) -> None:
        for key, value in {
            "ProjectName": self.project_name,
            "Stage": self.stage,
            "CostCenter": self.project_name,
            "ManagedBy": "CDK",
        }.items():
            Tags.of(self).add(key, value)

    def _configure_vpc(self, vpc_id: str | None, subnet_ids: list[str] | None) -> None:
        if vpc_id and subnet_ids:
            self.vpc = ec2.Vpc.from_lookup(self, "BaseVPC", vpc_id=vpc_id)
            self.vpc_subnets = ec2.SubnetSelection(
                subnets=[
                    ec2.Subnet.from_subnet_id(self, f"Subnet{i}", sid)
                    for i, sid in enumerate(subnet_ids)
                ]
            )
            return
        vpc = ec2.Vpc(
            self,
            "BaseVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )
        # S3 traffic (image layers from ECR, artifacts) bypasses the NAT gateway.
        vpc.add_gateway_endpoint(
            "S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3
        )
        self.vpc = vpc
        self.vpc_subnets = ec2.SubnetSelection(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        )

    def _create_security_group(self) -> ec2.SecurityGroup:
        # Jobs accept no connections; outbound is HTTPS only (AWS APIs, paper
        # sources, documentation sites).
        security_group = ec2.SecurityGroup(
            self,
            "PaperReviewSecurityGroup",
            vpc=self.vpc,
            allow_all_outbound=False,
            security_group_name=self._get_resource_name("paper-review"),
            # Changing the description replaces the group, and the fixed name
            # then collides with the old one: keep it as deployed.
            description="Security group for Paper Review (HTTPS egress only)",
        )
        security_group.add_egress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(443),
            description="Allow outbound HTTPS to AWS APIs and paper sources",
        )
        return security_group

    def _aws_partition_arn(self, service: str, resource: str) -> str:
        return f"arn:{self.partition}:{service}:{self.region}:{self.account}:{resource}"

    def _alarm_source_condition(self) -> dict[str, dict[str, str]]:
        """Only this account's CloudWatch alarms (confused-deputy guard)."""
        return {
            "StringEquals": {"aws:SourceAccount": self.account},
            "ArnLike": {
                "aws:SourceArn": self._aws_partition_arn("cloudwatch", "alarm:*")
            },
        }

    def _bedrock_statements(self) -> list[iam.PolicyStatement]:
        bedrock_regions = {self.region}
        if self.bedrock_region_name:
            bedrock_regions.add(self.bedrock_region_name)
        # Cross-region inference profiles route a call to any member region,
        # and newer model ids are region-less, so the vendors' foundation
        # models are allowed in every region. Inference profiles are account
        # resources: pinned.
        bedrock_resources: list[str] = [
            f"arn:{self.partition}:bedrock:{region}::foundation-model/{vendor}.*"
            for region in ("", "*")
            for vendor in BEDROCK_VENDORS
        ]
        for region in sorted(bedrock_regions):
            bedrock_resources.append(
                f"arn:{self.partition}:bedrock:{region}:{self.account}:inference-profile/*"
            )
        return [
            iam.PolicyStatement(
                sid="BedrockInvoke",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:CountTokens",
                ],
                resources=bedrock_resources,
            ),
            # List/describe actions have no resource-level ARNs.
            iam.PolicyStatement(
                sid="BedrockDiscovery",
                actions=[
                    "bedrock:ListInferenceProfiles",
                    "bedrock:GetInferenceProfile",
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {"aws:RequestedRegion": sorted(bedrock_regions)}
                },
            ),
        ]

    def _s3_statements(self) -> list[iam.PolicyStatement]:
        if not self.s3_bucket_name:
            return []
        bucket_arn = f"arn:{self.partition}:s3:::{self.s3_bucket_name}"
        object_arn = (
            f"{bucket_arn}/{self.s3_prefix}/*" if self.s3_prefix else f"{bucket_arn}/*"
        )
        # Posts, figures, the translation guideline and the result
        # manifest (<prefix>/results/<job id>.json) all live under the prefix.
        object_actions = ["s3:GetObject", "s3:PutObject"]
        if self.public_assets:
            object_actions.append("s3:PutObjectAcl")
        return [
            iam.PolicyStatement(
                sid="S3ObjectAccess", actions=object_actions, resources=[object_arn]
            ),
            # ListBucket lets a missing object (the optional translation
            # guideline) read as 404, not 403. A GetObject carries no
            # s3:prefix key, so a prefix condition would not apply to that
            # check; the bucket is the project's own.
            iam.PolicyStatement(
                sid="S3ListBucket", actions=["s3:ListBucket"], resources=[bucket_arn]
            ),
        ]

    def _create_job_policy_statements(self) -> list[iam.PolicyStatement]:
        """Least-privilege statements for the application running in-container."""
        statements = [*self._bedrock_statements(), *self._s3_statements()]
        statements += [
            iam.PolicyStatement(
                sid="SnsPublish",
                actions=["sns:Publish"],
                resources=[self.topic.topic_arn],
            ),
            # The topic is encrypted with a customer-managed key.
            iam.PolicyStatement(
                sid="SnsTopicKey",
                actions=["kms:GenerateDataKey", "kms:Decrypt"],
                resources=[self.topic_key.key_arn],
            ),
            # PutMetricData has no resource ARN; scope it to the app namespace.
            iam.PolicyStatement(
                sid="CloudWatchMetrics",
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={
                    "StringEquals": {"cloudwatch:namespace": METRICS_NAMESPACE}
                },
            ),
            iam.PolicyStatement(
                sid="SsmReadSecrets",
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=[
                    self._aws_partition_arn(
                        "ssm",
                        f"parameter{ssm_base(self.project_name, self.stage)}/{param.value}",
                    )
                    for param in RUN_SECRETS
                ],
            ),
        ]
        return statements

    def _create_job_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "PaperReviewJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Application (task) role for Scholar Lens containers",
            role_name=self._get_resource_name("paper-review-job"),
        )
        for statement in self._create_job_policy_statements():
            role.add_to_policy(statement)
        return role

    def _create_execution_role(self) -> iam.Role:
        # Pull the image and write logs only: the image asset and the log
        # driver grant exactly that on their own repository and log group.
        return iam.Role(
            self,
            "PaperReviewExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="ECS execution role for Scholar Lens (image pull + logs)",
            role_name=self._get_resource_name("paper-review-exec"),
        )

    def _create_instance_role(self) -> iam.Role:
        return iam.Role(
            self,
            "PaperReviewInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="EC2 instance role for the Scholar Lens Batch compute env",
            role_name=self._get_resource_name("paper-review-instance"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                )
            ],
        )

    def _create_sns_topic(self, email_address: str | None) -> sns.Topic:
        topic_key = kms.Key(
            self,
            "PaperReviewTopicKey",
            description="KMS key for Scholar Lens SNS notifications",
            enable_key_rotation=True,
            removal_policy=self.removal_policy,
            alias=self._get_resource_name("paper-review-sns"),
        )
        self.topic_key = topic_key
        # CloudWatch alarm actions need the key (EventBridge targets grant it
        # themselves).
        topic_key.grant_encrypt_decrypt(
            iam.ServicePrincipal(
                "cloudwatch.amazonaws.com",
                conditions=self._alarm_source_condition(),
            )
        )
        topic = sns.Topic(
            self,
            "PaperReviewTopic",
            topic_name=self._get_resource_name("paper-review"),
            display_name="Scholar Lens Notifications",
            master_key=topic_key,
            enforce_ssl=True,
        )
        # enforce_ssl creates an explicit topic policy, which replaces the
        # default account-wide publish permission; alarms need their own grant.
        topic.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowCloudWatchAlarms",
                principals=[iam.ServicePrincipal("cloudwatch.amazonaws.com")],
                actions=["sns:Publish"],
                resources=[topic.topic_arn],
                conditions=self._alarm_source_condition(),
            )
        )
        if email_address:
            topic.add_subscription(subscriptions.EmailSubscription(email_address))
        return topic

    def _create_job_definition(
        self, env_vars: dict[str, str]
    ) -> batch.EcsJobDefinition:
        image_asset = DockerImageAsset(
            self,
            "PaperReviewImage",
            directory=str(Path(__file__).parent.parent),
            file="Dockerfile",
            platform=Platform.LINUX_AMD64,
            # .dockerignore is an allowlist: only the files the build needs.
            ignore_mode=core.IgnoreMode.DOCKER,
        )
        self.log_group = logs.LogGroup(
            self,
            "WorkerLogGroup",
            log_group_name=f"/aws/batch/{self._get_resource_name('worker')}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=self.removal_policy,
        )
        container = batch.EcsEc2ContainerDefinition(
            self,
            "WorkerContainerDef",
            image=ecs.ContainerImage.from_docker_image_asset(image_asset),
            job_role=self.job_role,
            execution_role=self.execution_role,
            # PDF parsing and FAISS code embeddings are memory-hungry.
            cpu=2,
            memory=core.Size.mebibytes(8192),
            command=JOB_COMMAND,
            environment={
                EnvVars.TOPIC_ARN.value: self.topic.topic_arn,
                EnvVars.LOG_LEVEL.value: "INFO",
                **env_vars,
            },
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=f"{self.project_name}-{self.stage}",
                log_group=self.log_group,
            ),
        )
        return batch.EcsJobDefinition(
            self,
            "WorkerJobDefinition",
            container=container,
            job_definition_name=self._get_resource_name("worker"),
            timeout=Duration.hours(3),
            # Re-run only when the host went away; an application failure
            # (non-zero exit) is final, since a rerun repeats the Bedrock spend.
            retry_attempts=3,
            retry_strategies=[
                # Matches "Host EC2*" status reasons: the instance terminated.
                batch.RetryStrategy.of(
                    batch.Action.RETRY, batch.Reason.SPOT_INSTANCE_RECLAIMED
                ),
                batch.RetryStrategy.of(
                    batch.Action.EXIT, batch.Reason.custom(on_reason="*")
                ),
            ],
        )

    def _launch_template(self) -> ec2.LaunchTemplate:
        # IMDSv2 with hop limit 1: containers (bridge network) cannot reach the
        # instance role; they use the job role via the ECS credentials endpoint.
        return ec2.LaunchTemplate(
            self,
            "ComputeLaunchTemplate",
            launch_template_name=self._get_resource_name("compute"),
            require_imdsv2=True,
            http_put_response_hop_limit=1,
        )

    def _create_job_queue(self) -> batch.JobQueue:
        job_queue = batch.JobQueue(
            self,
            "PaperReviewJobQueue",
            job_queue_name=self._get_resource_name("paper-review"),
            priority=1,
            compute_environments=[],
        )
        launch_template = self._launch_template()
        # Explicit kwargs (not a **dict splat) keep mypy checking these calls.
        ondemand_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "PaperReviewOnDemandComputeEnv",
            allocation_strategy=batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
            compute_environment_name=self._get_resource_name("paper-review-ondemand"),
            maxv_cpus=4,
            instance_role=self.instance_role,
            instance_types=[ec2.InstanceType("optimal")],
            launch_template=launch_template,
            vpc=self.vpc,
            security_groups=[self.security_group],
            vpc_subnets=self.vpc_subnets,
        )
        # On-demand only: a spot reclaim re-runs the whole job and repeats its
        # Bedrock spend, which costs far more than the EC2 it saves.
        job_queue.add_compute_environment(ondemand_env, 1)
        return job_queue

    def _store_ssm_parameters(self) -> None:
        """Queue and job-definition names, read by ``scholar-lens --remote``."""
        for param, value, description in (
            (
                SSMParams.BATCH_JOB_QUEUE,
                self.job_queue.job_queue_name,
                "AWS Batch job queue for Scholar Lens",
            ),
            (
                SSMParams.BATCH_JOB_DEFINITION,
                self.job_definition.job_definition_name,
                "AWS Batch job definition for Scholar Lens (all artifact kinds)",
            ),
        ):
            ssm.StringParameter(
                self,
                f"SsmParam{param.name}",
                parameter_name=f"{ssm_base(self.project_name, self.stage)}/{param.value}",
                string_value=value,
                description=description,
                tier=ssm.ParameterTier.STANDARD,
            )

    def _create_observability(self) -> None:
        """Failure and cost alarms to the (email-subscribed) SNS topic.

        The worker notifies on success and failure itself and then exits 10
        (failed) or 11 (rejected); see ``APP_REPORTED_EXIT_CODES``. The rule
        below covers every other terminal failure: a timeout or interruption
        (SIGTERM, exit 143), an OOM kill or native crash, a Python crash or
        argument error before the worker ran (1, 2), a lost host (no exit
        code), or a worker that could not send its own notification (4).
        """
        sns_action = cw_actions.SnsAction(self.topic)
        failed_rule = events.Rule(
            self,
            "BatchJobFailedRule",
            rule_name=self._get_resource_name("batch-failed"),
            description="Notify on Scholar Lens Batch jobs that died abnormally.",
            event_pattern=events.EventPattern(
                source=["aws.batch"],
                detail_type=["Batch Job State Change"],
                detail={
                    "status": ["FAILED"],
                    "jobQueue": [self.job_queue.job_queue_arn],
                    "$or": [
                        {
                            "container": {
                                "exitCode": [
                                    {"anything-but": list(APP_REPORTED_EXIT_CODES)}
                                ]
                            }
                        },
                        {"container": {"exitCode": [{"exists": False}]}},
                    ],
                },
            ),
            targets=[
                targets.SnsTopic(
                    self.topic,
                    message=events.RuleTargetInput.from_text(
                        "Scholar Lens Batch job FAILED: "
                        f"{events.EventField.from_path('$.detail.jobName')} "
                        f"(id {events.EventField.from_path('$.detail.jobId')})\n"
                        f"Reason: {events.EventField.from_path('$.detail.statusReason')}"
                    ),
                )
            ],
        )
        # The target's generated grant lets any EventBridge rule publish; only
        # this rule may (confused-deputy guard).
        self.topic.add_to_resource_policy(
            iam.PolicyStatement(
                sid="OnlyTheFailureRule",
                effect=iam.Effect.DENY,
                principals=[iam.ServicePrincipal("events.amazonaws.com")],
                actions=["sns:Publish"],
                resources=[self.topic.topic_arn],
                conditions={"ArnNotEquals": {"aws:SourceArn": failed_rule.rule_arn}},
            )
        )

        # The app emits EstimatedCostUSD both per Mode and dimensionless; an
        # alarm can watch only one series, so it uses the dimensionless total.
        cost_alarm = cloudwatch.Alarm(
            self,
            "EstimatedCostAlarm",
            alarm_name=self._get_resource_name("estimated-cost"),
            metric=cloudwatch.Metric(
                namespace=METRICS_NAMESPACE,
                metric_name=COST_METRIC_NAME,
                statistic="Sum",
                period=Duration.hours(1),
            ),
            threshold=50,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        cost_alarm.add_alarm_action(sns_action)


def deploy_session(config: Config) -> boto3.Session:
    """Session for the target account: the configured profile, else
    ``AWS_PROFILE_NAME``, else the default credential chain."""
    profile_name = config.resources.profile_name or os.getenv(
        EnvVars.AWS_PROFILE_NAME.value
    )
    return boto3.Session(
        region_name=config.resources.default_region_name, profile_name=profile_name
    )


def check_deployable(config_path: Path, config: Config) -> None:
    """The image bakes in ``config.yaml`` and every job needs the bucket, so a
    deploy without either would yield jobs that all fail."""
    if not config_path.exists():
        raise SystemExit(
            f"{config_path} is missing; copy config-template.yaml and fill it in."
        )
    if not config.resources.s3_bucket_name:
        raise SystemExit("resources.s3_bucket_name must name an existing S3 bucket.")


def main() -> None:
    try:
        config = Config.load()
        check_deployable(CONFIG_PATH, config)
        resources = config.resources
        logger.info(
            "Synthesizing infrastructure for '%s' (%s stage)",
            resources.project_name,
            resources.stage,
        )
        account_id = get_account_id(deploy_session(config))
        env_vars = {
            EnvVars.LANGCHAIN_TRACING_V2.value: os.getenv(
                EnvVars.LANGCHAIN_TRACING_V2.value, "false"
            ),
            EnvVars.LANGCHAIN_ENDPOINT.value: os.getenv(
                EnvVars.LANGCHAIN_ENDPOINT.value, ""
            ),
            EnvVars.LANGCHAIN_PROJECT.value: resources.project_name,
        }
        app = core.App()
        PaperReviewStack(
            app,
            f"PaperReview{resources.stage.capitalize()}Stack",
            project_name=resources.project_name,
            stage=resources.stage,
            vpc_id=resources.vpc_id,
            subnet_ids=resources.subnet_ids,
            email_address=(
                str(resources.email_address) if resources.email_address else None
            ),
            s3_bucket_name=resources.s3_bucket_name,
            s3_prefix=resources.s3_prefix,
            bedrock_region_name=resources.bedrock_region_name,
            public_assets=resources.github.public_assets,
            environment_vars=env_vars,
            env=core.Environment(
                account=account_id, region=resources.default_region_name
            ),
        )
        app.synth()
    except Exception as e:
        logger.error("Error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
