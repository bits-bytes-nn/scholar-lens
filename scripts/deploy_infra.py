import os
import sys
from pathlib import Path
from typing import Any

import aws_cdk as core
import boto3
from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import (
    aws_apigatewayv2 as apigwv2,
)
from aws_cdk import (
    aws_apigatewayv2_integrations as apigwv2_integrations,
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
    aws_lambda as lambda_,
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
from scholar_lens.configs import Config
from scholar_lens.src import EnvVars, SSMParams, get_account_id, logger

# Secret SSM params are written as encrypted SecureString out-of-band (see
# put_secure_secrets), NOT baked into the CloudFormation template as plaintext.
SECRET_SSM_PARAMS = {
    SSMParams.GITHUB_TOKEN,
    SSMParams.LANGCHAIN_API_KEY,
    SSMParams.UPSTAGE_API_KEY,
    SSMParams.BRAVE_API_KEY,
    SSMParams.TAVILY_API_KEY,
    SSMParams.SLACK_BOT_TOKEN,
    SSMParams.SLACK_SIGNING_SECRET,
}


def put_secure_secrets(
    boto_session: boto3.Session,
    project_name: str,
    stage: str,
    secrets: dict[SSMParams, str | None],
) -> None:
    """Write secret values to SSM as encrypted SecureString parameters.

    Done with the SSM API (not CDK) because CloudFormation cannot create
    SecureString parameters and a plaintext StringParameter would expose the
    value in the readable stack template. Idempotent via Overwrite=True.
    """
    ssm_client = boto_session.client("ssm")
    for param_enum, value in secrets.items():
        if not value:
            continue
        name = f"/{project_name}/{stage}/{param_enum.value}"
        ssm_client.put_parameter(
            Name=name,
            Value=value,
            Type="SecureString",
            Overwrite=True,
        )
        logger.info("Stored SecureString SSM parameter '%s'.", name)


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
        github_token: str | None = None,
        langchain_api_key: str | None = None,
        upstage_api_key: str | None = None,
        brave_api_key: str | None = None,
        tavily_api_key: str | None = None,
        slack_bot_token: str | None = None,
        slack_signing_secret: str | None = None,
        s3_bucket_name: str | None = None,
        s3_prefix: str = "",
        bedrock_region_name: str | None = None,
        environment_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.stage = stage
        self.s3_bucket_name = s3_bucket_name
        self.s3_prefix = s3_prefix
        self.bedrock_region_name = bedrock_region_name

        self._add_tags()
        self._configure_vpc(vpc_id, subnet_ids)
        self.security_group = self._create_security_group()

        self.topic = self._create_sns_topic(email_address)
        self.instance_role = self._create_instance_role()
        self.job_role = self._create_job_role()
        self.execution_role = self._create_execution_role()

        self.log_groups: list[logs.LogGroup] = []
        review_job_def, guide_job_def = self._create_job_definitions(
            environment_vars or {}
        )
        self.job_queue = self._create_job_queue()
        job_queue = self.job_queue

        self._store_ssm_parameters(
            github_token=github_token,
            langchain_api_key=langchain_api_key,
            upstage_api_key=upstage_api_key,
            brave_api_key=brave_api_key,
            tavily_api_key=tavily_api_key,
            slack_bot_token=slack_bot_token,
            slack_signing_secret=slack_signing_secret,
            job_queue_name=job_queue.job_queue_name,
            job_definition_name=review_job_def.job_definition_name,
            guide_job_definition_name=guide_job_def.job_definition_name,
        )
        self._create_slack_functions()
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
        else:
            self.vpc = ec2.Vpc(
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
            self.vpc_subnets = ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            )

    def _create_security_group(self) -> ec2.SecurityGroup:
        # Egress is required for ECR image pulls, Bedrock, S3, SSM and arXiv/PDF
        # fetches over HTTPS. We deny plaintext HTTP and all inbound traffic
        # (batch jobs accept no connections). Outbound is restricted to 443.
        security_group = ec2.SecurityGroup(
            self,
            "PaperReviewSecurityGroup",
            vpc=self.vpc,
            allow_all_outbound=False,
            security_group_name=self._get_resource_name("paper-review"),
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

    def _bedrock_statements(self) -> list[iam.PolicyStatement]:
        """Bedrock invoke + discovery (shared by the Batch job and Slack worker)."""
        # Bedrock: invoke foundation models and inference profiles in the
        # default region and the (cross-region) Bedrock region.
        bedrock_regions = {self.region}
        if self.bedrock_region_name:
            bedrock_regions.add(self.bedrock_region_name)
        # Foundation models are public, read-only model endpoints. A cross-region
        # inference profile fans a single call out to whichever member region has
        # capacity (e.g. a us-* profile may land in us-east-1 even though we only
        # configured ap-northeast-2 + us-west-2), and newer short-form IDs are
        # invoked region-less (arn:...:bedrock:::foundation-model/...). Pinning
        # foundation-model ARNs to a fixed region list therefore causes sporadic
        # AccessDenied. Allow InvokeModel on foundation models in ANY region (the
        # `*` region segment, plus the region-less form). Inference profiles ARE
        # account-scoped resources, so keep those pinned to the regions we use.
        bedrock_resources: list[str] = [
            f"arn:{self.partition}:bedrock:::foundation-model/*",
            f"arn:{self.partition}:bedrock:*::foundation-model/*",
        ]
        for region in bedrock_regions:
            bedrock_resources.append(
                f"arn:{self.partition}:bedrock:{region}:{self.account}:inference-profile/*"
            )
        return [
            iam.PolicyStatement(
                sid="BedrockInvoke",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    # CountTokens is used to fit long papers to the context window
                    # using Claude's exact tokenizer before invoking the model.
                    "bedrock:CountTokens",
                ],
                resources=bedrock_resources,
            ),
            # Inference-profile/model discovery (used to resolve cross-region model
            # IDs) are list/describe actions that do not support resource-level
            # ARNs, so they require "*". We constrain them to the regions we
            # actually call via an aws:RequestedRegion condition.
            iam.PolicyStatement(
                sid="BedrockDiscovery",
                actions=[
                    "bedrock:ListInferenceProfiles",
                    "bedrock:GetInferenceProfile",
                    "bedrock:ListFoundationModels",
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {"aws:RequestedRegion": sorted(bedrock_regions)}
                },
            ),
        ]

    def _ssm_read_statement(self) -> iam.PolicyStatement:
        """Read only this project's SSM parameters (shared)."""
        return iam.PolicyStatement(
            sid="SsmReadParameters",
            actions=["ssm:GetParameter", "ssm:GetParameters"],
            resources=[
                self._aws_partition_arn(
                    "ssm", f"parameter/{self.project_name}/{self.stage}/*"
                )
            ],
        )

    def _ssm_read_named_statement(self, param_names: list[str]) -> iam.PolicyStatement:
        """Read ONLY the named SSM params under this project/stage.

        Used for the Slack worker so a worker compromise (it sits behind the
        public API Gateway ingress) can't decrypt every provider secret — only
        the bot token + Batch queue/definition params it actually needs. Contrast
        :meth:`_ssm_read_statement`, whose ``/*`` wildcard the in-container job
        role legitimately needs (it uses all provider keys)."""
        return iam.PolicyStatement(
            sid="SsmReadNamedParameters",
            actions=["ssm:GetParameter", "ssm:GetParameters"],
            resources=[
                self._aws_partition_arn(
                    "ssm", f"parameter/{self.project_name}/{self.stage}/{name}"
                )
                for name in param_names
            ],
        )

    def _batch_statements(self) -> list[iam.PolicyStatement]:
        """Submit/describe Batch jobs on this stack's queue + definitions (shared)."""
        # AWS Batch: submit jobs only to this stack's queue + job definitions
        # (needed when the bot/script triggers runs). SubmitJob supports
        # resource-level ARNs; DescribeJobs/ListJobs do not and require "*".
        # The queue is named "<project>-<stage>-paper-review", but there are TWO
        # job definitions ("...-paper-review" and "...-tech-guide") and job names
        # like "...-review-*"/"...-guide-*". Scope to the whole "<project>-<stage>"
        # prefix so guide submissions aren't denied (was pinned to paper-review*).
        stack_prefix = f"{self.project_name}-{self.stage}"
        return [
            iam.PolicyStatement(
                sid="BatchSubmit",
                actions=["batch:SubmitJob"],
                resources=[
                    self._aws_partition_arn(
                        "batch", f"job-queue/{stack_prefix}-paper-review"
                    ),
                    self._aws_partition_arn(
                        "batch", f"job-definition/{stack_prefix}-*"
                    ),
                    self._aws_partition_arn("batch", f"job/{stack_prefix}-*"),
                ],
            ),
            iam.PolicyStatement(
                sid="BatchDescribe",
                actions=["batch:DescribeJobs", "batch:ListJobs"],
                resources=["*"],
            ),
        ]

    def _create_job_policy_statements(self) -> list[iam.PolicyStatement]:
        """Least-privilege statements for the application running in-container."""
        statements: list[iam.PolicyStatement] = list(self._bedrock_statements())

        # S3: read/write only within this project's bucket + prefix.
        if self.s3_bucket_name:
            bucket_arn = f"arn:{self.partition}:s3:::{self.s3_bucket_name}"
            object_arn = (
                f"{bucket_arn}/{self.s3_prefix}/*"
                if self.s3_prefix
                else f"{bucket_arn}/*"
            )
            statements.append(
                iam.PolicyStatement(
                    sid="S3ObjectAccess",
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:PutObjectAcl",
                        "s3:DeleteObject",
                    ],
                    resources=[object_arn],
                )
            )
            statements.append(
                iam.PolicyStatement(
                    sid="S3ListBucket",
                    actions=["s3:ListBucket", "s3:GetBucketLocation"],
                    resources=[bucket_arn],
                )
            )

        # SNS: publish only to this stack's topic.
        statements.append(
            iam.PolicyStatement(
                sid="SnsPublish",
                actions=["sns:Publish"],
                resources=[self.topic.topic_arn],
            )
        )

        # KMS: the SNS topic is encrypted with a customer-managed key, so
        # publishing requires GenerateDataKey/Decrypt on that key.
        statements.append(
            iam.PolicyStatement(
                sid="SnsTopicKey",
                actions=["kms:GenerateDataKey", "kms:Decrypt"],
                resources=[self.topic_key.key_arn],
            )
        )

        # CloudWatch: emit custom run metrics (PutMetricData has no resource-level
        # ARN; scope it to the app's metrics namespace via a condition). Must
        # match METRICS_NAMESPACE in scholar_lens/src/metrics.py.
        statements.append(
            iam.PolicyStatement(
                sid="CloudWatchMetrics",
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": "ScholarLens"}},
            )
        )

        # SSM read + Batch submit/describe (shared with the Slack worker).
        statements.append(self._ssm_read_statement())
        statements.extend(self._batch_statements())
        return statements

    def _create_job_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "PaperReviewJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Application (task) role for Paper Review containers",
            role_name=self._get_resource_name("paper-review-job"),
        )
        for statement in self._create_job_policy_statements():
            role.add_to_policy(statement)
        return role

    def _create_execution_role(self) -> iam.Role:
        # The ECS execution role only needs to pull the image and write logs.
        return iam.Role(
            self,
            "PaperReviewExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="ECS execution role for Paper Review (image pull + logs)",
            role_name=self._get_resource_name("paper-review-exec"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )

    def _create_instance_role(self) -> iam.Role:
        # EC2 container instances in the Batch compute environment need the
        # standard ECS container-instance permissions only.
        return iam.Role(
            self,
            "PaperReviewInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="EC2 instance role for Paper Review Batch compute env",
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
            removal_policy=RemovalPolicy.DESTROY,
            alias=self._get_resource_name("paper-review-sns"),
        )
        self.topic_key = topic_key
        topic = sns.Topic(
            self,
            "PaperReviewTopic",
            topic_name=self._get_resource_name("paper-review"),
            display_name="Scholar Lens Notifications",
            master_key=topic_key,
            enforce_ssl=True,
        )
        if email_address:
            topic.add_subscription(subscriptions.EmailSubscription(email_address))
        return topic

    def _create_job_definitions(
        self, env_vars: dict[str, str]
    ) -> tuple[batch.EcsJobDefinition, batch.EcsJobDefinition]:
        """Build the paper-review and tech-guide job definitions.

        Both share the same container image and roles but run different
        entrypoints: ``scholar_lens.main`` (paper review/summary) and
        ``scholar_lens.tech_guide_main`` (technical guide).
        """
        docker_image_asset = DockerImageAsset(
            self,
            "PaperReviewImage",
            directory=str(Path(__file__).parent.parent),
            file="Dockerfile",
            platform=Platform.LINUX_AMD64,
            exclude=["cdk.out", ".venv", ".git", "**/__pycache__"],
        )
        image = ecs.ContainerImage.from_docker_image_asset(docker_image_asset)
        container_env = {
            EnvVars.TOPIC_ARN.value: self.topic.topic_arn,
            EnvVars.LOG_LEVEL.value: "INFO",
            **env_vars,
        }

        review = self._build_job_definition(
            name="paper-review",
            image=image,
            container_env=container_env,
            command=[
                "python3",
                "-m",
                "scholar_lens.main",
                "--source",
                "Ref::source",
                "--repo-urls",
                "Ref::repo_urls",
                "--parse-pdf",
                "Ref::parse_pdf",
                "--mode",
                "Ref::mode",
                "--slack-channel",
                "Ref::slack_channel",
                "--slack-thread-ts",
                "Ref::slack_thread_ts",
            ],
        )
        guide = self._build_job_definition(
            name="tech-guide",
            image=image,
            container_env=container_env,
            command=[
                "python3",
                "-m",
                "scholar_lens.tech_guide_main",
                "--urls",
                "Ref::urls",
                "--discover-subpages",
                "Ref::discover_subpages",
                "--search-queries",
                "Ref::search_queries",
                "--slack-channel",
                "Ref::slack_channel",
                "--slack-thread-ts",
                "Ref::slack_thread_ts",
            ],
        )
        return review, guide

    def _build_job_definition(
        self,
        *,
        name: str,
        image: ecs.ContainerImage,
        container_env: dict[str, str],
        command: list[str],
    ) -> batch.EcsJobDefinition:
        log_group = logs.LogGroup(
            self,
            f"{name.title().replace('-', '')}LogGroup",
            log_group_name=f"/aws/batch/{self._get_resource_name(name)}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        self.log_groups.append(log_group)
        container = batch.EcsEc2ContainerDefinition(
            self,
            f"{name.title().replace('-', '')}ContainerDef",
            image=image,
            job_role=self.job_role,
            execution_role=self.execution_role,
            # PDF parsing (Unstructured "fast" fallback / Upstage) plus FAISS code
            # embeddings are memory-hungry; 1 GiB OOM-killed (exit 137) on large
            # PDFs. 2 vCPU / 8 GiB gives ample headroom and stays within the
            # compute environments' maxvCpus (4 on-demand / 8 spot).
            cpu=2,
            memory=core.Size.mebibytes(8192),
            command=command,
            environment=container_env,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=f"{self.project_name}-{self.stage}",
                log_group=log_group,
            ),
        )
        return batch.EcsJobDefinition(
            self,
            f"{name.title().replace('-', '')}JobDefinition",
            container=container,
            job_definition_name=self._get_resource_name(name),
            retry_attempts=2,
            timeout=Duration.hours(3),
        )

    def _create_job_queue(self) -> batch.JobQueue:
        job_queue = batch.JobQueue(
            self,
            "PaperReviewJobQueue",
            job_queue_name=self._get_resource_name("paper-review"),
            priority=1,
            compute_environments=[],
        )

        compute_env_config = {
            "instance_role": self.instance_role,
            "instance_types": [ec2.InstanceType("optimal")],
            "vpc": self.vpc,
            "security_groups": [self.security_group],
            "vpc_subnets": self.vpc_subnets,
        }

        ondemand_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "PaperReviewOnDemandComputeEnv",
            allocation_strategy=batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
            compute_environment_name=self._get_resource_name("paper-review-ondemand"),
            maxv_cpus=4,
            **compute_env_config,
        )
        job_queue.add_compute_environment(ondemand_env, 1)

        spot_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "PaperReviewSpotComputeEnv",
            allocation_strategy=batch.AllocationStrategy.SPOT_CAPACITY_OPTIMIZED,
            compute_environment_name=self._get_resource_name("paper-review-spot"),
            spot=True,
            maxv_cpus=8,
            **compute_env_config,
        )
        job_queue.add_compute_environment(spot_env, 2)

        return job_queue

    def _store_ssm_parameters(
        self,
        *,
        github_token: str | None,
        langchain_api_key: str | None,
        upstage_api_key: str | None,
        brave_api_key: str | None,
        tavily_api_key: str | None,
        slack_bot_token: str | None,
        slack_signing_secret: str | None,
        job_queue_name: str,
        job_definition_name: str,
        guide_job_definition_name: str,
    ) -> None:
        # Both job definitions run on the single shared queue.
        ssm_params_to_create = {
            SSMParams.GITHUB_TOKEN: github_token,
            SSMParams.LANGCHAIN_API_KEY: langchain_api_key,
            SSMParams.UPSTAGE_API_KEY: upstage_api_key,
            SSMParams.BRAVE_API_KEY: brave_api_key,
            SSMParams.TAVILY_API_KEY: tavily_api_key,
            SSMParams.SLACK_BOT_TOKEN: slack_bot_token,
            SSMParams.SLACK_SIGNING_SECRET: slack_signing_secret,
            SSMParams.BATCH_JOB_QUEUE: job_queue_name,
            SSMParams.BATCH_JOB_DEFINITION: job_definition_name,
            SSMParams.GUIDE_JOB_QUEUE: job_queue_name,
            SSMParams.GUIDE_JOB_DEFINITION: guide_job_definition_name,
        }

        descriptions = {
            SSMParams.GITHUB_TOKEN: "GitHub Token",
            SSMParams.LANGCHAIN_API_KEY: "Langchain API Key",
            SSMParams.UPSTAGE_API_KEY: "Upstage API Key",
            SSMParams.BRAVE_API_KEY: "Brave Search API Key",
            SSMParams.TAVILY_API_KEY: "Tavily Search API Key",
            SSMParams.SLACK_BOT_TOKEN: "Slack Bot Token (chat.postMessage)",
            SSMParams.SLACK_SIGNING_SECRET: "Slack Signing Secret (Events API request verification)",
            SSMParams.BATCH_JOB_QUEUE: "AWS Batch Job Queue for Scholar Lens",
            SSMParams.BATCH_JOB_DEFINITION: "AWS Batch Job Definition for paper review/summary",
            SSMParams.GUIDE_JOB_QUEUE: "AWS Batch Job Queue for Scholar Lens tech guides",
            SSMParams.GUIDE_JOB_DEFINITION: "AWS Batch Job Definition for technical guides",
        }

        # Secrets are NOT created here: CloudFormation/CDK cannot make
        # SecureString parameters, and a plaintext StringParameter would embed
        # the secret value in the (readable) CloudFormation template. They are
        # written out-of-band as encrypted SecureString parameters by
        # :func:`put_secure_secrets` during deploy. Only non-secret operational
        # params (queue/definition names) are created in the template.
        for param_enum, param_value in ssm_params_to_create.items():
            if param_enum in SECRET_SSM_PARAMS:
                continue
            if param_value:
                param_name = f"/{self.project_name}/{self.stage}/{param_enum.value}"
                ssm.StringParameter(
                    self,
                    f"SsmParam{param_enum.name}",
                    parameter_name=param_name,
                    string_value=param_value,
                    description=descriptions[param_enum],
                    tier=ssm.ParameterTier.STANDARD,
                )

    def _create_slack_functions(self) -> None:
        """Create the Slack Events API receiver + worker Lambdas.

        Inbound Slack mentions hit a public Function URL on the *receiver* (a tiny
        zip Lambda) which verifies the request signature and async-invokes the
        *worker* (a container Lambda) to parse intent and dispatch a Batch job.
        This replaces the old local Socket Mode process. Result posting stays on
        the Batch side (notifier.post_slack_result), unchanged.
        """
        repo_root = str(Path(__file__).parent.parent)

        # Worker: container Lambda on the slim Lambda image. It parses intent
        # (Bedrock) and submits a Batch job, so it reuses the job-role policy
        # (Bedrock invoke + SSM read + Batch submit). No VPC — Slack, Bedrock and
        # SSM are public endpoints, and a VPC Lambda only adds ENI cold-start.
        worker_log_group = logs.LogGroup(
            self,
            "SlackWorkerLogGroup",
            log_group_name=f"/aws/lambda/{self._get_resource_name('slack-worker')}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        worker_role = iam.Role(
            self,
            "SlackWorkerRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Role for the Slack worker Lambda (intent parse + dispatch)",
            role_name=self._get_resource_name("slack-worker"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        # Least privilege: the worker only parses intent (Bedrock), reads the bot
        # token + Batch queue/definition params (SSM), and submits a Batch job. It
        # does NOT touch S3/SNS/KMS/CloudWatch, so it gets just those statement
        # groups — not the full job policy. The SSM read is scoped to the exact
        # params it uses (NOT the /* wildcard) so a worker compromise behind the
        # public ingress can't decrypt the github/upstage/brave/tavily/langchain
        # secrets it never touches.
        for statement in (
            *self._bedrock_statements(),
            self._ssm_read_named_statement(
                [
                    SSMParams.SLACK_BOT_TOKEN.value,
                    SSMParams.BATCH_JOB_QUEUE.value,
                    SSMParams.BATCH_JOB_DEFINITION.value,
                    SSMParams.GUIDE_JOB_QUEUE.value,
                    SSMParams.GUIDE_JOB_DEFINITION.value,
                ]
            ),
            *self._batch_statements(),
        ):
            worker_role.add_to_policy(statement)

        worker_fn = lambda_.DockerImageFunction(
            self,
            "SlackWorkerFunction",
            function_name=self._get_resource_name("slack-worker"),
            code=lambda_.DockerImageCode.from_image_asset(
                directory=repo_root,
                file="Dockerfile.lambda",
                exclude=["cdk.out", ".venv", ".git", "**/__pycache__"],
            ),
            architecture=lambda_.Architecture.X86_64,
            memory_size=2048,
            timeout=Duration.seconds(120),
            role=worker_role,
            log_group=worker_log_group,
            # No async retries: a failed invoke must NOT silently re-dispatch a
            # (costly) Batch job. Slack-retry suppression lives in the receiver.
            retry_attempts=0,
            environment={EnvVars.LOG_LEVEL.value: "INFO"},
        )

        # Receiver: tiny zip Lambda (stdlib + runtime boto3). Packages ONLY
        # lambda_receiver.py so it never imports the heavy scholar_lens package.
        receiver_log_group = logs.LogGroup(
            self,
            "SlackReceiverLogGroup",
            log_group_name=f"/aws/lambda/{self._get_resource_name('slack-receiver')}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        receiver_fn = lambda_.Function(
            self,
            "SlackReceiverFunction",
            function_name=self._get_resource_name("slack-receiver"),
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="lambda_receiver.handler",
            code=lambda_.Code.from_asset(
                str(Path(__file__).parent.parent / "scholar_lens" / "slack"),
                exclude=["*", "!lambda_receiver.py"],
            ),
            memory_size=256,
            timeout=Duration.seconds(10),
            log_group=receiver_log_group,
            environment={
                EnvVars.WORKER_FUNCTION_NAME.value: worker_fn.function_name,
                "PROJECT": self.project_name,
                "STAGE": self.stage,
                EnvVars.LOG_LEVEL.value: "INFO",
            },
        )
        # Receiver may invoke the worker and read the signing secret from SSM.
        worker_fn.grant_invoke(receiver_fn)
        receiver_fn.add_to_role_policy(
            iam.PolicyStatement(
                sid="SsmReadSigningSecret",
                actions=["ssm:GetParameter"],
                resources=[
                    self._aws_partition_arn(
                        "ssm",
                        f"parameter/{self.project_name}/{self.stage}/"
                        f"{SSMParams.SLACK_SIGNING_SECRET.value}",
                    )
                ],
            )
        )

        # Public ingress via HTTP API Gateway. We do NOT use a Lambda Function
        # URL: this account's SCP guardrails block anonymous (Principal:*)
        # lambda:InvokeFunctionUrl, so an auth-NONE Function URL returns 403.
        # API Gateway (execute-api) is permitted and is the supported public
        # entrypoint here. Slack cannot sign SigV4, so the route is unauthorized
        # at the gateway and security is the in-handler HMAC signature check. The
        # receiver already speaks API Gateway payload v2.0 (body/isBase64Encoded/
        # lowercased headers, {statusCode, body} response).
        http_api = apigwv2.HttpApi(
            self,
            "SlackHttpApi",
            api_name=self._get_resource_name("slack-events"),
            description="Slack Events API ingress for Paper Bot",
        )
        http_api.add_routes(
            path="/slack/events",
            methods=[apigwv2.HttpMethod.POST],
            integration=apigwv2_integrations.HttpLambdaIntegration(
                "SlackReceiverIntegration", handler=receiver_fn
            ),
        )
        request_url = f"{http_api.api_endpoint}/slack/events"
        CfnOutput(
            self,
            "SlackEventsRequestUrl",
            value=request_url,
            description="Slack Events API Request URL (paste into Event Subscriptions)",
        )

        # Include the Slack Lambda log groups in the ERROR-line alarm sweep
        # (_create_observability iterates self.log_groups). Otherwise ERRORs in
        # the inbound half — intent-parse/Bedrock/SSM/dispatch failures the worker
        # logs with exc_info — would never raise the CloudWatch alarm or email.
        self.log_groups.append(worker_log_group)
        self.log_groups.append(receiver_log_group)

    def _create_observability(self) -> None:
        """Wire failure/error/cost alarms to the (email-subscribed) SNS topic."""
        sns_action = cw_actions.SnsAction(self.topic)

        # 1) Any Batch job that FAILS (on this stack's queue) -> SNS, with the
        #    job name + status reason so the email is actionable.
        events.Rule(
            self,
            "BatchJobFailedRule",
            rule_name=self._get_resource_name("batch-failed"),
            description="Notify on Batch job failures for Scholar Lens.",
            event_pattern=events.EventPattern(
                source=["aws.batch"],
                detail_type=["Batch Job State Change"],
                detail={
                    "status": ["FAILED"],
                    "jobQueue": [self.job_queue.job_queue_arn],
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

        # 2) ERROR lines in any job log group -> metric -> alarm -> SNS.
        for index, log_group in enumerate(self.log_groups):
            metric_filter = logs.MetricFilter(
                self,
                f"ErrorMetricFilter{index}",
                log_group=log_group,
                metric_namespace=f"{self.project_name}/{self.stage}",
                metric_name="LogErrors",
                filter_pattern=logs.FilterPattern.any_term("ERROR"),
                metric_value="1",
                default_value=0,
            )
            alarm = cloudwatch.Alarm(
                self,
                f"LogErrorsAlarm{index}",
                alarm_name=self._get_resource_name(f"log-errors-{index}"),
                metric=metric_filter.metric(
                    statistic="Sum", period=Duration.minutes(5)
                ),
                threshold=1,
                evaluation_periods=1,
                comparison_operator=(
                    cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD
                ),
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            )
            alarm.add_alarm_action(sns_action)

        # 3) Estimated-cost guardrail on the custom metric emitted by the app
        #    (ScholarLens/EstimatedCostUSD). The app emits this metric BOTH with a
        #    `Mode` dimension (per-pipeline breakdown) AND dimensionless (a
        #    stack-wide aggregate). The alarm watches the dimensionless series — a
        #    CloudWatch alarm can only watch one time series (SEARCH/metric-math
        #    that returns many series is rejected: "SEARCH is not supported on
        #    Metric Alarms"). Alarms if a single 1h window's total spend crosses
        #    the threshold.
        cost_alarm = cloudwatch.Alarm(
            self,
            "EstimatedCostAlarm",
            alarm_name=self._get_resource_name("estimated-cost"),
            metric=cloudwatch.Metric(
                namespace="ScholarLens",
                metric_name="EstimatedCostUSD",
                statistic="Sum",
                period=Duration.hours(1),
            ),
            threshold=50,
            evaluation_periods=1,
            comparison_operator=(cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD),
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        cost_alarm.add_alarm_action(sns_action)


def main() -> None:
    try:
        config = Config.load()
        profile_name = os.environ.get(EnvVars.AWS_PROFILE_NAME.value)

        logger.info(
            "Deploying infrastructure for '%s' in '%s' stage",
            config.resources.project_name,
            config.resources.stage,
        )

        boto_session = boto3.Session(
            region_name=config.resources.default_region_name, profile_name=profile_name
        )
        account_id = get_account_id(boto_session)

        env_vars = {
            EnvVars.LANGCHAIN_TRACING_V2.value: os.getenv(
                EnvVars.LANGCHAIN_TRACING_V2.value, "false"
            ),
            EnvVars.LANGCHAIN_ENDPOINT.value: os.getenv(
                EnvVars.LANGCHAIN_ENDPOINT.value, ""
            ),
            EnvVars.LANGCHAIN_PROJECT.value: config.resources.project_name,
        }

        env = core.Environment(
            account=account_id,
            region=config.resources.default_region_name,
        )

        app = core.App()
        PaperReviewStack(
            app,
            f"PaperReview{config.resources.stage.capitalize()}Stack",
            project_name=config.resources.project_name,
            stage=config.resources.stage,
            vpc_id=config.resources.vpc_id,
            subnet_ids=config.resources.subnet_ids,
            email_address=(
                str(config.resources.email_address)
                if config.resources.email_address
                else None
            ),
            github_token=os.getenv(EnvVars.GITHUB_TOKEN.value),
            langchain_api_key=os.getenv(EnvVars.LANGCHAIN_API_KEY.value),
            upstage_api_key=os.getenv(EnvVars.UPSTAGE_API_KEY.value),
            brave_api_key=os.getenv(EnvVars.BRAVE_API_KEY.value),
            tavily_api_key=os.getenv(EnvVars.TAVILY_API_KEY.value),
            slack_bot_token=os.getenv(EnvVars.SLACK_BOT_TOKEN.value),
            slack_signing_secret=os.getenv(EnvVars.SLACK_SIGNING_SECRET.value),
            s3_bucket_name=config.resources.s3_bucket_name,
            s3_prefix=config.resources.s3_prefix,
            bedrock_region_name=config.resources.bedrock_region_name,
            environment_vars=env_vars,
            env=env,
        )
        app.synth()

        # Write secrets as encrypted SecureString params out-of-band so they are
        # never embedded in the CloudFormation template.
        put_secure_secrets(
            boto_session,
            config.resources.project_name,
            config.resources.stage,
            {
                SSMParams.GITHUB_TOKEN: os.getenv(EnvVars.GITHUB_TOKEN.value),
                SSMParams.LANGCHAIN_API_KEY: os.getenv(EnvVars.LANGCHAIN_API_KEY.value),
                SSMParams.UPSTAGE_API_KEY: os.getenv(EnvVars.UPSTAGE_API_KEY.value),
                SSMParams.BRAVE_API_KEY: os.getenv(EnvVars.BRAVE_API_KEY.value),
                SSMParams.TAVILY_API_KEY: os.getenv(EnvVars.TAVILY_API_KEY.value),
                SSMParams.SLACK_BOT_TOKEN: os.getenv(EnvVars.SLACK_BOT_TOKEN.value),
                SSMParams.SLACK_SIGNING_SECRET: os.getenv(
                    EnvVars.SLACK_SIGNING_SECRET.value
                ),
            },
        )

    except Exception as e:
        logger.error("Error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
