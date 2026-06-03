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
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_ecs as ecs,
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
from scholar_lens.configs import Config
from scholar_lens.src import EnvVars, SSMParams, get_account_id, logger


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

        job_definition = self._create_job_definition(environment_vars or {})
        job_queue = self._create_job_queue()

        self._store_ssm_parameters(
            github_token,
            langchain_api_key,
            upstage_api_key,
            job_queue.job_queue_name,
            job_definition.job_definition_name,
        )

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

    def _create_job_policy_statements(self) -> list[iam.PolicyStatement]:
        """Least-privilege statements for the application running in-container."""
        statements: list[iam.PolicyStatement] = []

        # Bedrock: invoke foundation models and inference profiles in the
        # default region and the (cross-region) Bedrock region.
        bedrock_regions = {self.region}
        if self.bedrock_region_name:
            bedrock_regions.add(self.bedrock_region_name)
        bedrock_resources: list[str] = []
        for region in bedrock_regions:
            bedrock_resources.extend(
                [
                    f"arn:{self.partition}:bedrock:{region}::foundation-model/*",
                    f"arn:{self.partition}:bedrock:{region}:{self.account}:inference-profile/*",
                ]
            )
        statements.append(
            iam.PolicyStatement(
                sid="BedrockInvoke",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=bedrock_resources,
            )
        )
        # Inference-profile/model discovery (used to resolve cross-region model
        # IDs) are list/describe actions that do not support resource-level ARNs,
        # so they require "*". We constrain them to the regions we actually call
        # via an aws:RequestedRegion condition.
        statements.append(
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
            )
        )

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

        # SSM: read only this project's parameters.
        statements.append(
            iam.PolicyStatement(
                sid="SsmReadParameters",
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=[
                    self._aws_partition_arn(
                        "ssm", f"parameter/{self.project_name}/{self.stage}/*"
                    )
                ],
            )
        )

        # AWS Batch: submit jobs only to this stack's queue + job definition
        # (needed when the bot/script triggers runs). SubmitJob supports
        # resource-level ARNs; DescribeJobs/ListJobs do not and require "*".
        batch_name = self._get_resource_name("paper-review")
        statements.append(
            iam.PolicyStatement(
                sid="BatchSubmit",
                actions=["batch:SubmitJob"],
                resources=[
                    self._aws_partition_arn("batch", f"job-queue/{batch_name}"),
                    self._aws_partition_arn("batch", f"job-definition/{batch_name}*"),
                ],
            )
        )
        statements.append(
            iam.PolicyStatement(
                sid="BatchDescribe",
                actions=["batch:DescribeJobs", "batch:ListJobs"],
                resources=["*"],
            )
        )
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

    def _create_job_definition(
        self, env_vars: dict[str, str]
    ) -> batch.EcsJobDefinition:
        docker_image_asset = DockerImageAsset(
            self,
            "PaperReviewImage",
            directory=str(Path(__file__).parent.parent),
            file="Dockerfile",
            platform=Platform.LINUX_AMD64,
            exclude=["cdk.out", ".venv", ".git", "**/__pycache__"],
        )

        container_env = {
            EnvVars.TOPIC_ARN.value: self.topic.topic_arn,
            EnvVars.LOG_LEVEL.value: "INFO",
            **env_vars,
        }

        log_group = logs.LogGroup(
            self,
            "PaperReviewLogGroup",
            log_group_name=f"/aws/batch/{self._get_resource_name('paper-review')}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        container = batch.EcsEc2ContainerDefinition(
            self,
            "PaperReviewContainerDef",
            image=ecs.ContainerImage.from_docker_image_asset(docker_image_asset),
            job_role=self.job_role,
            execution_role=self.execution_role,
            cpu=1,
            memory=core.Size.mebibytes(1024),
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
            ],
            environment=container_env,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=f"{self.project_name}-{self.stage}",
                log_group=log_group,
            ),
        )

        return batch.EcsJobDefinition(
            self,
            "PaperReviewJobDefinition",
            container=container,
            job_definition_name=self._get_resource_name("paper-review"),
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
        github_token: str | None,
        langchain_api_key: str | None,
        upstage_api_key: str | None,
        job_queue_name: str,
        job_definition_name: str,
    ) -> None:
        ssm_params_to_create = {
            SSMParams.GITHUB_TOKEN: github_token,
            SSMParams.LANGCHAIN_API_KEY: langchain_api_key,
            SSMParams.UPSTAGE_API_KEY: upstage_api_key,
            SSMParams.BATCH_JOB_QUEUE: job_queue_name,
            SSMParams.BATCH_JOB_DEFINITION: job_definition_name,
        }

        descriptions = {
            SSMParams.GITHUB_TOKEN: "GitHub Token",
            SSMParams.LANGCHAIN_API_KEY: "Langchain API Key",
            SSMParams.UPSTAGE_API_KEY: "Upstage API Key",
            SSMParams.BATCH_JOB_QUEUE: "AWS Batch Job Queue Name for Scholar Lens Paper Review",
            SSMParams.BATCH_JOB_DEFINITION: "AWS Batch Job Definition Name for Scholar Lens Paper Review",
        }

        # NOTE: CloudFormation/CDK cannot natively create SecureString SSM
        # parameters. The three secrets (GitHub token, LangChain/Upstage API
        # keys) are written here as standard parameters for bootstrap
        # convenience only; for production they should be migrated to AWS
        # Secrets Manager (rotation + encryption) or re-created out-of-band as
        # SecureString. Access is already restricted to this project's path via
        # the scoped job-role policy.
        secret_params = {
            SSMParams.GITHUB_TOKEN,
            SSMParams.LANGCHAIN_API_KEY,
            SSMParams.UPSTAGE_API_KEY,
        }
        for param_enum, param_value in ssm_params_to_create.items():
            if param_value:
                param_name = f"/{self.project_name}/{self.stage}/{param_enum.value}"
                description = descriptions[param_enum]
                if param_enum in secret_params:
                    description += " (rotate to Secrets Manager for production)"
                ssm.StringParameter(
                    self,
                    f"SsmParam{param_enum.name}",
                    parameter_name=param_name,
                    string_value=param_value,
                    description=description,
                    tier=ssm.ParameterTier.STANDARD,
                )


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
            s3_bucket_name=config.resources.s3_bucket_name,
            s3_prefix=config.resources.s3_prefix,
            bedrock_region_name=config.resources.bedrock_region_name,
            environment_vars=env_vars,
            env=env,
        )
        app.synth()

    except Exception as e:
        logger.error("Error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
