import os
import sys
from pathlib import Path
from typing import Any

import aws_cdk as core
import boto3
from aws_cdk import (
    Duration,
    Stack,
    Tags,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
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
        environment_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.project_name = project_name
        self.stage = stage

        self._add_tags()
        self._configure_vpc(vpc_id, subnet_ids)
        self.security_group = self._create_security_group()

        self.topic = self._create_sns_topic(email_address)
        self.role = self._create_iam_role()

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
        return ec2.SecurityGroup(
            self,
            "PaperReviewSecurityGroup",
            vpc=self.vpc,
            allow_all_outbound=True,
            security_group_name=self._get_resource_name("paper-review"),
            description="Security group for Paper Review",
        )

    def _create_iam_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "PaperReviewRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("ec2.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            ),
            role_name=self._get_resource_name("paper-review"),
        )

        policy = iam.Policy(
            self,
            "PaperReviewPolicy",
            policy_name=self._get_resource_name("paper-review"),
            statements=[
                iam.PolicyStatement(actions=["s3:*"], resources=["*"]),
                iam.PolicyStatement(actions=["bedrock:*"], resources=["*"]),
                iam.PolicyStatement(
                    actions=["ssm:GetParameter"],
                    resources=[
                        f"arn:aws:ssm:{self.region}:{self.account}:parameter/{self.project_name}/{self.stage}/*"
                    ],
                ),
                iam.PolicyStatement(
                    actions=["sns:Publish"],
                    resources=[self.topic.topic_arn],
                ),
                iam.PolicyStatement(
                    actions=[
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogStreams",
                    ],
                    resources=["*"],
                ),
            ],
        )
        role.attach_inline_policy(policy)
        return role

    def _create_sns_topic(self, email_address: str | None) -> sns.Topic:
        topic = sns.Topic(
            self,
            "PaperReviewTopic",
            topic_name=self._get_resource_name("paper-review"),
            display_name="Scholar Lens Notifications",
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
            directory=str(Path(__file__).parent.parent / "scholar_lens"),
            file="Dockerfile",
            platform=Platform.LINUX_AMD64,
            exclude=["cdk.out", ".venv", ".git", "**/__pycache__"],
        )

        container_env = {
            EnvVars.TOPIC_ARN.value: self.topic.topic_arn,
            EnvVars.LOG_LEVEL.value: "INFO",
            **env_vars,
        }

        container = batch.EcsEc2ContainerDefinition(
            self,
            "PaperReviewContainerDef",
            image=ecs.ContainerImage.from_docker_image_asset(docker_image_asset),
            job_role=self.role,
            execution_role=self.role,
            cpu=1,
            memory=core.Size.mebibytes(1024),
            command=[
                "python3",
                "-m",
                "scholar_lens.main",
                "--arxiv-id",
                "Ref::arxiv_id",
                "--repo-urls",
                "Ref::repo_urls",
                "--parse-pdf",
                "Ref::parse_pdf",
            ],
            environment=container_env,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=f"{self.project_name}-{self.stage}"
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
            "instance_role": self.role,
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

        for param_enum, param_value in ssm_params_to_create.items():
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
            langchain_api_key=os.getenv(EnvVars.LANGCHAIN_API_KEY.value),
            upstage_api_key=os.getenv(EnvVars.UPSTAGE_API_KEY.value),
            environment_vars=env_vars,
            env=env,
        )
        app.synth()

    except Exception as e:
        logger.error("Error occurred: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
