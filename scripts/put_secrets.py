"""Write provider secrets from the environment (or .env) to SSM SecureString.

Run explicitly after the first deploy and whenever a token rotates::

    poetry run python scripts/put_secrets.py               # every secret that is set
    poetry run python scripts/put_secrets.py github-token  # only these parameters

CloudFormation cannot create SecureString parameters, and a plaintext
StringParameter would expose the value in the template, so secrets are kept out
of the CDK app. Only variables that are set are written (Overwrite=True).
"""

import argparse
import os
import sys
from pathlib import Path

import boto3

sys.path.append(str(Path(__file__).parent.parent))
from scholar_lens.configs import Config
from scholar_lens.src.constants import RUN_SECRETS, SSMParams
from scholar_lens.src.jobs import ssm_base
from scholar_lens.src.logger import logger
from scripts.deploy_infra import deploy_session


def put_secure_secrets(
    session: boto3.Session,
    project_name: str,
    stage: str,
    secrets: dict[SSMParams, str | None],
) -> list[str]:
    """Store each non-empty secret; return the parameter names written."""
    ssm_client = session.client("ssm")
    written: list[str] = []
    for param, value in secrets.items():
        if not value:
            continue
        name = f"{ssm_base(project_name, stage)}/{param.value}"
        ssm_client.put_parameter(
            Name=name, Value=value, Type="SecureString", Overwrite=True
        )
        logger.info("Stored SecureString SSM parameter '%s'.", name)
        written.append(name)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    known = [p.value for p in RUN_SECRETS]
    parser.add_argument(
        "params",
        nargs="*",
        metavar="PARAM",
        help=f"SSM parameters to write ({', '.join(known)}); default: all that are set.",
    )
    wanted = set(parser.parse_args(argv).params)
    if unknown := wanted - set(known):
        parser.error(f"unknown parameter(s): {', '.join(sorted(unknown))}")
    config = Config.load()  # also loads .env
    secrets = {
        param: os.getenv(env.value)
        for param, env in RUN_SECRETS.items()
        if not wanted or param.value in wanted
    }
    # A secret named on the command line must be set: a rotation that writes
    # nothing would otherwise look like it succeeded.
    if wanted and (unset := sorted(p.value for p, v in secrets.items() if not v)):
        parser.error(f"not set in the environment or .env: {', '.join(unset)}")
    written = put_secure_secrets(
        deploy_session(config),
        config.resources.project_name,
        config.resources.stage,
        secrets,
    )
    if not written:
        logger.warning("No secret environment variables were set; nothing written.")


if __name__ == "__main__":
    main()
