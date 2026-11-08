"""AWS sessions: the credentials and region every AWS adapter is built on."""

from __future__ import annotations

import boto3

# The session type the composition root hands the adapters.
AwsSession = boto3.Session


def aws_session(
    *, profile_name: str | None = None, region_name: str | None = None
) -> AwsSession:
    """A session on the named profile (None: the default credential chain)."""
    return boto3.Session(profile_name=profile_name, region_name=region_name)
