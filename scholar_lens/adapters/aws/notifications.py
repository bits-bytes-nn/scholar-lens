"""Run notifications on SNS."""

from __future__ import annotations

import logging

import boto3

logger = logging.getLogger(__name__)


def publish_sns(
    session: boto3.Session, topic_arn: str, *, subject: str, message: str
) -> bool:
    """Publish a plain-text SNS notification; a failure must not fail the job.

    Returns whether the notification was sent.
    """
    try:
        session.client("sns").publish(
            TopicArn=topic_arn, Subject=subject, Message=message
        )
        return True
    except Exception as e:  # noqa: BLE001 - notification failure must not fail the job
        logger.warning("Failed to send SNS notification: %s", e)
        return False
