"""The engine's LanguageModel port on Strands Agents and Amazon Bedrock.

Each role (read, outline, write, evaluate) has its model, effort and output
limit. Every model call is recorded as a :class:`UsageEvent` under the current
stage (one per agent cycle). botocore owns retries of throttling and
transient errors here (standard mode); Strands retries a structured answer
that fails its schema.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from strands import Agent, tool
from strands.models import BedrockModel
from strands.models.model import CacheConfig

from ...application.ports import AgentLimits, AgentOutcome, ModelRole, SchemaT
from ...application.usage import current_stage
from ...domain.usage import UsageEvent

logger = logging.getLogger(__name__)

_CLIENT_CONFIG = Config(
    retries={"max_attempts": 6, "mode": "standard"},
    read_timeout=600,
    connect_timeout=20,
)
_STRUCTURED_ATTEMPTS = 3
# Repeats of a text call Bedrock's content filter stopped.
_FILTERED_RETRIES = 2


@dataclass(frozen=True)
class RoleModel:
    """How a role calls Bedrock: ``invoke_id`` is what is called (an
    inference profile id, or the kind's application profile ARN);
    ``model_label`` names the model in usage records."""

    invoke_id: str
    model_label: str
    effort: str = "medium"
    thinking: bool = False
    max_tokens: int = 32_000


class StrandsLanguageModel:
    def __init__(
        self,
        boto_session: boto3.Session,
        roles: dict[ModelRole, RoleModel],
        record_usage: Callable[[UsageEvent], None],
    ) -> None:
        self._session = boto_session
        self._roles = roles
        self._record = record_usage

    def _model(self, role: ModelRole, *, thinking: bool | None = None) -> BedrockModel:
        spec = self._roles[role]
        fields: dict[str, Any] = {"output_config": {"effort": spec.effort}}
        if spec.thinking if thinking is None else thinking:
            fields["thinking"] = {"type": "adaptive"}
        return BedrockModel(
            model_id=spec.invoke_id,
            boto_session=self._session,
            boto_client_config=_CLIENT_CONFIG,
            max_tokens=spec.max_tokens,
            additional_request_fields=fields,
            # An agent re-sends its conversation every cycle: cached, it is
            # billed at the cache-read price. "auto" does not recognize an
            # application profile ARN as Claude, so the format is named.
            cache_config=CacheConfig(
                strategy="anthropic", system_prompt_ttl=True, tools_ttl=True
            ),
        )

    def _usage(self, role: ModelRole, result: Any) -> None:
        label = self._roles[role].model_label
        stage = current_stage()
        invocations = getattr(result.metrics, "agent_invocations", None) or []
        cycles = invocations[-1].cycles if invocations else []
        usages = [c.usage for c in cycles] or [result.metrics.accumulated_usage]
        for usage in usages:
            self._record(
                UsageEvent(
                    stage=stage,
                    model_id=label,
                    input_tokens=int(usage.get("inputTokens", 0)),
                    output_tokens=int(usage.get("outputTokens", 0)),
                    cache_read_tokens=int(usage.get("cacheReadInputTokens", 0)),
                    cache_write_tokens=int(usage.get("cacheWriteInputTokens", 0)),
                )
            )

    async def structured(
        self, *, role: ModelRole, system: str, prompt: str, schema: type[SchemaT]
    ) -> SchemaT:
        agent = Agent(
            model=self._model(role), system_prompt=system, callback_handler=None
        )
        result = await agent.invoke_async(
            prompt,
            structured_output_model=schema,
            limits={"structured_output_attempts": _STRUCTURED_ATTEMPTS},
        )
        self._usage(role, result)
        output = result.structured_output
        if not isinstance(output, schema):
            raise ValueError(
                f"No valid {schema.__name__} after {_STRUCTURED_ATTEMPTS} attempts"
            )
        return output

    async def text(self, *, role: ModelRole, system: str, prompt: str) -> str:
        # Bedrock's content filter can stop a long answer about attacks
        # (jailbreaks, injections) mid-sentence, and not on every attempt:
        # the call is repeated, the last time without thinking.
        attempts = [None] * _FILTERED_RETRIES + [False]
        for n, thinking in enumerate([None, *attempts]):
            agent = Agent(
                model=self._model(role, thinking=thinking),
                system_prompt=system,
                callback_handler=None,
            )
            result = await agent.invoke_async(prompt)
            self._usage(role, result)
            if result.stop_reason != "content_filtered" or n == len(attempts):
                break
            logger.warning(
                "%s call content-filtered after %d characters; attempt %d again.",
                role,
                len(str(result)),
                n + 2,
            )
        text = str(result)
        if result.stop_reason != "end_turn" or len(text.strip()) < 200:
            logger.warning(
                "%s call ended with %s and %d characters of text.",
                role,
                result.stop_reason,
                len(text.strip()),
            )
        return text

    async def run_agent(
        self,
        *,
        role: ModelRole,
        system: str,
        prompt: str,
        tools: list[Callable[..., str]],
        limits: AgentLimits,
    ) -> AgentOutcome:
        agent = Agent(
            model=self._model(role),
            system_prompt=system,
            tools=[tool(fn) for fn in tools],
            callback_handler=None,
        )
        request_limits: dict[str, int] = {"turns": limits.turns}
        if limits.total_tokens:
            request_limits["total_tokens"] = limits.total_tokens
        result = await agent.invoke_async(prompt, limits=request_limits)  # type: ignore[arg-type]
        self._usage(role, result)
        return AgentOutcome(text=str(result), stop_reason=str(result.stop_reason))
