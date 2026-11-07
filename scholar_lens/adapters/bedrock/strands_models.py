"""The engine's LanguageModel port on Strands Agents and Amazon Bedrock.

Each role (read, outline, write, evaluate) has its model, effort and output
limit. Every model call is recorded as a :class:`UsageEvent` under the current
stage (one per agent cycle), also when the call fails. Retries are bounded:
botocore retries a request once (a dropped connection, a throttled start),
and Strands retries a throttle up to three attempts with backoff, also one
that arrives inside the response stream, which botocore never sees; one call
makes at most six requests. Strands also retries a structured
answer that fails its schema. An answer cut at the
output limit is returned as such (``stop_reason="max_tokens"``), not raised:
an agent's tool calls up to then stand.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from strands import Agent, ModelRetryStrategy, tool
from strands.models import BedrockModel
from strands.models.model import CacheConfig
from strands.telemetry.metrics import EventLoopMetrics
from strands.types.exceptions import MaxTokensReachedException

from ...application.ports import (
    AgentLimits,
    AgentOutcome,
    ModelRole,
    ModelText,
    SchemaT,
)
from ...application.usage import current_stage
from ...domain.usage import UsageEvent

logger = logging.getLogger(__name__)

_ATTEMPTS = 3
_CLIENT_CONFIG = Config(
    retries={"max_attempts": 2, "mode": "standard"},
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
    # Another model for a text call Bedrock's content filter keeps stopping.
    fallback: RoleModel | None = None


def _content(context: str, prompt: str) -> str | list[dict[str, object]]:
    """The user turn: ``context`` (what repeated calls share: evidence, source
    passages) before a cache point, then ``prompt``. Calls that differ only
    after the cache point read the shared part at the cache-read rate."""
    if not context.strip():
        return prompt
    return [{"text": context}, {"cachePoint": {"type": "default"}}, {"text": prompt}]


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

    def _model(
        self,
        role: ModelRole,
        *,
        thinking: bool | None = None,
        spec: RoleModel | None = None,
    ) -> BedrockModel:
        spec = spec or self._roles[role]
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

    def _agent(self, model: BedrockModel, system: str, **kwargs: Any) -> Agent:
        return Agent(
            model=model,
            system_prompt=system,
            callback_handler=None,
            retry_strategy=ModelRetryStrategy(
                max_attempts=_ATTEMPTS, initial_delay=4, max_delay=60
            ),
            **kwargs,
        )

    def _usage(
        self, role: ModelRole, metrics: EventLoopMetrics, label: str | None = None
    ) -> None:
        label = label or self._roles[role].model_label
        stage = current_stage()
        invocations = metrics.agent_invocations
        cycles = invocations[-1].cycles if invocations else []
        usages = [c.usage for c in cycles] or [metrics.accumulated_usage]
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

    async def _invoke(
        self, agent: Agent, role: ModelRole, label: str, content: Any, **kwargs: Any
    ) -> tuple[Any, str, str]:
        """(result or None, text, stop reason) of one invocation, its usage
        recorded whatever happens. An answer cut at the output limit comes
        back with its partial text and ``"max_tokens"``."""
        try:
            result = await agent.invoke_async(content, **kwargs)
        except MaxTokensReachedException:
            return None, _last_text(agent), "max_tokens"
        finally:
            self._usage(role, agent.event_loop_metrics, label)
        return result, str(result), str(result.stop_reason)

    def _tries(self, role: ModelRole) -> list[tuple[RoleModel, bool | None]]:
        """The attempts of a call Bedrock's content filter may stop (a long
        answer about attacks, not on every attempt): the role's model again,
        the last time without thinking, then the role's fallback model."""
        spec = self._roles[role]
        tries: list[tuple[RoleModel, bool | None]] = [(spec, None)] * (
            1 + _FILTERED_RETRIES
        ) + [(spec, False)]
        if spec.fallback is not None:
            tries.append((spec.fallback, None))
        return tries

    async def structured(
        self,
        *,
        role: ModelRole,
        system: str,
        prompt: str,
        schema: type[SchemaT],
        context: str = "",
    ) -> SchemaT:
        stop = ""
        for n, (model, thinking) in enumerate(self._tries(role)):
            agent = self._agent(
                self._model(role, thinking=thinking, spec=model), system
            )
            result, _, stop = await self._invoke(
                agent,
                role,
                model.model_label,
                _content(context, prompt),
                structured_output_model=schema,
                limits={"structured_output_attempts": _STRUCTURED_ATTEMPTS},
            )
            output = result.structured_output if result is not None else None
            if isinstance(output, schema):
                return output
            if stop != "content_filtered":
                break
            logger.warning(
                "%s structured call content-filtered (%s); attempt %d.",
                role,
                model.model_label,
                n + 2,
            )
        raise ValueError(
            f"No valid {schema.__name__} after {_STRUCTURED_ATTEMPTS} attempts"
            f" (stop reason {stop})"
        )

    async def text(
        self, *, role: ModelRole, system: str, prompt: str, context: str = ""
    ) -> str:
        # Bedrock's content filter can stop a long answer about attacks
        # (jailbreaks, injections) mid-sentence, and not on every attempt:
        # the call is repeated, the last time without thinking, then made
        # with the role's fallback model.
        tries = self._tries(role)
        text, stop = "", ""
        for n, (model, thinking) in enumerate(tries):
            agent = self._agent(
                self._model(role, thinking=thinking, spec=model), system
            )
            _, text, stop = await self._invoke(
                agent, role, model.model_label, _content(context, prompt)
            )
            if stop != "content_filtered" or n == len(tries) - 1:
                break
            logger.warning(
                "%s call content-filtered after %d characters (%s); attempt %d.",
                role,
                len(text),
                model.model_label,
                n + 2,
            )
        if stop != "end_turn" or len(text.strip()) < 200:
            logger.warning(
                "%s call ended with %s and %d characters of text.",
                role,
                stop,
                len(text.strip()),
            )
        return ModelText(text, cut=stop == "max_tokens")

    async def run_agent(
        self,
        *,
        role: ModelRole,
        system: str,
        prompt: str,
        tools: list[Callable[..., str]],
        limits: AgentLimits,
    ) -> AgentOutcome:
        agent = self._agent(self._model(role), system, tools=[tool(fn) for fn in tools])
        request_limits: dict[str, int] = {"turns": limits.turns}
        if limits.total_tokens:
            request_limits["total_tokens"] = limits.total_tokens
        _, text, stop = await self._invoke(
            agent,
            role,
            self._roles[role].model_label,
            prompt,
            limits=request_limits,
        )
        return AgentOutcome(text=text, stop_reason=stop)


def _last_text(agent: Agent) -> str:
    """The text of the agent's last (partial) assistant message."""
    for message in reversed(agent.messages):
        if message.get("role") == "assistant":
            return "".join(
                str(block["text"])
                for block in message.get("content", [])
                if isinstance(block, dict) and "text" in block
            )
    return ""
