"""The evidence engine's model port on Bedrock: each role (read, outline,
write, evaluate) on its cross-region profile, the artifact kind's
application inference profile when one is deployed (cost allocation), its
effort and thinking switch and the model's output limit."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping

import boto3

from scholar_lens.adapters.bedrock import strands_models
from scholar_lens.adapters.bedrock.chat_models import (
    BedrockCrossRegionModelHelper,
    _allocated_profile,
)
from scholar_lens.adapters.bedrock.strands_models import RoleModel
from scholar_lens.application.ports import (
    LanguageModel,
    ModelRole,
    RoleSpec,
    UsageTracker,
)
from scholar_lens.domain.constants import LanguageModelId
from scholar_lens.domain.model_registry import model_info_for_invoked_id
from scholar_lens.domain.usage import (
    TokenBudgetExceeded,
    UsageEvent,
    cost_ceiling_message,
)

logger = logging.getLogger(__name__)


def role_model(
    session: boto3.Session,
    region_name: str | None,
    model_id: str,
    effort: str,
    thinking: bool,
) -> RoleModel:
    region = region_name or session.region_name or ""
    system = BedrockCrossRegionModelHelper.get_cross_region_model_id(
        session, LanguageModelId(model_id), region
    )
    info = model_info_for_invoked_id(system)
    return RoleModel(
        invoke_id=_allocated_profile(session, region, system),
        model_label=system,
        effort=effort,
        thinking=thinking,
        max_tokens=min(64_000, info.max_output_tokens if info else 32_000),
    )


def budgeted(
    tracker: UsageTracker, max_cost_usd: float | None
) -> Callable[[UsageEvent], None]:
    """The model port's usage recorder: each call recorded, the run stopped
    (:class:`TokenBudgetExceeded`) once its estimated cost passes
    ``max_cost_usd``.

    The ceiling counts this attempt's spend only: ``tracker`` is the
    attempt's, and a stage loaded from its checkpoint makes no call, so a
    resumed run (a rerun, a Batch retry) is neither charged again for the
    stages it loads nor credited for them. A run stopped at the ceiling and
    resumed may spend up to the ceiling once more on the stages left."""

    def record(event: UsageEvent) -> None:
        tracker.record(event)
        if max_cost_usd and tracker.cost_usd > max_cost_usd:
            raise TokenBudgetExceeded(
                cost_ceiling_message(tracker.cost_usd, tracker.call_count, max_cost_usd)
            )

    return record


class BedrockEvidenceModels:
    """The :class:`~scholar_lens.application.ports.EvidenceModels` on
    Bedrock (Strands agents)."""

    def __init__(self, session: boto3.Session, region_name: str | None) -> None:
        self.session = session
        self.region_name = region_name

    def role(self, spec: RoleSpec) -> RoleModel:
        model = role_model(
            self.session, self.region_name, spec.model_id, spec.effort, spec.thinking
        )
        if spec.fallback is None:
            return model
        return dataclasses.replace(model, fallback=self.role(spec.fallback))

    def language_model(
        self,
        roles: Mapping[ModelRole, RoleSpec],
        tracker: UsageTracker,
        max_cost_usd: float | None,
    ) -> LanguageModel:
        return strands_models.StrandsLanguageModel(
            self.session,
            {role: self.role(spec) for role, spec in roles.items()},
            budgeted(tracker, max_cost_usd),
        )
