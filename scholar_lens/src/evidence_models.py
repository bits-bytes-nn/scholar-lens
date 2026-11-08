"""The evidence engines' glue on the run context: the model port's roles on
Bedrock, and which figures can be published.

Each role (read, outline, write, evaluate) gets its cross-region profile, the
artifact kind's application inference profile when one is deployed (cost
allocation), its effort and thinking switch and the model's output limit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from scholar_lens.adapters.aws.storage import S3Handler
from scholar_lens.adapters.bedrock.chat_models import (
    BedrockCrossRegionModelHelper,
    _allocated_profile,
)
from scholar_lens.adapters.bedrock.strands_models import RoleModel
from scholar_lens.adapters.bedrock.usage import TokenUsageTracker
from scholar_lens.adapters.scholarly.survey_documents import EvidenceTextStore
from scholar_lens.application.checkpoints import FreshStore
from scholar_lens.application.ports import TextStore
from scholar_lens.domain.constants import LanguageModelId, S3Paths
from scholar_lens.domain.model_registry import model_info_for_invoked_id
from scholar_lens.domain.usage import TokenBudgetExceeded, UsageEvent
from scholar_lens.src.logger import logger
from scholar_lens.src.runtime import RunContext


def role_model(
    context: RunContext, model_id: str, effort: str, thinking: bool
) -> RoleModel:
    session = context.bedrock_boto_session
    region = context.config.resources.bedrock_region_name or session.region_name or ""
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
    tracker: TokenUsageTracker, max_cost_usd: float | None
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
                f"Cost ceiling passed: ${tracker.cost_usd:.2f} > ${max_cost_usd:.2f} "
                f"(across {tracker.call_count} calls)."
            )

    return record


def _downloadable(url: str) -> bool:
    from scholar_lens.adapters.github.publisher import download_image

    try:
        download_image(url)
    except Exception as e:  # noqa: BLE001 - any failure means it cannot be published
        logger.warning("Figure %s cannot be downloaded: %s", url, e)
        return False
    return True


async def unservable(urls: list[str]) -> list[str]:
    """The figure URLs that cannot be downloaded (arXiv HTML lists some it
    does not serve): an image that fails to download fails the publish
    step."""
    ok = await asyncio.gather(*(asyncio.to_thread(_downloadable, u) for u in urls))
    return [u for u, good in zip(urls, ok, strict=True) if not good]


def evidence_store(
    local_dir: Path,
    s3_handler: S3Handler | None,
    s3_prefix: str,
    area: S3Paths = S3Paths.SURVEYS,
    *,
    fresh: bool = False,
) -> TextStore:
    """The run's reading cache and checkpoint store
    (:class:`EvidenceTextStore`); a ``fresh`` run writes it without reading
    it."""
    store = EvidenceTextStore(local_dir, s3_handler, s3_prefix, area)
    return FreshStore(store) if fresh else store
