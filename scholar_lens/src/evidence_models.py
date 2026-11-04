"""The evidence engines' glue on the run context: the model port's roles on
Bedrock, and which figures can be published.

Each role (read, outline, write, evaluate) gets its cross-region profile, the
artifact kind's application inference profile when one is deployed (cost
allocation), its effort and thinking switch and the model's output limit.
"""

from __future__ import annotations

import asyncio

from ..adapters.bedrock.strands_models import RoleModel
from .constants import LanguageModelId
from .logger import logger
from .runtime import RunContext
from .utils import BedrockCrossRegionModelHelper
from .utils.factories import _allocated_profile
from .utils.models import model_info_for_invoked_id


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


def _downloadable(url: str) -> bool:
    from .publisher import download_image

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
