"""The commit the running image was built from, for every job result."""

from __future__ import annotations

import os

from scholar_lens.domain.constants import EnvVars


def image_revision() -> str | None:
    """``SCHOLAR_LENS_REVISION`` as the deploy baked it into the Batch image
    (scripts/deploy_infra.py::source_revision); None outside such an image."""
    value = os.getenv(EnvVars.REVISION.value, "").strip()
    return value if value and value != "unknown" else None
