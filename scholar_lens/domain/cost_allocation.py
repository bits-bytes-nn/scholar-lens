"""Names of the per-kind Bedrock application inference profiles.

Each artifact kind calls every model through its own application inference
profile, tagged ``Kind=<kind>``, so Cost Explorer can split Bedrock spend by
kind (docs/redesign.md, 4.4). The stack creates the profiles and the app finds
them by these names; both use this function.
"""

from __future__ import annotations

import hashlib
import re
from typing import get_args

from scholar_lens.domain.jobs import ArtifactKind

# The JobRequest kinds; each gets its own profiles.
ARTIFACT_KINDS: tuple[str, ...] = get_args(ArtifactKind)

_NAME_UNSAFE = re.compile(r"[^0-9A-Za-z_-]+")


def application_profile_name(
    project: str, stage: str, kind: str, system_profile_id: str
) -> str:
    """``<project>-<stage>-<kind>-<model>`` from a system-defined profile id
    such as ``global.anthropic.claude-opus-5-5`` (model ``claude-opus-5-5``).
    Bedrock allows letters, digits, ``-`` and ``_`` up to 64 characters."""
    model = system_profile_id.rsplit(".", 1)[-1].split(":", 1)[0]
    name = _NAME_UNSAFE.sub("-", f"{project}-{stage}-{kind}-{model}")
    if len(name) <= 64:
        return name
    # Cut to fit, with a hash of the whole name: two models of one family
    # would otherwise share the cut name.
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name[:55].rstrip('-_')}-{digest}"
