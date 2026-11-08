"""A technical guide: the material gathered for it and the written guide."""

from __future__ import annotations

from dataclasses import dataclass, field

from scholar_lens.domain.research import ResearchCorpus


@dataclass
class TechGuide:
    topic: str
    body: str
    source_urls: list[str] = field(default_factory=list)
    # Page title of each source URL, for the source list's link text.
    source_titles: dict[str, str] = field(default_factory=dict)
    # Degraded-output notes for the PR checklist and the job result.
    notes: list[str] = field(default_factory=list)
    # Problems the post may not be published with as is (listed first).
    blocking: list[str] = field(default_factory=list)
    # Images the writer was offered; any other image in the body is dropped.
    image_urls: list[str] = field(default_factory=list)
    # The search description (``classify``); empty lets the blog derive one.
    description: str = ""


class NotTechnicalContentError(Exception):
    """Raised when the supplied URLs are not technical documentation."""


@dataclass
class GuideMaterial:
    """What :meth:`TechGuideGenerator.gather` collected."""

    topic: str
    corpus: ResearchCorpus
    # The user's reference URLs not already among the docs URLs.
    references: list[str] = field(default_factory=list)
    # The web-search queries run (explicit or planned).
    queries: list[str] = field(default_factory=list)
    # The run's target release (``SourceSet.target.release``): the one the
    # package facts and the code check read from the index (None: unknown).
    release: str | None = None
