"""The blog's post conventions (categories, covers, tags, forbidden keys).

The canonical file is the blog's ``_data/post_conventions.yml``; the same rules
drive the blog's own lint, so a generated post and a hand-written one answer to
one list. ``assets/post_conventions.yml`` is a copy for tests and offline runs
(CI fails when it drifts). At run time the blog's base-branch copy is read when
the GitHub integration is configured.

A few generator-side limits are not (yet) in the blog file; they default here
and are read from the spec when the blog adds them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .logger import logger

LOCAL_CONVENTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "post_conventions.yml"
)
CANONICAL_PATH = "_data/post_conventions.yml"
DEFAULT_COVER = "/assets/images/default.jpg"
# Specific (non-controlled) tags a generated post carries: short and reusable.
GENERATED_SPECIFIC_TAGS = 5
_FETCH_TIMEOUT_SECONDS = 20


class TypeSpec(BaseModel):
    """A ``categories[0]`` value. ``paper`` posts must name ``paper_author``."""

    model_config = ConfigDict(extra="ignore")
    paper: bool = False
    cover: str | None = None


class TagSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    controlled: list[str]
    lowercase_words: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    # Limits (optional in the blog file). The validator enforces max_specific
    # and only warns past max_controlled; generated posts carry fewer
    # specific tags (GENERATED_SPECIFIC_TAGS).
    max_specific: int = 10
    max_controlled: int = 3

    @field_validator("lowercase_words", mode="before")
    @classmethod
    def _unbool(cls, value: Any) -> Any:
        # YAML 1.1 (PyYAML, Ruby Psych) reads a bare `on` as true; the list
        # means the word, so map the boolean back.
        if isinstance(value, list):
            return ["on" if v is True else "off" if v is False else v for v in value]
        return value


class MathSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # Macros the blog's MathJax config defines on top of the standard set.
    defined_macros: list[str] = Field(
        default_factory=lambda: ["llbracket", "rrbracket", "textsc"]
    )


class PostConventions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    types: dict[str, TypeSpec]
    topics: dict[str, str | None]
    forbidden_keys: list[str] = Field(default_factory=list)
    tags: TagSpec
    math: MathSpec = Field(default_factory=MathSpec)
    # Titles longer than this need a shorter ``seo_title``.
    seo_title_max_length: int = 70

    def is_paper_type(self, category_type: str) -> bool:
        spec = self.types.get(category_type)
        return bool(spec and spec.paper)

    def cover_for(self, category_type: str, topic: str) -> str:
        """The type's own cover, else the topic's, else the site default."""
        type_spec = self.types.get(category_type)
        if type_spec and type_spec.cover:
            return type_spec.cover
        return self.topics.get(topic) or DEFAULT_COVER

    @classmethod
    def from_yaml(cls, text: str) -> PostConventions:
        data: Any = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("post conventions must be a YAML mapping")
        return cls.model_validate(data)


def load_local_conventions(path: Path = LOCAL_CONVENTIONS_PATH) -> PostConventions:
    return PostConventions.from_yaml(path.read_text(encoding="utf-8"))


def fetch_blog_conventions(
    repo_name: str, branch: str, token: str | None
) -> str | None:
    """Raw text of the blog's canonical file, or None when it does not exist."""
    headers = {"Accept": "application/vnd.github.raw+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        f"https://api.github.com/repos/{repo_name}/contents/{CANONICAL_PATH}",
        params={"ref": branch},
        headers=headers,
        timeout=_FETCH_TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def load_conventions(
    *, repo_name: str | None, branch: str, token: str | None, use_blog: bool
) -> PostConventions:
    """The blog's rules when reachable, else the bundled copy.

    The copy is kept identical to the blog file by CI, so falling back to it
    is safe; a run still logs which source it used and any drift it sees.
    """
    local = load_local_conventions()
    if not (use_blog and repo_name):
        return local
    try:
        text = fetch_blog_conventions(repo_name, branch, token)
    except requests.RequestException as e:
        logger.warning(
            "Could not read the blog's post conventions (%s); using the bundled copy.",
            e,
        )
        return local
    if text is None:
        logger.warning("Blog has no %s yet; using the bundled copy.", CANONICAL_PATH)
        return local
    blog = PostConventions.from_yaml(text)
    if blog != local:
        logger.warning(
            "Bundled post conventions differ from %s@%s; following the blog.",
            repo_name,
            branch,
        )
    return blog


def _front_matter_tags(text: str) -> list[str]:
    if not text.startswith("---\n"):
        return []
    end = text.find("\n---\n", 4)
    try:
        data = yaml.safe_load(text[4:end]) if end != -1 else None
    except yaml.YAMLError:
        return []
    tags = data.get("tags") if isinstance(data, dict) else None
    return [str(t) for t in tags] if isinstance(tags, list) else []


def fetch_blog_tags(
    repo_name: str, branch: str, token: str | None, *, limit: int = 200
) -> list[str]:
    """Tags the blog's posts already use, most used first — the candidates a
    new post should reuse before inventing one."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    listing = requests.get(
        f"https://api.github.com/repos/{repo_name}/contents/_posts",
        params={"ref": branch},
        headers=headers,
        timeout=_FETCH_TIMEOUT_SECONDS,
    )
    listing.raise_for_status()
    counts: dict[str, int] = {}
    raw_headers = {**headers, "Accept": "application/vnd.github.raw+json"}
    for entry in listing.json():
        if entry.get("type") != "file" or not entry.get("name", "").endswith(".md"):
            continue
        post = requests.get(
            entry["url"], headers=raw_headers, timeout=_FETCH_TIMEOUT_SECONDS
        )
        post.raise_for_status()
        for tag in _front_matter_tags(post.text):
            counts[tag] = counts.get(tag, 0) + 1
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    return ranked[:limit]
