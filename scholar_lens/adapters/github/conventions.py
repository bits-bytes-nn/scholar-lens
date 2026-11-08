"""The blog's post conventions and posts, read from GitHub."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests

from scholar_lens.domain.conventions import (
    CANONICAL_PATH,
    BlogPost,
    PostConventions,
    load_local_conventions,
    parse_blog_post,
)

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT_SECONDS = 20
_FETCH_WORKERS = 4  # stays clear of GitHub's secondary rate limits


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
    *,
    repo_name: str | None,
    branch: str,
    token: str | None,
    use_blog: bool,
    fallback: bool = True,
) -> PostConventions:
    """The blog's rules when reachable, else the bundled copy.

    The copy is kept identical to the blog file by CI, so falling back to it
    is safe; a run still logs which source it used and any drift it sees.
    With ``fallback`` False an unreachable blog raises
    (``requests.RequestException``) instead.
    """
    local = load_local_conventions()
    if not (use_blog and repo_name):
        return local
    try:
        text = fetch_blog_conventions(repo_name, branch, token)
    except requests.RequestException as e:
        if not fallback:
            raise
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


def fetch_blog_posts(repo_name: str, branch: str, token: str | None) -> list[BlogPost]:
    """The front matter of every post on the blog's ``branch``: the tags a
    new post should reuse before inventing one, and the descriptions and
    translation ids it must not clash with.

    One recursive tree listing finds every post under ``_posts/`` (also in
    sub-folders); the posts are then read in parallel. A post that cannot be
    read is skipped, but when most reads fail (rate limit, outage) the call
    raises rather than return a partial list that would mark known tags new.
    """
    api = f"https://api.github.com/repos/{repo_name}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    listing = requests.get(
        f"{api}/git/trees/{quote(branch, safe='')}",
        params={"recursive": "1"},
        headers=headers,
        timeout=_FETCH_TIMEOUT_SECONDS,
    )
    listing.raise_for_status()
    tree = listing.json()
    if tree.get("truncated"):
        logger.warning("The blog's file listing is truncated; some posts are missed.")
    entries = [
        entry
        for entry in tree.get("tree", [])
        if entry.get("type") == "blob"
        and entry.get("path", "").startswith("_posts/")
        and entry["path"].endswith(".md")
    ]
    blob_urls = [f"{api}/git/blobs/{entry['sha']}" for entry in entries]
    raw_headers = {**headers, "Accept": "application/vnd.github.raw+json"}

    def read(url: str) -> str | None:
        try:
            response = requests.get(
                url, headers=raw_headers, timeout=_FETCH_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        texts = list(pool.map(read, blob_urls))
    if failed := sum(text is None for text in texts):
        if failed * 2 > len(texts):
            raise RuntimeError(f"could not read {failed} of {len(texts)} blog posts")
        logger.warning("Could not read %d of %d blog posts.", failed, len(texts))
    return [
        parse_blog_post(entry["path"], text)
        for entry, text in zip(entries, texts, strict=True)
        if text is not None
    ]
