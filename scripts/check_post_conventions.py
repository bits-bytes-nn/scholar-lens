"""Fail when ``assets/post_conventions.yml`` drifts from the blog's canonical copy.

The blog repository owns ``_data/post_conventions.yml``; this repository ships a
copy in the job image. The check compares the parsed YAML of both::

    python scripts/check_post_conventions.py --repo owner/owner.github.io

Repository and branch default to ``BLOG_REPO`` / ``BLOG_BRANCH`` (branch
``main``). ``GITHUB_TOKEN`` is used when set (private repositories). When no
repository is configured the check prints a notice and passes. A missing
canonical file (HTTP 404) fails with exit 2 when a token is set (GitHub answers
404 both for a missing file and for a repository the token cannot read);
without a token a private repository also answers 404, so the check only
prints a notice.

Exit codes: 0 in sync or skipped, 1 drift, 2 the canonical copy could not be
read.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import requests
import yaml

DEFAULT_LOCAL = (
    Path(__file__).resolve().parent.parent / "assets" / "post_conventions.yml"
)
CANONICAL_PATH = "_data/post_conventions.yml"
TIMEOUT_SECONDS = 20


def fetch_canonical(repo: str, branch: str, path: str, token: str | None) -> str | None:
    """Return the file text from GitHub, or None when it does not exist."""
    headers = {"Accept": "application/vnd.github.raw+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        params={"ref": branch},
        headers=headers,
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.text


def check(
    local_path: Path,
    fetch: Callable[[], str | None],
    out: Callable[[str], object] = print,
    *,
    authenticated: bool = False,
) -> int:
    canonical = fetch()
    if canonical is None:
        if authenticated:
            out(
                f"error: {CANONICAL_PATH} not found: the branch lacks it or the "
                "token cannot read the repository."
            )
            return 2
        out(
            f"notice: {CANONICAL_PATH} not found (or the repository is private "
            "and no token is set); skipping drift check."
        )
        return 0
    if not local_path.exists():
        out(f"error: {local_path} is missing; copy the blog's {CANONICAL_PATH}.")
        return 1
    if yaml.safe_load(local_path.read_text(encoding="utf-8")) == yaml.safe_load(
        canonical
    ):
        out(f"{local_path.name} matches the blog's {CANONICAL_PATH}.")
        return 0
    out(
        f"error: {local_path} differs from the blog's {CANONICAL_PATH}; "
        "copy the canonical file here."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.getenv("BLOG_REPO"))
    parser.add_argument("--branch", default=os.getenv("BLOG_BRANCH") or "main")
    parser.add_argument("--path", default=CANONICAL_PATH)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    args = parser.parse_args(argv)
    if not args.repo:
        print("notice: BLOG_REPO not set; skipping post-conventions drift check.")
        return 0
    token = os.getenv("GITHUB_TOKEN") or None
    try:
        return check(
            args.local,
            lambda: fetch_canonical(args.repo, args.branch, args.path, token),
            authenticated=token is not None,
        )
    except requests.RequestException as e:
        print(f"error: could not fetch {args.repo}:{args.path}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
