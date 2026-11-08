"""The technical guide as data: TL;DR, the problem, how it works, how to
use and run it, when to choose it."""

from __future__ import annotations

from ..post_spec import (
    AuthorityRule,
    EvidencePolicy,
    KindCount,
    KindOrder,
    KindSpec,
    MaxSections,
    Ownership,
    PostSpec,
    TitleLength,
    TldrSpec,
)

TLDR_TITLE = "TL;DR"
# Authorities whose pages explain but do not own identifiers.
_SECONDARY = frozenset({"third_party", "older_docs"})

GUIDE_SPEC = PostSpec(
    outline_name="guide outline",
    kinds=(
        KindSpec("tldr", 0),
        KindSpec("overview", 1),
        KindSpec("concepts", 2),
        KindSpec("architecture", 2),
        KindSpec("usage", 3),
        KindSpec("operations", 3),
        KindSpec("choosing", 4),
    ),
    outline_checks=(
        TitleLength(),
        MaxSections(),
        KindCount(
            "tldr",
            "exactly_one",
            "the guide starts with exactly one tldr section",
            position="first",
        ),
        KindCount(
            "overview",
            "exactly_one",
            "exactly one overview section (the problem it solves)",
        ),
        KindCount(
            "architecture",
            "at_least_one",
            "at least one architecture section (how it works inside)",
        ),
        KindCount(
            "choosing",
            "exactly_one",
            "the guide ends with exactly one choosing section (when to choose it, "
            "the alternatives, how it fails)",
            position="last",
        ),
        KindOrder(
            "sections must go tldr, overview, concepts and architecture, usage "
            "and operations, choosing"
        ),
    ),
    ownership=Ownership(empty="section '{title}' has no evidence", exempt=("tldr",)),
    # Settings (commands, flags, defaults) come from the official docs when
    # there are any, and never from another version of them.
    evidence=EvidencePolicy(
        authority_rules=(
            AuthorityRule(
                kinds=frozenset({"code"}),
                rejected=_SECONDARY,
                when_corpus_has="official",
                message="code and commands come from the official docs (another "
                "page's may be stale); record what this page explains instead",
            ),
            AuthorityRule(
                exempt=frozenset({"setting", "code", "relation"}),
                rejected=_SECONDARY,
                when_corpus_has="official",
                names_identifier=True,
                message="flags, metrics, API and config names come from the "
                "official docs; state what this page explains without the "
                "identifier",
            ),
            AuthorityRule(
                kinds=frozenset({"setting"}),
                rejected=frozenset({"older_docs"}),
                message="this page documents another version: its settings are "
                "not current; record how the mechanism worked (method) instead",
            ),
            AuthorityRule(
                kinds=frozenset({"setting"}),
                rejected=frozenset({"", "reference", "third_party", "older_docs"}),
                when_corpus_has="official",
                message="settings come from the official docs only; record what "
                "this page explains (finding, method, relation) instead",
            ),
        ),
        recheck_cached=True,
        same_code_language=True,
    ),
    retry_hint="This is a technical guide to a software tool for practitioners.",
    tldr=TldrSpec(
        kind="tldr",
        title=TLDR_TITLE,
        retry_hint="This is the TL;DR of a technical guide for practitioners.",
        review_context="Section kind: tldr: what the tool is, the problem, its "
        "key mechanism and when to use it.",
        figures_in_prompt=True,
        checks_length=True,
        no_body_copies=True,
    ),
    depth_kinds=frozenset({"concepts", "architecture"}),
    # Sections whose code a reader runs (the running example); concepts and
    # architecture may quote internals to explain them.
    runnable_kinds=frozenset({"usage", "operations"}),
    # A section the checks emptied has nothing to state.
    empty_sections="drop",
    # The code trace and the repetition read are separate reads of the
    # guide: one reviewer asked for both reports the code.
    post_passes=("review", "repetition"),
)
