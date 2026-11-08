"""Paper identifiers: arXiv ids (modern and legacy, from URLs and DOIs)
and DOIs, as the blog, the indexes and the publisher compare them."""

import re
from urllib.parse import unquote, urlparse

# arXiv identifiers: modern "2401.06066" and legacy "hep-th/9901001" /
# "math.GT/0309136", each with an optional version suffix ("v2").
ARXIV_ID_BODY = r"(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Za-z]{2,})?/\d{7})(?:v\d+)?"
ARXIV_ID_PATTERN = re.compile(rf"^{ARXIV_ID_BODY}$")


# arXiv and the mirrors that serve a paper under its arXiv id
# (ar5iv.labs.arxiv.org and export.arxiv.org match too).
_ARXIV_HOST = re.compile(r"(^|\.)(arxiv|ar5iv|alphaxiv)\.org$", re.IGNORECASE)
_ARXIV_URL_PATH = re.compile(
    rf"/(?:abs|pdf|html|overview)/({ARXIV_ID_BODY})(?:\.pdf)?/?$", re.IGNORECASE
)
# New-style ids got a 5-digit sequence number from 2015-01 (YYMM 1501) on.
_FIVE_DIGIT_SINCE_YYMM = 1501
# New-style ids start at 2007-04 (YYMM 0704).
_MODERN_SINCE_YYMM = 704


def arxiv_id_from_url(url: str | None) -> str | None:
    """The arXiv id of an arXiv (or ar5iv/alphaXiv) abs/pdf/html URL, validated
    like :func:`normalize_arxiv_id`; None for any other URL."""
    if not url:
        return None
    parsed = urlparse(url.strip())
    if not _ARXIV_HOST.search(parsed.hostname or ""):
        return None
    match = _ARXIV_URL_PATH.search(parsed.path)
    return normalize_arxiv_id(match.group(1)) if match else None


def arxiv_paper_key(url: str | None) -> str | None:
    """The arXiv id of an arXiv URL without its version: the abs, pdf and
    html links of a paper and all its versions share it. None for any other
    URL."""
    arxiv_id = arxiv_id_from_url(url)
    return re.sub(r"v\d+$", "", arxiv_id) if arxiv_id else None


# The blog lint's paper_doi rule (tech-blog script/lint-posts.rb lines
# 134-139): a bare DOI, no URL, query or fragment (js/citations.js puts it in a
# request path as is), and an arXiv DOI without the paper's version ("v2"):
# arXiv registers one DOI per paper. Ruby's \s is ASCII whitespace.
BARE_DOI = re.compile(r"10\.\d{4,9}/[^ \t\r\n\f\v?#]+")
ARXIV_DOI_VERSION = re.compile(r"10\.48550/.*v\d+", re.IGNORECASE)
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


def bare_doi(value: str | None) -> str | None:
    """``value`` as the blog's ``paper_doi``: a ``doi.org`` URL or ``doi:``
    prefix dropped and an arXiv DOI's version cut; None when what is left is
    no bare DOI (:data:`BARE_DOI`). Other DOIs are kept as registered: a
    "/v1/" in one (``10.18653/v1/…``) is part of its name."""
    if not value:
        return None
    doi = _DOI_PREFIX.sub("", value.strip())
    if ARXIV_DOI_VERSION.fullmatch(doi):
        doi = re.sub(r"v\d+$", "", doi, flags=re.IGNORECASE)
    return doi if BARE_DOI.fullmatch(doi) else None


def is_arxiv_host(url: str) -> bool:
    """Whether ``url`` is served by arXiv itself (arxiv.org or a subdomain
    such as export.arxiv.org and ar5iv.labs.arxiv.org), whose rate limit
    every request there shares."""
    host = (urlparse(url).hostname or "").lower()
    return host == "arxiv.org" or host.endswith(".arxiv.org")


def arxiv_id_from_reference(value: str | None) -> str | None:
    """Like :func:`normalize_arxiv_id`, but also accepts an arXiv URL (how
    references often cite a paper)."""
    return arxiv_id_from_url(value) or normalize_arxiv_id(value)


def normalize_arxiv_id(value: str | None) -> str | None:
    """Return the bare arXiv id in ``value`` (``arXiv:`` prefix stripped), or
    None when it is not a valid arXiv id."""
    if not value:
        return None
    candidate = value.strip()
    if candidate.lower().startswith("arxiv:"):
        candidate = candidate[len("arxiv:") :].strip()
    candidate = re.sub(r"V(\d+)$", r"v\1", candidate)
    if not ARXIV_ID_PATTERN.match(candidate):
        return None
    if modern := re.match(r"^(\d{4})\.(\d+)", candidate):
        yymm = int(modern.group(1))
        # The YYMM prefix is a real month from 0704, when the scheme began:
        # "2024.01234" (month 24) is a made-up id, not a typo of one.
        if yymm < _MODERN_SINCE_YYMM or not 1 <= yymm % 100 <= 12:
            return None
        four_digit = len(modern.group(2)) == 4
        if four_digit != (yymm < _FIVE_DIGIT_SINCE_YYMM):
            return None
    return candidate


# arXiv DOIs (10.48550/arXiv.<id>), bare or as a doi.org URL.
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\S+)$", re.IGNORECASE)


def arxiv_id_from_doi(doi: str | None) -> str | None:
    """The arXiv id behind an arXiv DOI, or None for any other value."""
    match = _ARXIV_DOI.search((doi or "").strip())
    return normalize_arxiv_id(match.group(1)) if match else None


def arxiv_link_key(url: str | None) -> str | None:
    """The unversioned arXiv id an arXiv URL (any mirror, abs/pdf/html,
    with or without ``www.``) or arXiv DOI link names, or None for any
    other link."""
    if key := arxiv_paper_key(url):
        return key
    arxiv_id = arxiv_id_from_doi(url)
    return re.sub(r"v\d+$", "", arxiv_id) if arxiv_id else None


_DOI_HOST = re.compile(r"^(?:(?:dx|www)\.)?doi\.org$", re.IGNORECASE)


def doi_from_link(url: str | None) -> str | None:
    """The DOI a ``doi.org`` (or ``dx.doi.org``) link names, lower-cased
    (DOIs ignore case), percent-encoding decoded; None for any other link."""
    parsed = urlparse(unquote((url or "").strip()))
    if not _DOI_HOST.match(parsed.hostname or ""):
        return None
    doi = bare_doi(parsed.path.lstrip("/"))
    return doi.lower() if doi else None


def paper_link_key(url: str | None) -> str | None:
    """One key per paper whatever link names it: ``arxiv:<id>`` for any
    arXiv URL or arXiv DOI (version dropped), ``doi:<doi>`` for any other
    DOI link; None for a link that names no paper by an identifier."""
    if key := arxiv_link_key(url):
        return f"arxiv:{key}"
    if doi := doi_from_link(url):
        return f"doi:{doi}"
    return None


def paper_url(paper_id: str) -> str:
    """The link of a paper by its id (an arXiv id or a DOI): its arXiv
    abstract page, else its ``doi.org`` page."""
    if arxiv_id := normalize_arxiv_id(paper_id):
        return f"https://arxiv.org/abs/{arxiv_id}"
    return f"https://doi.org/{paper_id}"
