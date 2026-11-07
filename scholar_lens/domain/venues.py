"""Publication venues as a post names them.

Index records name a paper's venue by whatever hosted it: a repository
(Zenodo), the authors' institute (CISPA), a publisher's book series
("Lecture notes in computer science"), or the proceedings' catalogue title in
lower case. :func:`venue_name` keeps only names of where a paper was
published and gives the major ML and security venues their usual short names.
"""

from __future__ import annotations

import re

# Hosts and series that are not where a paper was published.
_NOT_A_VENUE = re.compile(
    r"\b(?:arxiv|zenodo|openreview|ssrn|research ?square|biorxiv|medrxiv|"
    r"hal|figshare|github|lecture notes|ceur|university|universit[äa]t|"
    r"institute|institut|helmholtz|center|centre|laborator(?:y|ies)|"
    r"repository|preprint)\b",
    re.IGNORECASE,
)

# (pattern on the full name, canonical name); first match wins.
_CANONICAL = [
    # Findings papers are not main-conference papers: matched first.
    (r"findings of the association for computational linguistics", "Findings of ACL"),
    (r"neural information processing systems|\bneurips\b|\bnips\b", "NeurIPS"),
    (r"international conference on machine learning|\bicml\b", "ICML"),
    (r"international conference on learning representations|\biclr\b", "ICLR"),
    (r"empirical methods in natural language processing|\bemnlp\b", "EMNLP"),
    (
        r"north american chapter of the association for computational linguistics|\bnaacl\b",
        "NAACL",
    ),
    (r"association for computational linguistics|\bacl\b", "ACL"),
    (r"conference on language modeling|\bcolm\b", "COLM"),
    (
        r"transactions on machine learning research|trans\. mach\. learn\. res\.|\btmlr\b",
        "TMLR",
    ),
    (r"journal of machine learning research|\bjmlr\b", "JMLR"),
    (r"aaai conference on artificial intelligence|\baaai\b", "AAAI"),
    (r"international joint conference on artificial intelligence|\bijcai\b", "IJCAI"),
    (r"computer vision and pattern recognition|\bcvpr\b", "CVPR"),
    (r"international conference on computer vision|\biccv\b", "ICCV"),
    (r"european conference on computer vision|\beccv\b", "ECCV"),
    (r"knowledge discovery and data mining|\bkdd\b", "KDD"),
    (r"usenix security", "USENIX Security"),
    (r"network and distributed system security|\bndss\b", "NDSS"),
    (r"computer and communications security|\bccs\b", "ACM CCS"),
    (r"symposium on security and privacy", "IEEE S&P"),
    (r"the web conference|world wide web conference|\bwww\b", "WWW"),
    (r"transactions on software engineering", "IEEE TSE"),
]
_CANONICAL_RES = [(re.compile(p, re.IGNORECASE), name) for p, name in _CANONICAL]


# Words a proceedings title wraps a venue's name in ("Proceedings of the 41st
# International Conference on Machine Learning, Volume 1: Long Papers").
_FILLER = frozenset(
    "proceedings of the annual meeting conference international joint ieee acm "
    "cvf usenix aaai on in and volume long short papers main track advances "
    "symposium sp proc sigkdd sigsac sigir hlt human language technologies "
    "poster oral spotlight computer vision first second third fourth fifth "
    "sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
    "fifteenth twenty thirty forty fiftieth".split()
)
_WORDS = re.compile(r"[^\W\d_]+")


def _qualified(text: str, pattern: re.Pattern[str]) -> bool:
    """Whether the name says more than the matched venue: another event
    built on it ("Workshop", "Transactions of", "Pacific-Asia", "European",
    "and Applications", "Symposium Series"). Such a name is shown as it is,
    not as the main venue."""
    rest = pattern.sub(" ", text)
    words = [w for w in _WORDS.findall(rest.casefold()) if not _ORDINAL.fullmatch(w)]
    return any(w not in _FILLER for w in words)


_ORDINAL = re.compile(r"st|nd|rd|th")


_FINDINGS_OF = re.compile(r"\b(NAACL|EMNLP|EACL|AACL|ACL)\b")


def venue_name(name: str | None) -> str | None:
    """The venue to show for an index's venue string, or None when it names
    a host, an institute or a series rather than a publication venue."""
    text = " ".join((name or "").split())
    if not text:
        return None
    if re.search(r"\bfindings of\b", text, re.IGNORECASE):
        # "Findings of the ACL: NAACL 2025" are NAACL's Findings.
        after = text.split(":", 1)[1] if ":" in text else ""
        conf = _FINDINGS_OF.search(after)
        return f"Findings of {conf.group(1) if conf else 'ACL'}"
    for pattern, canonical in _CANONICAL_RES:
        if pattern.search(text) and not _qualified(text, pattern):
            return canonical
    if _NOT_A_VENUE.search(text):
        return None
    # A catalogue title in lower case ("journal of foo") reads as a typo.
    return text if any(ch.isupper() for ch in text) else text.title()


def reference_venue(name: str | None) -> str | None:
    """The venue of a reference-list entry: :func:`venue_name`, or "arXiv"
    for a paper the index knows only as an arXiv preprint."""
    if venue := venue_name(name):
        return venue
    return "arXiv" if name and "arxiv" in name.lower() else None
