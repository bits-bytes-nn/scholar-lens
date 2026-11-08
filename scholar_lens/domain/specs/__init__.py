"""The post kinds as data (docs/engine.md, 4.0): one ``PostSpec`` each."""

from .guide import GUIDE_SPEC
from .paper import PAPER_REVIEW_SPEC, PAPER_SUMMARY_SPEC, paper_spec
from .survey import SURVEY_SPEC

__all__ = [
    "GUIDE_SPEC",
    "PAPER_REVIEW_SPEC",
    "PAPER_SUMMARY_SPEC",
    "SURVEY_SPEC",
    "paper_spec",
]
