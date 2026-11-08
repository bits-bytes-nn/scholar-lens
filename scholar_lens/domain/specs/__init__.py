"""The post kinds as data (docs/engine.md, 4.0): one ``PostSpec`` each."""

from scholar_lens.domain.specs.guide import GUIDE_SPEC
from scholar_lens.domain.specs.paper import (
    PAPER_REVIEW_SPEC,
    PAPER_SUMMARY_SPEC,
    paper_spec,
)
from scholar_lens.domain.specs.survey import SURVEY_SPEC

__all__ = [
    "GUIDE_SPEC",
    "PAPER_REVIEW_SPEC",
    "PAPER_SUMMARY_SPEC",
    "SURVEY_SPEC",
    "paper_spec",
]
