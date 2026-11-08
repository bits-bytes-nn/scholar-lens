from scholar_lens.src.prompts.base import BasePrompt
from scholar_lens.src.prompts.extraction import (
    AttributesExtractionPrompt,
    CitationExtractionPrompt,
    CitationSummaryPrompt,
    CodeAnalysisPrompt,
    CodebaseSummaryPrompt,
    FigureAnalysisPrompt,
)
from scholar_lens.src.prompts.review import (
    PaperEditPrompt,
    PaperFinalizationPrompt,
    PaperPlanningPrompt,
    PaperSectionCheckPrompt,
    PaperSectionPrompt,
)
from scholar_lens.src.prompts.summary import (
    PaperSummaryFixPrompt,
    PaperSummaryPrompt,
    PaperSummaryShortenPrompt,
)
from scholar_lens.src.prompts.survey import (
    SurveyFinalizationPrompt,
    SurveyScopePrompt,
    SurveySelectionPrompt,
)
from scholar_lens.src.prompts.tech_guide import (
    TechGuideRelevancePrompt,
    TechGuideResearchPlanPrompt,
)

__all__ = [
    "AttributesExtractionPrompt",
    "BasePrompt",
    "CitationExtractionPrompt",
    "CitationSummaryPrompt",
    "CodeAnalysisPrompt",
    "CodebaseSummaryPrompt",
    "FigureAnalysisPrompt",
    "PaperEditPrompt",
    "PaperFinalizationPrompt",
    "PaperPlanningPrompt",
    "PaperSectionCheckPrompt",
    "PaperSectionPrompt",
    "PaperSummaryFixPrompt",
    "PaperSummaryPrompt",
    "PaperSummaryShortenPrompt",
    "SurveyFinalizationPrompt",
    "SurveyScopePrompt",
    "SurveySelectionPrompt",
    "TechGuideRelevancePrompt",
    "TechGuideResearchPlanPrompt",
]
