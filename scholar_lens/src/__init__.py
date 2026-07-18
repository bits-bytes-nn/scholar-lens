"""Public API for ``scholar_lens.src``, loaded lazily (PEP 562).

Importing a submodule (e.g. ``from scholar_lens.src.utils import is_affirmative``)
runs this package ``__init__`` first. If that eagerly imported every symbol
below, it would drag in the heavy paper-processing stack (arxiv, pymupdf, faiss,
nltk, pillow, numpy) on ANY ``scholar_lens.src.*`` import — including the slim
Slack worker Lambda, which needs none of it and fails to import (e.g. "No module
named 'arxiv'").

So the re-exports are resolved on first attribute access via ``__getattr__``
instead. ``from scholar_lens.src import Paper`` still works exactly as before; the
heavy module (``explainer``) is only imported when ``Paper`` is actually touched.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# ``constants`` and ``logger`` are dependency-light (stdlib only) and imported
# everywhere, so they are eager. Eager-importing ``logger`` is also REQUIRED for
# correctness: the submodule file is ``logger.py`` and the re-exported object is
# also ``logger``. If left lazy, any ``from scholar_lens.src.logger import ...``
# elsewhere would bind the package attribute ``logger`` to the *module*, so
# ``from scholar_lens.src import logger`` would yield the module (no ``.info``)
# instead of the logger object. Eager binding resolves the name to the object.
from .constants import (  # noqa: E402
    AppConstants as AppConstants,
)
from .constants import (
    EmbeddingModelId as EmbeddingModelId,
)
from .constants import (
    EnvVars as EnvVars,
)
from .constants import (
    LanguageModelId as LanguageModelId,
)
from .constants import (
    LocalPaths as LocalPaths,
)
from .constants import (
    S3Paths as S3Paths,
)
from .constants import (
    SSMParams as SSMParams,
)
from .logger import (  # noqa: E402
    is_running_in_aws as is_running_in_aws,
)
from .logger import (
    logger as logger,
)

if TYPE_CHECKING:
    # Static analysers (mypy) need the real symbols + types; these imports are
    # never executed at runtime, so they cost nothing and don't drag in the
    # heavy stack. Runtime resolution happens lazily via __getattr__ below.
    from .arxiv_handler import ArxivHandler as ArxivHandler
    from .aws_helpers import S3Handler as S3Handler
    from .aws_helpers import get_account_id as get_account_id
    from .aws_helpers import get_ssm_param_value as get_ssm_param_value
    from .aws_helpers import submit_batch_job as submit_batch_job
    from .aws_helpers import (
        wait_for_batch_job_completion as wait_for_batch_job_completion,
    )
    from .citation_metadata import ChainedMetadataResolver as ChainedMetadataResolver
    from .citation_metadata import CrossrefProvider as CrossrefProvider
    from .citation_metadata import ReferenceMetadata as ReferenceMetadata
    from .citation_metadata import SemanticScholarProvider as SemanticScholarProvider
    from .citation_summarizer import CitationSummarizer as CitationSummarizer
    from .code_retriever import CodeRetriever as CodeRetriever
    from .code_retriever import NoPythonFilesError as NoPythonFilesError
    from .content_extractor import Attributes as Attributes
    from .content_extractor import Citation as Citation
    from .content_extractor import ContentExtractor as ContentExtractor
    from .explainer import ExplainerGraph as ExplainerGraph
    from .explainer import Paper as Paper
    from .markdown_math import normalize_math_underscores as normalize_math_underscores
    from .metrics import MetricsEmitter as MetricsEmitter
    from .metrics import TokenBudgetExceeded as TokenBudgetExceeded
    from .metrics import TokenUsageTracker as TokenUsageTracker
    from .paper_source import ArxivSource as ArxivSource
    from .paper_source import NotAPdfError as NotAPdfError
    from .paper_source import PaperSource as PaperSource
    from .paper_source import PaperSourceError as PaperSourceError
    from .paper_source import PdfUrlSource as PdfUrlSource
    from .paper_source import resolve_paper_source as resolve_paper_source
    from .parser import Content as Content
    from .parser import Figure as Figure
    from .parser import HTMLRichParser as HTMLRichParser
    from .parser import ParserError as ParserError
    from .parser import PDFParser as PDFParser
    from .publisher import Publisher as Publisher
    from .publisher import PublishRequest as PublishRequest
    from .publisher import build_pr_body as build_pr_body
    from .rate_limiter import RateLimiter as RateLimiter
    from .summarizer import PaperSummarizer as PaperSummarizer
    from .tech_guide import NotTechnicalContentError as NotTechnicalContentError
    from .tech_guide import TechGuide as TechGuide
    from .tech_guide import TechGuideGenerator as TechGuideGenerator
    from .utils import arg_as_bool as arg_as_bool
    from .utils import (
        create_robust_xml_output_parser as create_robust_xml_output_parser,
    )
    from .utils import escape_yaml_double_quoted as escape_yaml_double_quoted
    from .utils import is_affirmative as is_affirmative
    from .utils import is_placeholder as is_placeholder
    from .utils import plot_langchain_graph as plot_langchain_graph
    from .web_research import BraveSearchProvider as BraveSearchProvider
    from .web_research import NullSearchProvider as NullSearchProvider
    from .web_research import TavilySearchProvider as TavilySearchProvider
    from .web_research import WebResearcher as WebResearcher
    from .web_research import WebSearchProvider as WebSearchProvider

# Map each public symbol to the submodule that defines it. Access triggers a
# one-time import of just that submodule (cached on the package thereafter).
_SYMBOL_MODULES: dict[str, str] = {
    "ArxivHandler": "arxiv_handler",
    "S3Handler": "aws_helpers",
    "get_account_id": "aws_helpers",
    "get_ssm_param_value": "aws_helpers",
    "submit_batch_job": "aws_helpers",
    "wait_for_batch_job_completion": "aws_helpers",
    "ChainedMetadataResolver": "citation_metadata",
    "CrossrefProvider": "citation_metadata",
    "ReferenceMetadata": "citation_metadata",
    "SemanticScholarProvider": "citation_metadata",
    "CitationSummarizer": "citation_summarizer",
    "CodeRetriever": "code_retriever",
    "NoPythonFilesError": "code_retriever",
    "Attributes": "content_extractor",
    "Citation": "content_extractor",
    "ContentExtractor": "content_extractor",
    "ExplainerGraph": "explainer",
    "Paper": "explainer",
    "normalize_math_underscores": "markdown_math",
    "MetricsEmitter": "metrics",
    "TokenBudgetExceeded": "metrics",
    "TokenUsageTracker": "metrics",
    "ArxivSource": "paper_source",
    "NotAPdfError": "paper_source",
    "PaperSource": "paper_source",
    "PaperSourceError": "paper_source",
    "PdfUrlSource": "paper_source",
    "resolve_paper_source": "paper_source",
    "Content": "parser",
    "Figure": "parser",
    "HTMLRichParser": "parser",
    "ParserError": "parser",
    "PDFParser": "parser",
    "Publisher": "publisher",
    "PublishRequest": "publisher",
    "build_pr_body": "publisher",
    "RateLimiter": "rate_limiter",
    "PaperSummarizer": "summarizer",
    "NotTechnicalContentError": "tech_guide",
    "TechGuide": "tech_guide",
    "TechGuideGenerator": "tech_guide",
    "arg_as_bool": "utils",
    "create_robust_xml_output_parser": "utils",
    "escape_yaml_double_quoted": "utils",
    "is_affirmative": "utils",
    "is_placeholder": "utils",
    "plot_langchain_graph": "utils",
    "BraveSearchProvider": "web_research",
    "NullSearchProvider": "web_research",
    "TavilySearchProvider": "web_research",
    "WebResearcher": "web_research",
    "WebSearchProvider": "web_research",
}

# Eagerly-bound names (constants + logger) plus the lazily-resolved ones.
_EAGER_NAMES = [
    "AppConstants",
    "EmbeddingModelId",
    "EnvVars",
    "LanguageModelId",
    "LocalPaths",
    "S3Paths",
    "SSMParams",
    "is_running_in_aws",
    "logger",
]
__all__ = sorted([*_EAGER_NAMES, *_SYMBOL_MODULES])


def __getattr__(name: str) -> Any:
    """Resolve a public symbol by importing its submodule on first access."""
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return __all__
