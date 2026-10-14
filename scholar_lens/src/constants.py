import re
from enum import Enum, auto
from urllib.parse import urlparse

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
        four_digit = len(modern.group(2)) == 4
        if four_digit != (int(modern.group(1)) < _FIVE_DIGIT_SINCE_YYMM):
            return None
    return candidate


class AutoNamedEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name.lower()


class EmbeddingModelId(str, Enum):
    EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"
    EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    TITAN_EMBED_V1 = "amazon.titan-embed-text-v1"
    TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
    # NOTE: add new models here


class EnvVars(str, Enum):
    AWS_PROFILE_NAME = "AWS_PROFILE_NAME"
    BRAVE_API_KEY = "BRAVE_API_KEY"
    # Optional: Crossref's "polite pool" (a contact address), an OpenAlex key
    # and a Semantic Scholar key, all for higher limits on reference lookups.
    CROSSREF_MAILTO = "CROSSREF_MAILTO"
    GITHUB_TOKEN = "GITHUB_TOKEN"
    LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"
    LANGCHAIN_TRACING_V2 = "LANGCHAIN_TRACING_V2"
    LANGCHAIN_ENDPOINT = "LANGCHAIN_ENDPOINT"
    LANGCHAIN_PROJECT = "LANGCHAIN_PROJECT"
    LOG_LEVEL = "LOG_LEVEL"
    OPENALEX_API_KEY = "OPENALEX_API_KEY"
    SEMANTIC_SCHOLAR_API_KEY = "SEMANTIC_SCHOLAR_API_KEY"
    TAVILY_API_KEY = "TAVILY_API_KEY"
    TOPIC_ARN = "TOPIC_ARN"
    UPSTAGE_API_KEY = "UPSTAGE_API_KEY"


class LanguageModelId(str, Enum):
    CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_V4_5_HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_V5_5_HAIKU = "anthropic.claude-haiku-5-5"
    CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_V3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_V3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_V4_SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_V4_5_SONNET = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_V4_6_SONNET = "anthropic.claude-sonnet-4-6"
    CLAUDE_V5_SONNET = "anthropic.claude-sonnet-5"
    CLAUDE_V5_5_SONNET = "anthropic.claude-sonnet-5-5"
    CLAUDE_V4_OPUS = "anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_V4_1_OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"
    CLAUDE_V4_5_OPUS = "anthropic.claude-opus-4-5-20251101-v1:0"
    CLAUDE_V4_6_OPUS = "anthropic.claude-opus-4-6-v1"
    CLAUDE_V4_7_OPUS = "anthropic.claude-opus-4-7"
    CLAUDE_V4_8_OPUS = "anthropic.claude-opus-4-8"
    CLAUDE_V5_5_OPUS = "anthropic.claude-opus-5-5"
    CLAUDE_V5_1_FABLE = "anthropic.claude-fable-5-1"
    # NOTE: add new models here


class LocalPaths(str, Enum):
    ASSETS_DIR = "assets"
    FIGURES_DIR = "figures"
    LOGS_DIR = "logs"
    PAPERS_DIR = "papers"
    POSTS_DIR = "_posts"
    REFERENCES_DIR = "references"
    REPOS_DIR = "repos"
    LOGS_FILE = "logs.txt"
    PAPER_FILE = "paper.json"
    PARSED_FILE = "parsed.json"
    TRANSLATION_GUIDELINE_FILE = "translation_guideline.json"


class S3Paths(AutoNamedEnum):
    POSTS = auto()
    RESULTS = auto()
    SURVEYS = auto()
    TRANSLATION_GUIDELINE = auto()


class SSMParams(AutoNamedEnum):
    BATCH_JOB_DEFINITION = "batch-job-definition"
    BATCH_JOB_QUEUE = "batch-job-queue"
    BRAVE_API_KEY = "brave-api-key"
    CROSSREF_MAILTO = "crossref-mailto"
    GITHUB_TOKEN = "github-token"
    LANGCHAIN_API_KEY = "langchain-api-key"
    OPENALEX_API_KEY = "openalex-api-key"
    SEMANTIC_SCHOLAR_API_KEY = "semantic-scholar-api-key"
    TAVILY_API_KEY = "tavily-api-key"
    UPSTAGE_API_KEY = "upstage-api-key"


class AppConstants:
    NULL_STRING: str = "null"

    class External(str, Enum):
        ARXIV_HTML = "https://arxiv.org/html"
        # A paper's LaTeX source (tar.gz, or one gzipped .tex), the fallback
        # for a figure the HTML rendering lost.
        ARXIV_E_PRINT = "https://arxiv.org/e-print"
        ARXIV_PDF = "https://arxiv.org/pdf"
        UPSTAGE_DOCUMENT_PARSE = "https://api.upstage.ai/v1/document-ai/document-parse"


# Provider secrets every run may need: SSM SecureString name → environment
# variable. One mapping for the job (runtime), IAM (deploy_infra) and the
# writer script (put_secrets).
RUN_SECRETS: dict[SSMParams, EnvVars] = {
    SSMParams.GITHUB_TOKEN: EnvVars.GITHUB_TOKEN,
    SSMParams.LANGCHAIN_API_KEY: EnvVars.LANGCHAIN_API_KEY,
    SSMParams.UPSTAGE_API_KEY: EnvVars.UPSTAGE_API_KEY,
    SSMParams.TAVILY_API_KEY: EnvVars.TAVILY_API_KEY,
    SSMParams.BRAVE_API_KEY: EnvVars.BRAVE_API_KEY,
    SSMParams.CROSSREF_MAILTO: EnvVars.CROSSREF_MAILTO,
    SSMParams.OPENALEX_API_KEY: EnvVars.OPENALEX_API_KEY,
    SSMParams.SEMANTIC_SCHOLAR_API_KEY: EnvVars.SEMANTIC_SCHOLAR_API_KEY,
}
