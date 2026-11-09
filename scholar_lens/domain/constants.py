"""Shared vocabulary: model ids, storage paths, SSM parameter and
environment variable names (plain enums, no I/O)."""

from enum import Enum, auto

# A soft token budget (guide, survey plan and write) is a hard ceiling at this
# multiple of it: past the budget the optional checks are skipped, past the
# ceiling the run aborts before the next call.
HARD_BUDGET_MULTIPLE: float = 1.5


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
    GUIDES = auto()
    PAPERS = auto()
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


# The key area of the post engine's stage checkpoints in its store
# (docs/engine.md, 4.9).
CHECKPOINT_AREA = "checkpoints"
