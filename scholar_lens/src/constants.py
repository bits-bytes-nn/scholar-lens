from enum import Enum, auto


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
    GITHUB_TOKEN = "GITHUB_TOKEN"
    LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"
    LANGCHAIN_TRACING_V2 = "LANGCHAIN_TRACING_V2"
    LANGCHAIN_ENDPOINT = "LANGCHAIN_ENDPOINT"
    LANGCHAIN_PROJECT = "LANGCHAIN_PROJECT"
    LOG_LEVEL = "LOG_LEVEL"
    TOPIC_ARN = "TOPIC_ARN"
    UPSTAGE_API_KEY = "UPSTAGE_API_KEY"


class LanguageModelId(str, Enum):
    CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_V4_5_HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_V3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_V3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_V4_SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_V4_5_SONNET = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_V4_OPUS = "anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_V4_1_OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"
    # NOTE: add new models here


class LocalPaths(str, Enum):
    ASSETS_DIR = "assets"
    FAISS_INDEX_DIR = "faiss_index"
    FIGURES_DIR = "figures"
    GITHUB_CLONE_DIR = "github_clone"
    LOGS_DIR = "logs"
    PAPERS_DIR = "papers"
    POSTS_DIR = "_posts"
    REFERENCES_DIR = "references"
    REPOS_DIR = "repos"
    CONFIG_FILE = "config.yaml"
    CONTENT_FILE = "content.json"
    FIGURES_FILE = "figures.json"
    GRAPH_FILE = "graph.png"
    KEYWORDS_FILE = "keywords.txt"
    LOGS_FILE = "logs.txt"
    PAPER_FILE = "paper.json"
    PARSED_FILE = "parsed.json"
    TRANSLATION_GUIDELINE_FILE = "translation_guideline.json"


class S3Paths(AutoNamedEnum):
    ASSETS = auto()
    KEYWORDS = auto()
    POSTS = auto()
    TRANSLATION_GUIDELINE = auto()


class SSMParams(AutoNamedEnum):
    BATCH_JOB_DEFINITION = "batch-job-definition"
    BATCH_JOB_QUEUE = "batch-job-queue"
    GITHUB_TOKEN = "github-token"
    LANGCHAIN_API_KEY = "langchain-api-key"
    UPSTAGE_API_KEY = "upstage-api-key"


class AppConstants:
    NULL_STRING: str = "null"

    class External(str, Enum):
        ARXIV_HTML = "https://arxiv.org/html"
        AR5IV_LABS_HTML = "https://ar5iv.labs.arxiv.org/html"
        ARXIV_PDF = "https://arxiv.org/pdf"
        DOI_ORG = "https://doi.org"
        UPSTAGE_DOCUMENT_PARSE = "https://api.upstage.ai/v1/document-ai/document-parse"
