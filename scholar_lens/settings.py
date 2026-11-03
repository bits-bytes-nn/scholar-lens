"""Process settings: the environment variables the app reads. (The
config.yaml model stays in scholar_lens.configs until the bootstrap step.)"""

from __future__ import annotations

from enum import Enum


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
