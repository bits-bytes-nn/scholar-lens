import asyncio
import sys
from pathlib import Path
from pprint import pformat

import boto3

sys.path.append(str(Path(__file__).parent.parent))
from scholar_lens.configs import Config
from scholar_lens.src import (
    CodeRetriever,
    EmbeddingModelId,
    LanguageModelId,
    LocalPaths,
    NoPythonFilesError,
    logger,
)

ARXIV_ID: str = "2302.13971"
REPO_URLS: list[str] | str = "https://github.com/meta-llama/llama"
CONTENT: str = """
2.2 Architecture
Following recent work on large language models,
our network is based on the transformer architec-
ture (Vaswani et al., 2017). We leverage various
improvements that were subsequently proposed,
and used in different models such as PaLM. Here
are the main difference with the original architec-
ture, and where we were found the inspiration for
this change (in bracket):
Pre-normalization [GPT3]. To improve the
training stability, we normalize the input of each
transformer sub-layer, instead of normalizing the
output. We use the RMSNorm normalizing func-
tion, introduced by Zhang and Sennrich (2019).
SwiGLU activation function [PaLM]. We re-
place the ReLU non-linearity by the SwiGLU ac-
tivation function, introduced by Shazeer (2020) to
improve the performance. We use a dimension of
2 / 3 4d instead of 4d as in PaLM.
Rotary Embeddings [GPTNeo]. We remove the
absolute positional embeddings, and instead, add
rotary positional embeddings (RoPE), introduced
by Su et al. (2021), at each layer of the network.
The details of the hyper-parameters for our dif-
ferent models are given in Table 2.
"""


async def main(arxiv_id: str, repo_urls: list[str] | str, content: str):
    config = Config.load()
    profile_name = config.resources.profile_name
    boto_session = boto3.Session(
        profile_name=profile_name,
        region_name=config.resources.bedrock_region_name,
    )
    normalized_repo_urls = [repo_urls] if isinstance(repo_urls, str) else repo_urls

    paper_dir = (
        Path(__file__).parent.parent
        / LocalPaths.PAPERS_DIR
        / arxiv_id.replace(".", "_")
    )

    code_retriever = CodeRetriever(
        LanguageModelId(config.code.code_analysis_model_id),
        LanguageModelId(config.code.code_summarization_model_id),
        EmbeddingModelId(config.code.embed_model_id),
        paper_dir=paper_dir,
        boto_session=boto_session,
        chunk_size=config.code.chunk_size,
        chunk_overlap=config.code.chunk_overlap,
    )

    await code_retriever.download_repositories(normalized_repo_urls)

    try:
        await code_retriever.create_or_load_index()
    except NoPythonFilesError:
        logger.warning("No Python files found in repositories. Skipping code analysis.")
        return

    codebase_summary = await code_retriever.generate_codebase_summary()
    logger.debug("Codebase summary: '%s'", codebase_summary)

    results = await code_retriever.search_similar_code(content, k=10)
    logger.debug("Retrieved code: '%s'", pformat(results))

    await code_retriever.delete_index()


if __name__ == "__main__":
    asyncio.run(main(ARXIV_ID, REPO_URLS, CONTENT))
