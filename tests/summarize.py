import sys
from pathlib import Path
from pprint import pformat

import boto3

sys.path.append(str(Path(__file__).parent.parent))
from scholar_lens.configs import Config
from scholar_lens.src import (
    CitationSummarizer,
    LanguageModelId,
    LocalPaths,
    logger,
)

ARXIV_ID: str = "2302.13971"
REFERENCE_IDENTIFIERS: list[str] = [
    "Attention is all you need",
    "Root mean square layer normalization",
    "2002.05202",
    "2104.09864",
    "Jonn Doe's attention",
]
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


def main(arxiv_id: str, reference_identifiers: list[str], original_content: str):
    config = Config.load()
    profile_name = config.resources.profile_name
    boto_session = boto3.Session(
        profile_name=profile_name,
        region_name=config.resources.bedrock_region_name,
    )

    paper_dir = (
        Path(__file__).parent.parent
        / LocalPaths.PAPERS_DIR
        / arxiv_id.replace(".", "_")
    )

    summarizer = CitationSummarizer(
        LanguageModelId(config.citations.citation_summarization_model_id),
        LanguageModelId(config.citations.citation_analysis_model_id),
        boto_session,
        paper_dir=paper_dir,
    )

    citation_summaries = summarizer.summarize(reference_identifiers, original_content)
    logger.debug("Citation summaries: '%s'", pformat(citation_summaries))


if __name__ == "__main__":
    main(ARXIV_ID, REFERENCE_IDENTIFIERS, CONTENT)
