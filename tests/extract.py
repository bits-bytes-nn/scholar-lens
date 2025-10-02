import asyncio
import json
import sys
from pathlib import Path
from pprint import pformat

import boto3

sys.path.append(str(Path(__file__).parent.parent))
from scholar_lens.configs import Config
from scholar_lens.src import ContentExtractor, LocalPaths, logger

ARXIV_ID: str = "2302.13971"


async def main(arxiv_id: str):
    config = Config.load()
    profile_name = config.resources.profile_name
    default_boto_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.default_region_name
    )
    bedrock_boto_session = boto3.Session(
        profile_name=profile_name, region_name=config.resources.bedrock_region_name
    )

    root_dir = Path(__file__).parent.parent
    paper_dir = root_dir / LocalPaths.PAPERS_DIR.value / arxiv_id.replace(".", "_")
    with open(paper_dir / LocalPaths.CONTENT_FILE.value, "r", encoding="utf-8") as f:
        content = json.load(f)

    content_extractor = await ContentExtractor.create(
        config.paper.citation_extraction_model_id,
        config.paper.attributes_extraction_model_id,
        default_boto_session,
        bedrock_boto_session,
        root_dir=root_dir,
        bucket_name=config.resources.s3_bucket_name,
        prefix=config.resources.s3_prefix,
    )

    citations = await content_extractor.extract_citations(content.text)
    attributes = await content_extractor.extract_attributes(content.text)
    table_of_contents = await content_extractor.extract_table_of_contents(content.text)

    logger.debug("Extracted citations: '%s'", pformat(citations))
    logger.debug("Extracted attributes: '%s'", pformat(attributes))
    logger.debug("Extracted table of contents: '%s'", pformat(table_of_contents))


if __name__ == "__main__":
    asyncio.run(main(ARXIV_ID))
