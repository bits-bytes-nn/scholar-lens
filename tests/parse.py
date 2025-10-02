import asyncio
import json
import sys
from pathlib import Path
from pprint import pformat
import boto3

sys.path.append(str(Path(__file__).parent.parent))
from scholar_lens.configs import Config
from scholar_lens.src import (
    ArxivHandler,
    HTMLRichParser,
    LanguageModelId,
    LocalPaths,
    PDFParser,
    logger,
)

ARXIV_ID: str = "2302.13971"
PARSE_PDF: bool = True


async def main(arxiv_id: str, parse_pdf: bool):
    config = Config.load()
    profile_name = config.resources.profile_name
    boto_session = boto3.Session(
        profile_name=profile_name,
        region_name=config.resources.bedrock_region_name,
    )
    arxiv_handler = ArxivHandler()

    root_dir = Path(__file__).parent.parent
    papers_dir = (root_dir / LocalPaths.PAPERS_DIR.value).absolute()
    paper_dir = papers_dir / arxiv_id.replace(".", "_")
    paper_dir.mkdir(parents=True, exist_ok=True)

    parser_kwargs = {
        "figure_analysis_model_id": LanguageModelId(
            config.paper.figure_analysis_model_id
        ),
        "boto_session": boto_session,
    }

    if parse_pdf:
        logger.info("Parsing content from PDF for arXiv ID: '%s'", arxiv_id)
        figures_dir = (paper_dir / LocalPaths.FIGURES_DIR.value).absolute()
        figures_dir.mkdir(parents=True, exist_ok=True)

        paper_path = arxiv_handler.download_paper(arxiv_id, papers_dir)

        pdf_parser = PDFParser(**parser_kwargs)
        async with pdf_parser:
            figures, content = await pdf_parser.parse(paper_path, figures_dir)
    else:
        logger.info("Parsing content from HTML for arXiv ID: '%s'", arxiv_id)
        html_parser = HTMLRichParser(**parser_kwargs)
        async with html_parser:
            result = await html_parser.parse(arxiv_id)
            figures, content = result.figures, result.content

    logger.debug("Extracted figures: '%s'", pformat(figures))
    logger.debug("Extracted content text (first 100 chars): '%s'", content.text[:100])

    content_path = paper_dir / LocalPaths.CONTENT_FILE.value
    content_data = content.model_dump()
    await asyncio.to_thread(
        content_path.write_text,
        json.dumps(content_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved content to '%s'", content_path)

    figures_path = paper_dir / LocalPaths.FIGURES_FILE.value
    figures_data = [fig.model_dump(mode="json") for fig in figures]
    await asyncio.to_thread(
        figures_path.write_text,
        json.dumps(figures_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d figures data to '%s'", len(figures), figures_path)

    metadata = arxiv_handler.fetch_metadata(arxiv_id)
    logger.debug("Fetched metadata: '%s'", pformat(metadata))


if __name__ == "__main__":
    asyncio.run(main(ARXIV_ID, PARSE_PDF))
