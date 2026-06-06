from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import BaseOutputParser

from ..constants import LanguageModelId
from ..logger import logger
from .factories import BedrockLanguageModelFactory
from .parsers import RobustXMLOutputParser


def create_robust_xml_output_parser(
    factory: BedrockLanguageModelFactory,
    enable_output_fixing: bool,
    output_fixing_model_id: LanguageModelId,
) -> BaseOutputParser:
    base_parser = RobustXMLOutputParser()
    if not enable_output_fixing:
        return base_parser

    try:
        fixing_llm = factory.get_model(model_id=output_fixing_model_id)
        logger.info(
            f"Created OutputFixingParser with model: '{output_fixing_model_id.value}'"
        )
        return OutputFixingParser.from_llm(parser=base_parser, llm=fixing_llm)
    except Exception as e:
        logger.error(
            f"Failed to create OutputFixingParser with model {output_fixing_model_id.value}: {e}"
        )
        raise RuntimeError(f"Failed to create OutputFixingParser: {e}") from e
