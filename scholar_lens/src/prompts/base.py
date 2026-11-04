from abc import ABC
from dataclasses import dataclass

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ...prompts.rules import (  # noqa: F401 - moved; re-exported
    CITATION_KEY_RULES,
    DEPTH_RULES,
    EXCLUDED_CONTENT_RULES,
    HEADING_STRUCTURE_RULES,
    IMAGE_PATH_RULES,
    KOREAN_STYLE,
    MATH_RULES,
    NATURALNESS_CHECK,
    NUMBER_RULES,
    PAPER_TABLE_RULES,
    PROSE_UNIT,
    READER_PROFILE,
    STYLE_RULES,
    SUMMARY_DEPTH_RULES,
    SUPPLEMENTARY_CONTEXT_RULES,
    TABLE_RULES,
    TABLE_SIZE_RULE,
    TERMINOLOGY_RULES,
    VISUAL_RULES,
    blocking_issues_block,
)

# Shared writing rules for the review, summary and guide prompts. Each block
# states one concern once; prompts compose the blocks they need instead of
# restating them, so the rules cannot drift between artifacts.


@dataclass(frozen=True)
class BasePrompt(ABC):
    system_prompt_template: str
    human_prompt_template: str
    input_variables: list[str]
    output_variables: list[str] | None = None

    def __post_init__(self) -> None:
        self._validate_prompt_variables()

    def _validate_prompt_variables(self) -> None:
        if not self.input_variables:
            return
        for var in self.input_variables:
            if not isinstance(var, str) or not var:
                raise ValueError(f"Invalid input variable: {var}")
            if (
                var != "image_data"
                and f"{{{var}}}" not in self.human_prompt_template
                and f"{{{var}}}" not in self.system_prompt_template
            ):
                raise ValueError(
                    f"Input variable '{var}' not found in any prompt template."
                )

    @classmethod
    def get_prompt(cls) -> ChatPromptTemplate:
        instance = cls(
            input_variables=cls.input_variables,
            output_variables=cls.output_variables,
            system_prompt_template=cls.system_prompt_template,
            human_prompt_template=cls.human_prompt_template,
        )
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    instance.system_prompt_template
                ),
                HumanMessagePromptTemplate.from_template(
                    instance.human_prompt_template
                ),
            ]
        )
