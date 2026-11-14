from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from scholar_lens.domain.prompt_safety import register_prompt_templates

# Shared writing rules for the review, summary and guide prompts live in
# scholar_lens.domain.prompts.rules: each block states one concern once and
# prompts compose the blocks they need, so the rules cannot drift.


_CACHE_POINT: dict[str, Any] = {"cachePoint": {"type": "default"}}


@dataclass(frozen=True)
class BasePrompt(ABC):
    system_prompt_template: str
    human_prompt_template: str
    input_variables: list[str]
    output_variables: list[str] | None = None
    # Prompt caching for a prompt sent many times in a run (a section
    # writer's or checker's): whether its system prompt is cached, and the
    # tags of its human message after which a cache point goes (in order;
    # what precedes each stays the same across the calls that share it, a
    # run's or a section's). The first occurrence in the rendered message
    # counts (a value holding the tag moves the point, never the text); a
    # tag the message lacks places none, and a blank tail after the last
    # point is left out (Bedrock rejects a blank block).
    cache_system: ClassVar[bool] = False
    cache_after: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Untrusted input can never close one of this prompt's fences.
        register_prompt_templates(
            str(getattr(cls, "system_prompt_template", "") or ""),
            str(getattr(cls, "human_prompt_template", "") or ""),
        )

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


def with_cache_points(prompt_cls: type[BasePrompt]) -> Any:
    """What turns ``prompt_cls``'s rendered prompt into messages with its
    cache points (:attr:`BasePrompt.cache_system`, ``cache_after``): the text
    is unchanged, split into content blocks at the points."""

    def place(value: PromptValue) -> list[BaseMessage]:
        messages = value.to_messages()
        placed: list[BaseMessage] = []
        for message in messages:
            text = message.content
            if not isinstance(text, str):
                placed.append(message)
            elif isinstance(message, SystemMessage) and prompt_cls.cache_system:
                placed.append(SystemMessage(content=[{"text": text}, _CACHE_POINT]))
            elif isinstance(message, HumanMessage) and prompt_cls.cache_after:
                placed.append(HumanMessage(content=_split_at(text, prompt_cls)))
            else:
                placed.append(message)
        return placed

    return place


def _split_at(text: str, prompt_cls: type[BasePrompt]) -> list[str | dict[str, Any]]:
    blocks: list[str | dict[str, Any]] = []
    start = 0
    for tag in prompt_cls.cache_after:
        at = text.find(tag, start)
        if at < 0:
            continue
        end = at + len(tag)
        blocks += [{"text": text[start:end]}, _CACHE_POINT]
        start = end
    if text[start:].strip():
        blocks.append({"text": text[start:]})
    return blocks
