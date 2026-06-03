import re
from collections import defaultdict
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.output_parsers import BaseOutputParser, XMLOutputParser
from lxml import etree

from ..logger import logger


class HTMLTagOutputParser(BaseOutputParser):
    tag_names: str | list[str]

    def parse(self, text: str) -> str | dict[str, str]:
        if not text:
            return {} if isinstance(self.tag_names, list) else ""
        soup = BeautifulSoup(text, "html.parser")
        parsed: dict[str, str] = {}
        tag_list = (
            self.tag_names if isinstance(self.tag_names, list) else [self.tag_names]
        )
        for tag_name in tag_list:
            if tag := soup.find(tag_name):
                if hasattr(tag, "decode_contents"):
                    parsed[tag_name] = str(tag.decode_contents()).strip()
                else:
                    parsed[tag_name] = str(tag).strip()
        if isinstance(self.tag_names, list):
            return parsed
        return next(iter(parsed.values()), "")

    @property
    def _type(self) -> str:
        return "html_tag_output_parser"


class RobustXMLOutputParser(XMLOutputParser):
    def parse(self, text: str) -> dict[str, Any]:
        text = self._unescape_xml_tags(text)
        original_sections = self._detect_xml_sections(text)

        try:
            result = super().parse(text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in parsed result")
        except Exception as e:
            logger.debug(
                f"Standard XML parsing failed: {type(e).__name__}: {e}. Trying lxml recovery..."
            )

        try:
            cleaned_text = self._clean_xml_for_lxml(text)
            result = self._try_lxml_recover_parse(cleaned_text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in lxml result")
        except Exception as e:
            logger.debug(
                f"LXML recovery parsing failed: {type(e).__name__}: {e}. Trying sanitization..."
            )

        try:
            sanitized_text = self._sanitize_xml_content(text)
            result = super().parse(sanitized_text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in sanitized result")
        except Exception as e:
            logger.debug(
                f"Sanitized XML parsing failed: {type(e).__name__}: {e}. Trying truncated XML fix..."
            )

        try:
            fixed_text = self._fix_truncated_xml(text)
            result = super().parse(fixed_text)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in truncated fix result")
        except Exception as e:
            logger.debug(
                f"Truncated XML fix failed: {type(e).__name__}: {e}. Trying aggressive cleaning..."
            )

        try:
            aggressively_cleaned = self._aggressively_clean_xml(text)
            result = super().parse(aggressively_cleaned)
            if self._sections_preserved(original_sections, result):
                return result
            raise ValueError("Missing sections in aggressive result")
        except Exception as e:
            logger.debug(
                f"Aggressive cleaning parsing failed: {type(e).__name__}: {e}. Trying XML fallback..."
            )

        try:
            fallback_result = self._extract_xml_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                f"XML fallback extraction failed: {type(e).__name__}: {e}. Trying tags fallback..."
            )

        try:
            fallback_result = self._extract_tags_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                f"Tags fallback extraction failed: {type(e).__name__}: {e}. Trying list fallback..."
            )

        try:
            fallback_result = self._extract_list_fallback(text)
            if fallback_result:
                return fallback_result
        except Exception as e:
            logger.debug(
                f"List fallback extraction failed: {type(e).__name__}: {e}. All methods exhausted."
            )

        logger.error("All XML parsing attempts failed for content: '%s...'", text[:200])
        raise ValueError(
            f"Failed to parse XML after multiple attempts. Content preview: '{text[:200]}...'"
        )

    @staticmethod
    def _detect_xml_sections(text: str) -> set[str]:
        pattern = r"<([a-zA-Z0-9_]+)>.*?</\1>"
        matches = re.findall(pattern, text, re.DOTALL)
        return set(matches)

    @staticmethod
    def _sections_preserved(
        original_sections: set[str], parsed_result: dict[str, Any]
    ) -> bool:
        if not original_sections:
            return True

        parsed_sections = (
            set(parsed_result.keys()) if isinstance(parsed_result, dict) else set()
        )
        missing_sections = original_sections - parsed_sections

        if missing_sections:
            return False
        return True

    @staticmethod
    def _extract_xml_fallback(text: str) -> dict[str, Any] | None:
        result = {}

        try:
            section_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
            section_matches = re.findall(section_pattern, text, re.DOTALL)

            for section_name, section_content in section_matches:
                section_result = RobustXMLOutputParser._parse_xml_section(
                    section_content
                )
                if section_result is not None:
                    result[section_name] = section_result

            return result if result else None

        except Exception:
            return None

    @staticmethod
    def _parse_xml_section(
        content: str,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        content = content.strip()
        if not content:
            return None

        child_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
        child_matches = re.findall(child_pattern, content, re.DOTALL)

        if not child_matches:
            return {"#text": content}

        children_by_tag = defaultdict(list)
        for child_tag, child_content in child_matches:
            parsed_child = RobustXMLOutputParser._parse_xml_element(child_content)
            children_by_tag[child_tag].append(parsed_child)

        result = {}
        for tag, children in children_by_tag.items():
            result[tag] = children[0] if len(children) == 1 else children

        if len(children_by_tag) == 1:
            child_tag = list(children_by_tag.keys())[0]
            children = children_by_tag[child_tag]
            if len(children) > 1:
                return {child_tag: children}

        return result

    @staticmethod
    def _parse_xml_element(content: str) -> dict[str, Any] | str:
        content = content.strip()
        if not content:
            return ""

        nested_pattern = r"<([a-zA-Z0-9_]+)>(.*?)</\1>"
        nested_matches = re.findall(nested_pattern, content, re.DOTALL)

        if not nested_matches:
            return content

        result: dict[str, Any] = {}
        for nested_tag, nested_content in nested_matches:
            parsed_nested = RobustXMLOutputParser._parse_xml_element(nested_content)

            if nested_tag in result:
                if not isinstance(result[nested_tag], list):
                    result[nested_tag] = [result[nested_tag]]
                result[nested_tag].append(parsed_nested)
            else:
                result[nested_tag] = parsed_nested

        text_content = content
        for nested_tag, nested_content in nested_matches:
            full_nested = f"<{nested_tag}>{nested_content}</{nested_tag}>"
            text_content = text_content.replace(full_nested, "").strip()

        if text_content and result:
            result["#text"] = text_content
        elif text_content and not result:
            return text_content

        return result

    @staticmethod
    def _clean_xml_for_lxml(text: str) -> bytes:
        def is_valid_xml_char(char: str) -> bool:
            cp = ord(char)
            return (
                cp == 0x9
                or cp == 0xA
                or cp == 0xD
                or (0x20 <= cp <= 0xD7FF)
                or (0xE000 <= cp <= 0xFFFD)
                or (0x10000 <= cp <= 0x10FFFF)
            )

        cleaned = "".join(char for char in text if is_valid_xml_char(char))
        return cleaned.strip().encode("utf-8")

    @staticmethod
    def _try_lxml_recover_parse(xml_bytes: bytes) -> dict[str, Any]:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        tree = etree.fromstring(xml_bytes, parser=parser)

        if tree is None:
            raise ValueError("lxml parser recovered a null tree")

        def _convert_etree_to_dict(element: etree._Element) -> dict[str, Any]:
            result: dict[str, Any] = {}
            children = list(element)

            if children:
                child_dict = defaultdict(list)
                for child in children:
                    child_result = _convert_etree_to_dict(child)
                    for key, value in child_result.items():
                        child_dict[key].append(value)

                processed_children = {
                    key: val[0] if len(val) == 1 else val
                    for key, val in child_dict.items()
                }
                result[element.tag] = processed_children
            else:
                result[element.tag] = {}

            if element.attrib:
                if not isinstance(result[element.tag], dict):
                    result[element.tag] = {"#text": result[element.tag]}
                if isinstance(result[element.tag], dict):
                    result[element.tag].update(
                        {f"@{k}": v for k, v in element.attrib.items()}
                    )

            if element.text and element.text.strip():
                text = element.text.strip()
                if not result[element.tag]:
                    result[element.tag] = text
                elif isinstance(result[element.tag], dict):
                    if "#text" not in result[element.tag]:
                        result[element.tag]["#text"] = text

            if not result[element.tag]:
                result[element.tag] = {}

            return result

        return _convert_etree_to_dict(tree)

    @staticmethod
    def _sanitize_xml_content(xml_content: str) -> str:
        def escape_text_only(text: str) -> str:
            placeholders = {}
            entity_pattern = r"&(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);"
            counter = [0]

            def save_entity(match: re.Match) -> str:
                key = f"\x00ENTITY{counter[0]}\x00"
                placeholders[key] = match.group(0)
                counter[0] += 1
                return key

            text = re.sub(entity_pattern, save_entity, text)

            text = text.replace("&", "&amp;")
            text = text.replace("<", "&lt;")
            text = text.replace(">", "&gt;")

            for key, value in placeholders.items():
                text = text.replace(key, value)

            return text

        def escape_leaf_content(match: re.Match) -> str:
            tag_open = match.group(1)
            content = match.group(2)
            tag_close = match.group(3)

            if not re.search(r"<[a-zA-Z_]", content):
                escaped_content = escape_text_only(content)
                return f"{tag_open}{escaped_content}{tag_close}"
            return match.group(0)

        pattern = r"(<([a-zA-Z_][a-zA-Z0-9_]*)\s*[^>]*>)(.*?)(</\2>)"
        prev_content = ""
        while prev_content != xml_content:
            prev_content = xml_content
            xml_content = re.sub(
                pattern, escape_leaf_content, xml_content, flags=re.DOTALL
            )

        return xml_content

    @staticmethod
    def _aggressively_clean_xml(xml_content: str) -> str:
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", xml_content)

        def escape_text_between_tags(match: re.Match[str]) -> str:
            content = match.group(1)
            content = re.sub(
                r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)",
                "&amp;",
                content,
            )
            return f">{content}<"

        cleaned = re.sub(r">([^<]*)<", escape_text_between_tags, cleaned)
        return cleaned.strip()

    @staticmethod
    def _unescape_xml_tags(text: str) -> str:
        text = re.sub(
            r"&lt;([a-zA-Z_][a-zA-Z0-9_]*(?:\s+[^&]*?)?)&gt;",
            r"<\1>",
            text,
        )
        text = re.sub(
            r"&lt;/([a-zA-Z_][a-zA-Z0-9_]*)&gt;",
            r"</\1>",
            text,
        )
        return text

    @staticmethod
    def _fix_truncated_xml(text: str) -> str:
        opening_tags = re.findall(r"<([a-zA-Z_][a-zA-Z0-9_]*)\b[^/>]*>", text)
        closing_tags = re.findall(r"</([a-zA-Z_][a-zA-Z0-9_]*)>", text)

        tag_stack = []
        for tag in opening_tags:
            tag_stack.append(tag)
        for tag in closing_tags:
            if tag_stack and tag_stack[-1] == tag:
                tag_stack.pop()
            elif tag in tag_stack:
                idx = len(tag_stack) - 1 - tag_stack[::-1].index(tag)
                tag_stack.pop(idx)

        text = re.sub(r"<[a-zA-Z_][a-zA-Z0-9_]*\s+[^>]*$", "", text)

        for tag in reversed(tag_stack):
            text += f"</{tag}>"

        return text

    @staticmethod
    def _extract_tags_fallback(text: str) -> dict[str, Any] | None:
        pattern = re.compile(r"<([a-zA-Z0-9_]+)\s*.*?>(.*?)</\1>", re.DOTALL)
        matches = pattern.findall(text)

        if not matches:
            return None

        content_map = defaultdict(list)
        for tag, content in matches:
            stripped_content = content.strip()
            if stripped_content:
                content_map[tag].append(stripped_content)

        if not content_map:
            return None

        result = {
            key: val[0] if len(val) == 1 else val for key, val in content_map.items()
        }

        return result

    @staticmethod
    def _extract_list_fallback(text: str) -> dict[str, Any] | None:
        item_patterns = [
            r"^\s*[•\-\*]\s*(.+?)(?=\n\s*[•\-\*]|\Z)",
            r"^\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)",
        ]

        for pattern in item_patterns:
            items = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if items:
                stripped_items = [item.strip() for item in items if item.strip()]
                if stripped_items:
                    return {"items": stripped_items}

        return None
