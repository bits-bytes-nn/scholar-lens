import html
import re

from langchain_core.output_parsers import BaseOutputParser

# A trailing run of closing tags at the very end of extracted content, e.g. a
# stray "</path></name>" the model appended after the real body. We only strip
# these when they are ORPHANS (no matching opening tag earlier in the content),
# so legitimate prose ending in "...defined by </response>" is preserved.
_TRAILING_CLOSE_TAGS = re.compile(r"(?:\s*</([a-zA-Z][a-zA-Z0-9_]*)>)\s*$")


class HTMLTagOutputParser(BaseOutputParser):
    tag_names: str | list[str]

    @staticmethod
    def _strip_orphan_close_tags(content: str) -> str:
        """Drop trailing closing tags that have no matching opening tag.

        Repeatedly removes a single trailing ``</tag>`` only when no ``<tag``
        appears earlier in the text — i.e. it is structurally orphaned (the
        ``</path></name>`` envelope-leak case). A closing tag that pairs with an
        opening tag (real inline HTML the author intended) is left untouched.
        """
        while match := _TRAILING_CLOSE_TAGS.search(content):
            tag = match.group(1)
            head = content[: match.start()]
            # If an opening tag for this name exists earlier, it's not an orphan.
            if re.search(rf"<{re.escape(tag)}(\s|>|/)", head):
                break
            content = head
        return content

    @staticmethod
    def _clean(content: str) -> str:
        """Normalise extracted tag content for Markdown/plaintext output.

        The body is sliced out by string search (not HTML-parsed), so raw
        angle-bracket tokens — generics like ``Vec<u8>`` / ``HashMap<String,i32>``,
        comparisons like ``a < b``, code fences — survive verbatim. We still drop
        a trailing run of orphan closing tags the model appends, and unescape HTML
        entities (``&amp;`` -> ``&``, ``&lt;kbd&gt;`` -> ``<kbd>``) the model emits
        for genuinely intended inline markup.
        """
        content = HTMLTagOutputParser._strip_orphan_close_tags(content)
        return html.unescape(content).strip()

    @staticmethod
    def _extract_tag_body(text: str, tag_name: str) -> str | None:
        """Slice the body of ``<tag ...>…</tag>`` out of a model response.

        Extracts the wrapper's inner content by string search on the known tag
        name — NOT by HTML parsing — so angle-bracket tokens inside the body
        (generics, comparisons, code fences) survive untouched. Uses the LAST
        matching ``</tag>`` so a stray closing tag inside the body doesn't cut it
        short.
        """
        opens = list(
            re.finditer(rf"<{re.escape(tag_name)}(?:\s[^>]*)?>", text, re.IGNORECASE)
        )
        if not opens:
            return None
        close_pattern = re.compile(rf"</{re.escape(tag_name)}\s*>", re.IGNORECASE)
        # A model that "corrects itself" repeats the whole envelope; the last
        # complete block is the answer it settled on. Otherwise take the first
        # opening tag so a stray closing tag inside the body can't cut it short.
        candidates = opens[::-1] if len(opens) > 1 else opens
        for open_match in candidates:
            closes = list(close_pattern.finditer(text, open_match.end()))
            if closes:
                return text[open_match.end() : closes[-1].start()]
        return text[opens[0].end() :]

    def parse(self, text: str) -> str | dict[str, str]:
        if not text:
            return {} if isinstance(self.tag_names, list) else ""
        parsed: dict[str, str] = {}
        tag_list = (
            self.tag_names if isinstance(self.tag_names, list) else [self.tag_names]
        )
        for tag_name in tag_list:
            body = self._extract_tag_body(text, tag_name)
            if body is not None:
                parsed[tag_name] = self._clean(body)
        if isinstance(self.tag_names, list):
            return parsed
        return next(iter(parsed.values()), "")

    @property
    def _type(self) -> str:
        return "html_tag_output_parser"
