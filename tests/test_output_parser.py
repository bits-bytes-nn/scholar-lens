"""HTMLTagOutputParser: entity unescaping and trailing-tag handling."""

from __future__ import annotations

from scholar_lens.src.utils.parsers import HTMLTagOutputParser

# --------------------------------------------------------------------------- #
# HTMLTagOutputParser
# --------------------------------------------------------------------------- #


def test_parser_unescapes_entities() -> None:
    p = HTMLTagOutputParser(tag_names="summary")
    assert p.parse("<summary>A &amp; B &gt; C &lt; D</summary>") == "A & B > C < D"


def test_parser_strips_trailing_orphan_tags() -> None:
    p = HTMLTagOutputParser(tag_names="summary")
    assert p.parse("<summary>real body</path></name></summary>") == "real body"


def test_parser_keeps_trailing_closing_tag_that_is_real_prose() -> None:
    # Regression guard: a closing-tag-like token in legitimate prose, with NO
    # matching opening tag, must be preserved (not silently stripped). Here the
    # content ends with an escaped </response> the author meant to show.
    p = HTMLTagOutputParser(tag_names="guide")
    out = p.parse("<guide>The end marker is &lt;/response&gt;</guide>")
    assert out == "The end marker is </response>"


def test_parser_keeps_paired_trailing_close_tag() -> None:
    # A closing tag that pairs with an earlier opening tag is intentional inline
    # HTML and must be kept.
    p = HTMLTagOutputParser(tag_names="guide")
    out = p.parse("<guide>see &lt;kbd&gt;Enter&lt;/kbd&gt;</guide>")
    assert out == "see <kbd>Enter</kbd>"


def test_parser_keeps_inline_markup_but_unescapes() -> None:
    p = HTMLTagOutputParser(tag_names="guide")
    # A markdown table header with an escaped ampersand must come back literal.
    out = p.parse("<guide>| Model &amp; Method | x |</guide>")
    assert out == "| Model & Method | x |"


def test_parser_dict_mode_multiple_tags() -> None:
    p = HTMLTagOutputParser(tag_names=["summary", "tags"])
    out = p.parse("<summary>body &amp; more</summary><tags>A, B</tags>")
    assert out == {"summary": "body & more", "tags": "A, B"}


def test_parser_preserves_generics_and_comparisons() -> None:
    # Regression (quality-0): the body is Markdown/code and may contain raw
    # angle-bracket tokens. It must NOT be HTML-parsed (which lowercased tag
    # names, injected junk attributes, and appended synthetic close tags).
    p = HTMLTagOutputParser(tag_names="summary")
    out = p.parse(
        "<summary>Returns a Vector<float> and a Map<str,int>; a < b holds.</summary>"
    )
    assert out == "Returns a Vector<float> and a Map<str,int>; a < b holds."
    assert "</float>" not in out
    assert "</str" not in out


def test_parser_preserves_code_fence_with_generics() -> None:
    p = HTMLTagOutputParser(tag_names="section_markdown")
    body = (
        "```rust\n"
        "let m: HashMap<String, i32> = HashMap::new();\n"
        "let v: Vec<u8> = vec![];\n"
        "```"
    )
    out = p.parse(f"<section_markdown>{body}</section_markdown>")
    assert out == body
    assert "HashMap<String, i32>" in out  # not lowercased, no injected attrs
    assert 'i32=""' not in out


def test_parser_preserves_list_generic_takeaway() -> None:
    p = HTMLTagOutputParser(tag_names="key_takeaways")
    out = p.parse("<key_takeaways>- Handles List<T> generics</key_takeaways>")
    assert out == "- Handles List<T> generics"
    assert "</t>" not in out.lower()
