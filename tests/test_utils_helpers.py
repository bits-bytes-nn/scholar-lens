"""Tests for utils.helpers and utils.parsers (pure logic, no AWS)."""

from __future__ import annotations

import asyncio

import pytest

from scholar_lens.src.utils.helpers import (
    arg_as_bool,
    extract_text_from_html,
    measure_execution_time,
)
from scholar_lens.src.utils.parsers import HTMLTagOutputParser


class TestArgAsBool:
    @pytest.mark.parametrize("value", ["yes", "true", "1", "y", "t", "TRUE", "Yes"])
    def test_truthy(self, value: str) -> None:
        assert arg_as_bool(value) is True

    @pytest.mark.parametrize("value", ["no", "false", "0", "n", "f", "FALSE"])
    def test_falsy(self, value: str) -> None:
        assert arg_as_bool(value) is False

    def test_passthrough_bool(self) -> None:
        assert arg_as_bool(True) is True
        assert arg_as_bool(False) is False

    def test_invalid_raises(self) -> None:
        with pytest.raises(Exception):
            arg_as_bool("maybe")


class TestExtractTextFromHtml:
    def test_strips_tags_and_scripts(self) -> None:
        html = "<p>Hi <b>there</b></p><script>evil()</script>"
        assert extract_text_from_html(html) == "Hi there"

    def test_empty_input(self) -> None:
        assert extract_text_from_html("") == ""

    def test_collapses_whitespace(self) -> None:
        assert extract_text_from_html("<p>a\n\n   b</p>") == "a b"

    def test_keeps_block_structure(self) -> None:
        html = "<h2>2 Method</h2><p>First.</p><p>Second.</p><ul><li>one</li></ul>"
        assert extract_text_from_html(html) == (
            "## 2 Method\n\nFirst.\n\nSecond.\n\n- one"
        )

    def test_drops_page_chrome(self) -> None:
        html = "<nav>Docs Home Pricing</nav><header>Site</header><p>Body</p><footer>©</footer>"
        assert extract_text_from_html(html) == "Body"

    def test_pre_becomes_fenced_code_with_indentation(self) -> None:
        out = extract_text_from_html("<pre>def f(x):\n    return x</pre>")
        assert out == "```\ndef f(x):\n    return x\n```"

    def test_backslash_n_in_latex_and_code_survives(self) -> None:
        # Regression: a blanket "\\n" -> " " replacement turned \nabla into "abla".
        html = (
            '<p><math alttext="\\nabla_{\\theta} \\nu \\neq 0"></math></p>'
            "<code>print('a\\nb')</code>"
        )
        out = extract_text_from_html(html)
        assert "\\nabla_{\\theta} \\nu \\neq 0" in out
        assert "print('a\\nb')" in out

    def test_math_uses_alttext_with_blog_delimiters(self) -> None:
        html = (
            '<p>Loss <math alttext="\\mathcal{L}=\\sum_i x_i" display="inline">'
            "<mi>L</mi><annotation>junk</annotation></math> holds.</p>"
            '<math alttext="a+b" display="block"><mi>a</mi></math>'
        )
        out = extract_text_from_html(html)
        assert "Loss $$\\mathcal{L}=\\sum_i x_i$$ holds." in out
        assert "\n\n$$ a+b $$" in out
        assert "\\(" not in out and "junk" not in out

    def test_math_falls_back_to_tex_annotation(self) -> None:
        html = (
            "<math><mi>x</mi>"
            '<annotation encoding="application/x-tex">x^2</annotation></math>'
        )
        assert extract_text_from_html(html) == "$$x^2$$"

    def test_fragment_links_keep_text_only(self) -> None:
        html = (
            '<p>see <a href="#bib.12">[12]</a> and <a href="https://x.dev">docs</a></p>'
        )
        assert extract_text_from_html(html) == "see [12] and docs (https://x.dev)"

    def test_image_marker_alt_is_single_line(self) -> None:
        html = '<img alt="Mixture of\nExperts" src="3.png">'
        assert (
            extract_text_from_html(html) == "[Image: alt=Mixture of Experts, src=3.png]"
        )


def _table_lines(html: str) -> list[str]:
    return [
        line
        for line in extract_text_from_html(html).splitlines()
        if line.startswith("|")
    ]


class TestTableToMarkdown:
    def test_two_row_header_with_colspan_and_rowspan_merges(self) -> None:
        html = (
            "<table><thead>"
            '<tr><th rowspan="2">Model</th><th colspan="2">AIME 2024</th><th>MATH-500</th></tr>'
            "<tr><th>pass@1</th><th>cons@64</th><th>pass@1</th></tr>"
            "</thead><tbody><tr><td>o1-mini</td><td>63.6</td><td>80.0</td><td>90.0</td></tr>"
            "</tbody></table>"
        )
        assert _table_lines(html) == [
            "| Model | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 |",
            "| --- | --- | --- | --- |",
            "| o1-mini | 63.6 | 80.0 | 90.0 |",
        ]

    def test_header_rows_detected_without_thead(self) -> None:
        # Document-parse output uses <td> for headers: leading rows with no
        # numeric cell are the header.
        html = (
            "<table><tr><td></td><td>Active</td><td colspan='2'>French</td></tr>"
            "<tr><td>Model</td><td>Params</td><td>Arc-c</td><td>MMLU</td></tr>"
            "<tr><td>Mixtral 8x7B</td><td>13B</td><td>58.2%</td><td>70.9%</td></tr></table>"
        )
        assert _table_lines(html)[0] == (
            "| Model | Active Params | French Arc-c | French MMLU |"
        )

    def test_body_rowspan_repeats_and_colspan_does_not(self) -> None:
        html = (
            "<table><tr><th>Group</th><th>A</th><th>B</th></tr>"
            "<tr><td rowspan='2'>G</td><td colspan='2'>63.6</td></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
        )
        assert _table_lines(html)[2:] == ["| G | 63.6 | - |", "| G | 1 | 2 |"]

    def test_every_row_has_header_width_and_pipes_escaped(self) -> None:
        html = (
            "<table><tr><th>Metric</th><th>Value</th><th>Note</th></tr>"
            "<tr><td>|x|</td><td>3</td></tr></table>"
        )
        lines = _table_lines(html)
        assert lines[2] == "| \\|x\\| | 3 | - |"
        assert len({line.count(" | ") for line in lines}) == 1

    def test_text_only_table_keeps_body_rows(self) -> None:
        html = (
            "<table><tr><td>Method</td><td>Idea</td></tr>"
            "<tr><td>LoRA</td><td>low-rank update</td></tr></table>"
        )
        assert _table_lines(html)[2] == "| LoRA | low-rank update |"

    def test_caption_kept_above_table(self) -> None:
        html = "<table><caption>Table 4: Results</caption><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        assert extract_text_from_html(html).startswith("Table 4: Results\n\n| A |")

    def test_mixtral_fixture_tables_are_rectangular(self) -> None:
        import json
        import re
        from pathlib import Path

        fixture = (
            Path(__file__).resolve().parents[1] / "papers/2401_04088v1/parsed.json"
        )
        if not fixture.exists():
            pytest.skip("local Mixtral parse fixture not present")
        html = json.loads(fixture.read_text(encoding="utf-8"))["content"]["html"]
        tables = re.findall(r"<table.*?</table>", html, re.S)
        assert tables
        for table in tables:
            lines = _table_lines(table)
            widths = {len(re.split(r"(?<!\\)\|", line)) for line in lines}
            assert len(widths) == 1, table[:200]
            header_cells = [c.strip() for c in re.split(r"(?<!\\)\|", lines[0])[2:-1]]
            assert all(header_cells), lines[0]


class TestHTMLTagOutputParser:
    def test_multiple_tags_returns_dict(self) -> None:
        parser = HTMLTagOutputParser(tag_names=["title", "body"])
        assert parser.parse("<title>Hello</title><body>World</body>") == {
            "title": "Hello",
            "body": "World",
        }

    def test_single_tag_returns_string(self) -> None:
        parser = HTMLTagOutputParser(tag_names="answer")
        assert parser.parse("noise <answer>42</answer> noise") == "42"

    def test_missing_tag_is_omitted(self) -> None:
        # The parser only emits keys for tags actually present in the output.
        parser = HTMLTagOutputParser(tag_names=["present", "absent"])
        result = parser.parse("<present>x</present>")
        assert result["present"] == "x"
        assert "absent" not in result


class TestMeasureExecutionTime:
    def test_decorates_sync_function(self) -> None:
        @measure_execution_time
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_decorates_async_function(self) -> None:
        @measure_execution_time
        async def aadd(a: int, b: int) -> int:
            return a + b

        assert asyncio.run(aadd(2, 3)) == 5


class TestHTMLTagOutputParserRepeatedEnvelope:
    def test_self_corrected_output_uses_the_last_block(self) -> None:
        from scholar_lens.src.utils.parsers import HTMLTagOutputParser

        raw = (
            "<category>Finetuning</category>\n<controlled_tags>Language-Models</controlled_tags>\n"
            "Note: that tag is not in the list, corrected output:\n"
            "<category>Finetuning</category>\n<controlled_tags>Quantization</controlled_tags>"
        )
        parsed = HTMLTagOutputParser(tag_names=["category", "controlled_tags"]).parse(
            raw
        )
        assert parsed == {"category": "Finetuning", "controlled_tags": "Quantization"}

    def test_single_block_keeps_stray_close_tag_inside_body(self) -> None:
        from scholar_lens.src.utils.parsers import HTMLTagOutputParser

        raw = "<summary>Use </summary> literally in prose.</summary>"
        parsed = HTMLTagOutputParser(tag_names="summary").parse(raw)
        assert parsed == "Use </summary> literally in prose."


class TestTableAndEquationFidelity:
    """Round-2 regressions in the HTML → text serializer."""

    def test_latexml_equation_table_becomes_display_math(self) -> None:
        html = (
            '<table class="ltx_equation ltx_eqn_table"><tr><td></td>'
            '<td><math alttext="\\|W\\|_{F}=\\sqrt{x}" display="block"></math></td>'
            '<td class="ltx_eqn_eqno">(1)</td></tr></table>'
        )
        out = extract_text_from_html(html)
        assert out == "$$ \\|W\\|_{F}=\\sqrt{x} $$"
        assert "(1)" not in out and "| ---" not in out

    def test_text_only_table_keeps_one_header_row(self) -> None:
        rows = "".join(
            f"<tr><td>{a}</td><td>{b}</td></tr>"
            for a, b in [
                ("Hyper", "Value"),
                ("optimizer", "AdamW"),
                ("schedule", "cosine"),
                ("warmup", "linear"),
            ]
        )
        out = extract_text_from_html(f"<table>{rows}</table>").splitlines()
        assert out[0] == "| Hyper | Value |"
        assert "| optimizer | AdamW |" in out

    def test_pipes_inside_math_cells_keep_their_meaning(self) -> None:
        html = (
            "<table><tr><th>Model</th><th>Norm</th></tr>"
            '<tr><td>A | B</td><td><math alttext="|x|+\\|W\\|"></math></td></tr></table>'
        )
        out = extract_text_from_html(html)
        assert "| A \\| B |" in out
        assert "$$\\vert x\\vert +\\Vert W\\Vert $$" in out

    def test_nested_table_is_flattened_into_its_cell(self) -> None:
        html = (
            "<table><tr><td>Outer</td><td><table><tr><td>a</td><td>b</td></tr>"
            "<tr><td>c</td><td>d</td></tr></table></td></tr>"
            "<tr><td>x</td><td>1</td></tr></table>"
        )
        assert "| Outer | a / b; c / d |" in extract_text_from_html(html)

    def test_article_header_and_aside_are_kept(self) -> None:
        html = (
            "<body><header>Site nav</header><article><header><h1>Doc Title</h1>"
            "</header><aside>Note: keep</aside><p>Body</p></article></body>"
        )
        out = extract_text_from_html(html)
        assert "Site nav" not in out
        assert "## Doc Title" in out and "Note: keep" in out
