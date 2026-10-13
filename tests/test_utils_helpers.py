"""Tests for utils.helpers and utils.parsers (pure logic, no AWS)."""

from __future__ import annotations

import asyncio

import pytest

from scholar_lens.src.utils.helpers import (
    extract_text_from_html,
    measure_execution_time,
)
from scholar_lens.src.utils.parsers import HTMLTagOutputParser


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
            extract_text_from_html(html)
            == '[Image: alt="Mixture of Experts", src="3.png"]'
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

    def test_body_spans_repeat_their_value(self) -> None:
        # A shared value ("Warmup Ratio 0.06" over every dataset) applies to
        # each spanned column; "-" would read as "no value".
        html = (
            "<table><tr><th>Group</th><th>A</th><th>B</th></tr>"
            "<tr><td rowspan='2'>G</td><td colspan='2'>63.6</td></tr>"
            "<tr><td>1</td><td>2</td></tr></table>"
        )
        assert _table_lines(html)[2:] == ["| G | 63.6 | 63.6 |", "| G | 1 | 2 |"]

    def test_full_width_label_row_is_not_repeated(self) -> None:
        html = (
            "<table><tr><th>Model</th><th>A</th><th>B</th></tr>"
            "<tr><td colspan='3'>Encoder models</td></tr>"
            "<tr><td>BERT</td><td>1</td><td>2</td></tr></table>"
        )
        assert _table_lines(html)[2] == "| Encoder models | - | - |"

    def test_first_row_rowspan_marks_a_two_row_header(self) -> None:
        # "0513" looks numeric, but "Benchmark" spans both header rows.
        html = (
            "<table><tr><td rowspan='2'>Benchmark</td><td>Claude-3.5-</td>"
            "<td>GPT-4o</td></tr>"
            "<tr><td>Sonnet-1022</td><td>0513</td></tr>"
            "<tr><td>MMLU</td><td>88.3</td><td>87.2</td></tr></table>"
        )
        lines = _table_lines(html)
        assert lines[0] == "| Benchmark | Claude-3.5- Sonnet-1022 | GPT-4o 0513 |"
        assert lines[2:] == ["| MMLU | 88.3 | 87.2 |"]

    def test_rowspan_does_not_pull_a_section_row_into_the_header(self) -> None:
        # "Size" spans three rows, but the second is a section label.
        html = (
            "<table><tr><td>Model</td><td rowspan='3'>Size</td><td>GSM8K</td></tr>"
            "<tr><td colspan='2'>Closed-Source Base Model</td></tr>"
            "<tr><td>Minerva</td><td>58.8</td></tr></table>"
        )
        lines = _table_lines(html)
        assert lines[0] == "| Model | Size | GSM8K |"
        assert "Closed-Source Base Model" not in lines[0]

    def test_spanning_sub_header_inside_thead_stays_in_the_header(self) -> None:
        html = (
            "<table><thead><tr><th rowspan='2'>Model</th><th colspan='2'>AIME</th>"
            "</tr><tr><th colspan='2'>pass@1</th></tr></thead>"
            "<tbody><tr><td>R1</td><td>79.8</td><td>80.1</td></tr></tbody></table>"
        )
        lines = _table_lines(html)
        assert lines[0] == "| Model | AIME pass@1 | AIME pass@1 |"
        assert lines[2:] == ["| R1 | 79.8 | 80.1 |"]

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


class TestLatexmlStructures:
    """LaTeXML apparatus seen on real arXiv HTML pages (R1, Transformer, LoRA)."""

    def test_aligned_rows_stay_one_equation(self) -> None:
        def row(lhs: str, rhs: str, number: str) -> str:
            return (
                '<tr class="ltx_equation ltx_eqn_row">'
                '<td class="ltx_eqn_cell ltx_eqn_center_padleft"></td>'
                f'<td class="ltx_eqn_cell"><math alttext="{lhs}"></math></td>'
                f'<td class="ltx_eqn_cell"><math alttext="{rhs}"></math></td>'
                '<td class="ltx_eqn_cell ltx_eqn_center_padright"></td>'
                f'<td class="ltx_eqn_cell ltx_eqn_eqno">({number})</td></tr>'
            )

        html = (
            '<table class="ltx_equationgroup ltx_eqn_align ltx_eqn_table">'
            + row("\\mathrm{MultiHead}(Q,K,V)", "=\\mathrm{Concat}(h_1)W^{O}", "1")
            + row("h_i", "=\\mathrm{Attention}(Q)", "2")
            + "</table>"
        )
        assert extract_text_from_html(html) == (
            "$$ \\begin{aligned} \\mathrm{MultiHead}(Q,K,V) & "
            "=\\mathrm{Concat}(h_1)W^{O} \\\\ h_i & =\\mathrm{Attention}(Q) "
            "\\end{aligned} $$"
        )

    def test_single_row_split_cells_are_joined(self) -> None:
        html = (
            '<table class="ltx_equation ltx_eqn_table"><tr>'
            '<td class="ltx_eqn_cell"><math alttext="a"></math></td>'
            '<td class="ltx_eqn_cell"><math alttext="=b\\$"></math></td></tr></table>'
        )
        assert extract_text_from_html(html) == "$$ a =b\\$ $$"

    def test_listing_is_fenced_line_by_line_without_download_blob(self) -> None:
        html = (
            '<div class="ltx_listing ltx_lstlisting"><div class="ltx_listing_data">'
            '<a download="x.py" href="data:text/plain;base64,QUFB">⬇</a></div>'
            '<div class="ltx_listingline"><span class="ltx_tag ltx_tag_listingline">1'
            "</span><span>##</span><span>\xa0</span><span>Question</span></div>"
            '<div class="ltx_listingline"><span>def</span><span>\xa0</span>'
            "<span>f():</span></div>"
            '<div class="ltx_listingline"><span>\xa0\xa0\xa0\xa0return</span>'
            "<span>\xa0</span><span>1</span></div></div>"
        )
        assert extract_text_from_html(html) == (
            "```\n## Question\ndef f():\n    return 1\n```"
        )

    def test_algorithm_keeps_step_numbers_and_math(self) -> None:
        # LaTeXML renders ``algorithmic`` as a listing without ``ltx_lstlisting``.
        html = (
            '<div class="ltx_listing ltx_listing">'
            '<div class="ltx_listingline"><span class="ltx_tag ltx_tag_listingline">'
            "1:</span> <span>policy model </span>"
            '<math alttext="\\pi_{\\theta}\\leftarrow\\pi_{0}"><semantics><mi>π</mi>'
            '<annotation encoding="application/x-tex">\\pi_{\\theta}</annotation>'
            "</semantics></math></div>"
            '<div class="ltx_listingline"><span class="ltx_tag ltx_tag_listingline">'
            "2:</span> <b>for</b> step = 1 <b>do</b></div></div>"
        )
        assert extract_text_from_html(html) == (
            "1: policy model $$\\pi_{\\theta}\\leftarrow\\pi_{0}$$\n2: for step = 1 do"
        )

    def test_listing_figure_keeps_its_caption(self) -> None:
        html = (
            '<figure class="ltx_float ltx_lstlisting"><figcaption>Listing 1: '
            "Prompt.</figcaption>"
            '<span class="ltx_listing ltx_lstlisting">'
            '<span class="ltx_listingline"><span class="ltx_tag ltx_tag_listingline">'
            "1</span><span>if</span></span>\n"
            '<span class="ltx_listingline">\n  <span class="ltx_text ltx_lst_space">'
            "    </span><span>x\xa0=\xa01</span>\n</span></span></figure>"
        )
        out = extract_text_from_html(html)
        assert "Listing 1: Prompt." in out and "```\nif\n    x = 1\n```" in out

    def test_latexml_items_keep_text_on_the_bullet_line(self) -> None:
        bullet = (
            '<ul class="ltx_itemize"><li class="ltx_item"><span class="ltx_tag '
            'ltx_tag_item">•</span><div class="ltx_para"><p class="ltx_p">First.'
            "</p></div></li></ul>"
        )
        numbered = (
            '<ol class="ltx_enumerate"><li class="ltx_item"><span class="ltx_tag '
            'ltx_tag_item">(a)</span><div class="ltx_para"><p>One.</p></div></li></ol>'
        )
        assert extract_text_from_html(bullet) == "- First."
        assert extract_text_from_html(numbered) == "- (a) One."

    def test_inline_data_objects_and_mixed_case_data_links_are_dropped(
        self,
    ) -> None:
        html = (
            '<p>a <object type="image/svg+xml" data="DATA:image/svg+xml;base64,AA">'
            '</object><a href=" Data:text/plain,x">dl</a></p>'
        )
        assert extract_text_from_html(html) == "a dl"

    def test_data_links_are_dropped(self) -> None:
        html = '<p>See <a href="data:text/plain;base64,QUFB">file</a>.</p>'
        assert extract_text_from_html(html) == "See file."

    def test_footnote_reads_in_place_without_doubled_marks(self) -> None:
        html = (
            "<p>training all biases"
            '<span class="ltx_note ltx_role_footnote"><sup class="ltx_note_mark">2'
            '</sup><span class="ltx_note_outer"><span class="ltx_note_content">'
            '<sup class="ltx_note_mark">2</sup><span class="ltx_tag ltx_tag_note">2'
            "</span>They are negligible.</span></span></span> is cheap.</p>"
        )
        assert extract_text_from_html(html) == (
            "training all biases (footnote: They are negligible.) is cheap."
        )

    def test_image_sources_resolve_against_base_url(self) -> None:
        html = (
            '<img src="img/arch.png" alt="arch"><img data-src="/lazy.png" alt="lazy">'
        )
        out = extract_text_from_html(html, base_url="https://docs.x.io/guide/page")
        assert 'src="https://docs.x.io/guide/img/arch.png"]' in out
        assert 'src="https://docs.x.io/lazy.png"]' in out
        # Paper parsing keeps the raw src (markers are matched by file name).
        assert 'src="img/arch.png"]' in extract_text_from_html(html)


class TestListItemCode:
    def test_code_in_a_list_item_keeps_blank_lines_and_indentation(self) -> None:
        html = (
            "<ol><li><p>Define:</p><pre><code>def f():\n    x = 1\n\n    return x"
            "</code></pre></li></ol><p>Then   run.</p>"
        )
        out = extract_text_from_html(html)
        assert "def f():\n    x = 1\n\n    return x\n```" in out
        assert out.endswith("Then run.")

    def test_item_starting_with_code_opens_the_fence_on_its_own_line(self) -> None:
        html = (
            "<ul><li><pre><code>pip install x</code></pre></li></ul><p>After   it.</p>"
        )
        out = extract_text_from_html(html)
        assert "\n```\npip install x\n```" in out
        assert out.endswith("After it.")  # fence state did not flip


class TestDocsPageStructure:
    def test_comments_and_doctype_are_dropped(self) -> None:
        html = "<!DOCTYPE html><html><body><!-- nav.html prune --><p>Hello</p></body></html>"
        assert extract_text_from_html(html) == "Hello"

    def test_ordered_and_nested_lists_keep_numbers_and_depth(self) -> None:
        html = (
            "<ol start='3'><li>First<ul><li>Note A</li><li>Note B"
            "<ul><li>deep</li></ul></li></ul></li><li>Second</li></ol>"
        )
        out = extract_text_from_html(html)
        assert "3. First\n   - Note A\n   - Note B\n     - deep" in out
        assert "4. Second" in out

    def test_code_holding_a_backtick_fence_uses_tildes(self) -> None:
        html = "<pre>```python\nprint(1)\n```</pre><p>after   it</p>"
        out = extract_text_from_html(html)
        assert out.startswith("~~~\n```python\nprint(1)\n```\n~~~")
        assert out.endswith("after it")

    @pytest.mark.parametrize(
        "code,marker",
        [
            ("~~~\nx  =  1\n~~~", "```"),  # a tilde fence first in the code
            ("```py\na  = 1\n```\n~~~\nb\n~~~", "````"),  # both fence styles
        ],
    )
    def test_code_fence_cannot_be_closed_by_its_own_lines(
        self, code: str, marker: str
    ) -> None:
        out = extract_text_from_html(f"<pre>{code}</pre><p>after   it</p>")
        assert out == f"{marker}\n{code}\n{marker}\n\nafter it"

    @pytest.mark.parametrize(
        "code,span",
        [("```", "```` ``` ````"), ("a`b", "``a`b``"), ("plain", "`plain`")],
    )
    def test_inline_code_cannot_open_a_fence(self, code: str, span: str) -> None:
        html = f"<p><code>{code}</code> opens   a block</p><p>after   it</p>"
        out = extract_text_from_html(html)
        # The text after the span is still normalized prose, not code.
        assert out == f"{span} opens a block\n\nafter it"

    def test_item_paragraphs_stay_inside_nested_items(self) -> None:
        html = (
            "<ul><li><p>outer</p><ul><li><p>inner</p><p>inner b</p></li></ul>"
            "<p>outer b</p></li></ul><div>\n    stray   indent\n</div>"
        )
        out = extract_text_from_html(html)
        assert "- outer\n  - inner\n    inner b\n  outer b" in out
        assert out.endswith("\nstray indent")  # never an indented code block


class TestDocsAndLatexmlStructure:
    def test_latexml_span_table_becomes_a_gfm_table(self) -> None:
        # Regression: DeepSeekMath's results tables are spans, not <table>s,
        # and came out as one long line.
        html = (
            '<span class="ltx_tabular">'
            '<span class="ltx_tr">'
            '<span class="ltx_td ltx_rowspan ltx_rowspan_2">Corpus</span>'
            '<span class="ltx_td ltx_colspan ltx_colspan_2">English</span></span>'
            '<span class="ltx_tr"><span class="ltx_td">GSM8K</span>'
            '<span class="ltx_td">MATH</span></span>'
            '<span class="ltx_tr"><span class="ltx_td">MathPile</span>'
            '<span class="ltx_td">2.7%</span><span class="ltx_td">3.3%</span></span>'
            "</span>"
        )
        out = extract_text_from_html(html)
        assert "| Corpus | English GSM8K | English MATH |" in out
        assert "| MathPile | 2.7% | 3.3% |" in out

    def test_katex_visual_copy_is_dropped(self) -> None:
        html = (
            '<p>Update <span class="katex"><span class="katex-mathml">'
            '<math><semantics><annotation encoding="application/x-tex">'
            "\\Delta W</annotation></semantics></math></span>"
            '<span class="katex-html" aria-hidden="true">ΔW</span></span>.</p>'
        )
        assert extract_text_from_html(html) == "Update $$\\Delta W$$."

    def test_katex_html_only_output_is_kept(self) -> None:
        html = '<p>Loss <span class="katex"><span class="katex-html">x+1</span></span> here.</p>'
        assert "x+1" in extract_text_from_html(html)

    def test_relative_links_resolve_and_header_links_drop(self) -> None:
        html = (
            '<h2>Install<a class="headerlink" href="#install">#</a></h2>'
            '<p>See <a href="/docs/x/setup">setup</a>.</p>'
        )
        out = extract_text_from_html(html, base_url="https://docs.x.dev/docs/x/a")
        assert out == "## Install\n\nSee setup (https://docs.x.dev/docs/x/setup)."

    def test_whitespace_only_lines_are_paragraph_breaks(self) -> None:
        out = extract_text_from_html("<ul><li>a</li></ul>\n   \n<p>b</p>")
        assert "\n \n" not in out and out.endswith("b")
