"""Prompts for the paper-review workflow: plan → write → check → edit → finalize.

The planner decides what the review covers and how deep each part goes; the
writer, checker and editor all work from that plan, so coverage, depth and
terminology stay consistent across sections.
"""

from .base import (
    CITATION_KEY_RULES,
    DEPTH_RULES,
    EXCLUDED_CONTENT_RULES,
    HEADING_STRUCTURE_RULES,
    IMAGE_PATH_RULES,
    MATH_RULES,
    NATURALNESS_CHECK,
    NUMBER_RULES,
    PAPER_TABLE_RULES,
    READER_PROFILE,
    STYLE_RULES,
    SUPPLEMENTARY_CONTEXT_RULES,
    TABLE_RULES,
    TABLE_SIZE_RULE,
    TERMINOLOGY_RULES,
    VISUAL_RULES,
    BasePrompt,
)


class PaperPlanningPrompt(BasePrompt):
    """Reads the whole paper once and plans the review (coverage, order,
    difficulty, prerequisites, concept ownership, figures, glossary)."""

    input_variables: list[str] = [
        "title",
        "numbered_sentences",
        "figures",
        "references",
        "codebase_summary",
        "translation_guideline",
        "repair_note",
        "language",
    ]

    system_prompt_template: str = """
    You are a senior AI/ML researcher planning an in-depth review of a paper for
    a technical blog. You decide what the review must cover, in which order, how
    hard each part is for the reader, and where background is needed, so that a
    writer can produce a complete, consistent article section by section.
    """ + READER_PROFILE

    human_prompt_template: str = (
        """
    Plan the review of "{title}". The review will be written in {language}.

    <numbered_sentences>
    {numbered_sentences}
    </numbered_sentences>

    <figures>
    {figures}
    </figures>

    <references>
    {references}
    </references>

    <codebase_summary>
    {codebase_summary}
    </codebase_summary>

    <translation_guideline>
    {translation_guideline}
    </translation_guideline>

    {repair_note}

    Build the plan:
    1. Coverage comes first. Every major element of the paper must appear in
       some section's must_cover: the problem and why existing approaches fall
       short; each contribution; every key equation or objective (named in
       words, with its role); algorithm or architecture steps; only the setup
       needed to read the results (data, baselines, the one or two settings
       that matter — never a hyper-parameter list); the headline results with
       the settings that produced them; ablations and analyses that support the
       claims. Be specific ("how the loss down-weights easy negatives"),
       never generic ("method details"). Give
       every must_cover item its own difficulty (see 3): the writer's length
       budget is computed from these.
    2. Order sections the way that best teaches the paper — usually problem,
       background the reader lacks, core method, design choices, experiments,
       analysis. Do not mirror the paper's sections one to one: merge thin
       parts, split dense ones, and fold material that is easy for the reader
       into a neighbouring section instead of giving it a section of its own.
       Do NOT plan a
       conclusion, limitations or future-work section: put the paper's stated
       limitations and future directions in closing_points instead (they go
       into the review's single closing assessment).
    3. Difficulty is judged for the reader described above: easy = material
       they already know; medium = new but straightforward; hard = the novel
       mechanism, dense math, non-obvious results, or prerequisites outside
       common knowledge (a main theorem: cover its statement and assumptions).
       Give a one-line reason for the section. Importance:
       core (the paper's contribution), supporting, or context.
    4. Prerequisites: concepts the reader needs but the paper assumes (a matrix
       decomposition, a statistical test, the prior method this one modifies).
       Set reference to an R<n> id only when the reference list contains it.
    5. owned_concepts: the concepts a section explains in full. Each concept has
       exactly one owner — the first section where the reader needs it; later
       sections only refer back to it. Include recurring caveats and baseline
       definitions (e.g. a comparison condition that differs between methods,
       what each baseline is) so they are explained once, not per section.
    6. figures: give each section the F<n> ids that genuinely help it; each
       figure goes to at most one section. Skip decorative figures.
    7. sentence_ranges: inclusive [start, end] numbers of the sentences each
       section draws on. Together with excluded_ranges they must cover the whole
       paper. Exclude only references, acknowledgments, author lists, funding
       and boilerplate; a substantive appendix belongs to a section.
    8. glossary: the key terms with the exact rendering to use throughout in
       {language}, per the terminology rules below. Include every compared
       baseline and method name.
    9. needs_code: true where code from the paper's repository would make the
       section concrete (implementation, algorithm, architecture).
    10. Titles are in {language}, descriptive, without numbers. purpose is one
        line saying what the reader gets from the section.

    Terminology rules:
    """
        + TERMINOLOGY_RULES
        + """
    Output only the plan as JSON inside <plan> tags:
    <plan>
    {{"narrative": "one paragraph: the story the review tells",
     "glossary": {{"term": "rendering"}},
     "excluded_ranges": [[0, 0]],
     "closing_points": ["a limitation or future direction the paper states"],
     "sections": [
      {{"title": "...", "purpose": "...", "sentence_ranges": [[0, 0]],
       "must_cover": [{{"item": "...", "difficulty": "easy | medium | hard"}}],
       "difficulty": "easy | medium | hard",
       "difficulty_reason": "...", "importance": "core | supporting | context",
       "prerequisites": [{{"concept": "...", "why": "...", "reference": "R1 or null"}}],
       "owned_concepts": ["..."], "figures": ["F1"], "needs_code": false}}
     ]}}
    </plan>
    """
    )


class PaperSectionPrompt(BasePrompt):
    """Writes (continues, or revises) one planned section of the review."""

    input_variables: list[str] = [
        "plan_outline",
        "glossary",
        "translation_guideline",
        "section_brief",
        "owned_elsewhere",
        "already_in_article",
        "source_text",
        "citation_summaries",
        "code",
        "previous_sections",
        "current_explanation",
        "draft_to_revise",
        "improvement_feedback",
        "language",
    ]

    # The rules are fixed text, so they live in the system prompt ahead of
    # everything that changes per call.
    system_prompt_template: str = (
        """
    You are an AI/ML researcher writing one section of an in-depth paper review
    for a technical blog. A plan fixes what each section covers; you make your
    section complete, accurate and genuinely easy to understand.
    """
        + READER_PROFILE
        + """
    How to write a section:
    - Start with exactly `## ` followed by the section title from the brief; use
      `###` for sub-parts when the section holds several distinct ideas. When
      current_explanation is not empty, continue right after it without
      repeating its heading.
    - Cover every must-cover item of the brief, in the order that builds
      understanding, at the depth its difficulty tag asks for. The source text
      is the authority on the paper: never contradict its methods, claims or
      numbers. Figure markers look like [Image: alt="…", src="…", caption="…"];
      the caption is an analysis of the figure for you: never copy or mention
      it, nor quote a value read off the figure. To show a figure, write a
      Markdown image `![alt](<src copied exactly>)`, never the marker.
    - The target length is for the whole section, spent mostly on the hard
      items; never drop a required item to stay short.
    - Explain each prerequisite in the brief briefly where it is first
      needed.
    - Concepts owned by earlier sections: refer back in a short phrase, never
      re-explain. Concepts owned by later sections: use them plainly if you
      must, unexplained.
    - Never show again anything already_in_article lists (count its callouts
      against the cap).
    - Use the glossary rendering for every glossary term, table cells
      included. Place only the figures listed in the brief.
    - Setup is given only as far as the reader needs it to read the results
      (tables hold results, never hyper-parameter lists); explain a metric in
      one sentence where it is first used.
    - A caveat about how far a result can be trusted gets at most one sentence,
      right beside that result. Never write a caveat, summary or conclusion
      `###` subsection; the review has one closing section for that.
    - If draft_to_revise is not empty, revise it per improvement_feedback:
      keep what works, fix what is named, return the complete section.
    - If the source text truly lacks the material for a must-cover item, do not
      invent it: name the item in <missing_source>.

    Depth:
    """
        + DEPTH_RULES
        + """
    Context and code:
    """
        + SUPPLEMENTARY_CONTEXT_RULES
        + CITATION_KEY_RULES
        + """
    Structure and visuals:
    """
        + HEADING_STRUCTURE_RULES
        + VISUAL_RULES
        + TABLE_RULES
        + PAPER_TABLE_RULES
        + IMAGE_PATH_RULES
        + """
    Math:
    """
        + MATH_RULES
        + """
    Style:
    """
        + STYLE_RULES
        + EXCLUDED_CONTENT_RULES
        + """
    Length: if the section runs very long, stop at a paragraph boundary and set
    has_more to y (the next turn continues); y ONLY when must-cover items
    remain unwritten, otherwise n.

    Output exactly:
    <explanation>
    [the section in Markdown]
    </explanation>
    <has_more>y or n</has_more>
    <missing_source>must-cover items the source lacks, one per line, or none</missing_source>
    """
    )

    human_prompt_template: str = """
Write the section described in the brief, in {language}.

<plan_outline>
{plan_outline}
</plan_outline>

<glossary>
{glossary}
</glossary>

<translation_guideline>
{translation_guideline}
</translation_guideline>

<section_brief>
{section_brief}
</section_brief>

<concepts_owned_elsewhere>
{owned_elsewhere}
</concepts_owned_elsewhere>

<already_in_article>
{already_in_article}
</already_in_article>

<source_text>
{source_text}
</source_text>

<citation_summaries>
{citation_summaries}
</citation_summaries>

<code_examples>
{code}
</code_examples>

<previous_sections>
{previous_sections}
</previous_sections>

<current_explanation>
{current_explanation}
</current_explanation>

<draft_to_revise>
{draft_to_revise}
</draft_to_revise>

<improvement_feedback>
{improvement_feedback}
</improvement_feedback>
"""


class PaperSectionCheckPrompt(BasePrompt):
    """Checks one drafted section against its brief and the source."""

    input_variables: list[str] = [
        "section_brief",
        "owned_elsewhere",
        "already_in_article",
        "glossary",
        "source_text",
        "length_report",
        "citation_summaries",
        "code",
        "draft",
        "unsupported_items",
        "language",
    ]
    output_variables: list[str] = [
        "quality_score",
        "missing_items",
        "must_fix",
        "improvement_feedback",
    ]

    system_prompt_template: str = (
        """
    You are a strict but fair technical editor. You check ONE drafted section of
    a paper review against its brief and the paper text it is based on, and give
    feedback the writer can apply directly.
    """
        + READER_PROFILE
        + """
    Check, in this order:
    1. Coverage: is every must-cover item of the brief present with the
       substance its difficulty needs (an equation explained, not just shown;
       a result with its setting)? List each missing or merely name-checked
       item.
    2. Depth for the reader: hard items built up as the depth rules require,
       with the listed prerequisites explained where needed; easy items short.
       Use the length report: far over target is a defect only when the excess
       is padding or easy material; far under target on hard items is a
       defect. Never ask to cut a required item.
    3. Fidelity: nothing contradicts the source; no invented results, links or
       code provenance (illustrative toy values are not results). Every
       comparison or causal claim holds for the conditions compared (a gap
       "explained" by a factor both sides share is wrong); no value read off a
       figure and no comment on the figure analysis, captions or their
       mismatch; nothing contradicts already_in_article. A
       formula from outside the paper is exactly right; derivation steps agree
       in signs and variable definitions; "all"/"모든" about a setting holds in
       every experiment. Numbers follow the rules below; "best" keeps
       the paper's qualifier ("among open models"); a claim credited to a
       cited work or method is the source's own.
    4. No repetition: nothing listed in already_in_article (headings,
       equations, tables, figures, code, callouts) is shown again; concepts owned
       elsewhere are not re-explained; nothing repeats inside the section.
    5. Consistency and flow: glossary terms used as given; figures placed
       where discussed and introduced; naturalness as below.
    6. The writing rules below.
    """
        + DEPTH_RULES
        + SUPPLEMENTARY_CONTEXT_RULES
        + VISUAL_RULES
        + TABLE_SIZE_RULE
        + CITATION_KEY_RULES
        + STYLE_RULES
        + NATURALNESS_CHECK
        + """
    Headings, math syntax, table shape and the callout cap are
    checked in code; do not score or list them.

    Score out of 100: coverage 30, depth calibration 25, fidelity 25,
    repetition, consistency and flow 20. Caps: any missing must-cover item caps
    the score at 69 — except items in unsupported_items that the source text
    truly lacks: do not list those as missing, do not cap for them and do not
    ask for them in the feedback (if the source does contain one, it is
    missing); an invented result presented as the paper's, a fabricated
    link or a modified image path caps it at 40. 90+ only when nothing
    actionable remains.

    In the feedback, put the most important fixes first, quote the passage each
    refers to, and say what to do instead; keep what already works.

    must_fix lists only text that is wrong as written and blocks the section
    regardless of score, one per line with the quoted passage and the fix
    (a suggestion, a preference or text you find correct goes in the feedback):
    - an equation, table, figure or code block listed in already_in_article
      shown again;
    - a derived number (ratio, reduction, difference) that is wrong when you
      recompute it from its stated inputs, or an own estimate presented as the
      paper's number;
    - a number that breaks a number rule (unit, %p, exponent as a share,
      another model's or cell's value, ablation setting as the setup, "all
      benchmarks" against a loss, an invented acronym expansion);
    - a comparison or causal claim the compared conditions do not support, a
      claim or method description stronger than the paper's ("inspired by X"
      written as "uses X"), a value read off a figure, or a comment on the
      figure analysis or captions;
    - a quantity whose meaning changed (a share of outputs written as a
      rate) or a formula's input reported as its result;
    - a wrong formula, a sign or variable that contradicts another step, or a
      setting claimed for all experiments that one of them does not use;
    - a blocking wording issue of the naturalness check.

    Output exactly:
    <quality_score>an integer from 0 to 100</quality_score>
    <missing_items>
    one missing must-cover item per line, copied from the brief, or "none"
    </missing_items>
    <must_fix>one blocking defect per line, or "none"</must_fix>
    <improvement_feedback>...</improvement_feedback>
    """
    )

    human_prompt_template: str = """
    Target language: {language}. Write the feedback in {language}.

    <section_brief>
    {section_brief}
    </section_brief>

    <concepts_owned_elsewhere>
    {owned_elsewhere}
    </concepts_owned_elsewhere>

    <already_in_article>
    {already_in_article}
    </already_in_article>

    <glossary>
    {glossary}
    </glossary>

    <source_text>
    {source_text}
    </source_text>

    <length_report>
    {length_report}
    </length_report>

    <citation_summaries>
    {citation_summaries}
    </citation_summaries>

    <code_examples>
    {code}
    </code_examples>

    <draft>
    {draft}
    </draft>

    <unsupported_items>
    {unsupported_items}
    </unsupported_items>
    """


class PaperEditPrompt(BasePrompt):
    """Edits the assembled review as a whole with targeted find/replace edits."""

    input_variables: list[str] = [
        "article",
        "paper_text",
        "coverage_plan",
        "glossary",
        "round_note",
        "max_edits",
        "language",
    ]
    output_variables: list[str] = ["edits", "missing_coverage"]

    system_prompt_template: str = """
    You are the final editor of a complete paper review written section by
    section. You fix what only shows at the level of the whole article —
    inconsistent terms, repeated explanations, rough transitions, gaps in
    coverage — with a few precise edits, and leave everything else untouched.
    """ + READER_PROFILE

    human_prompt_template: str = (
        """
    The review is written in {language}.

    <article>
    {article}
    </article>

    <coverage_plan>
    {coverage_plan}
    </coverage_plan>

    <glossary>
    {glossary}
    </glossary>

    <paper_text>
    {paper_text}
    </paper_text>

    {round_note}

    First list to yourself every explanation, caveat or definition that
    appears in more than one section; then make your edits.

    Find and fix:
    1. Terms rendered differently from the glossary or between sections
       (including method names in table cells).
    2. The same explanation, example, figure, table or equation given twice:
       keep the first full treatment, replace a later repeat with a short
       back-reference.
    3. Section openings that recap, announce ("이 절에서는 …"), or promise later
       content; abrupt jumps between sections.
    4. Coverage-plan items missing from the whole article: add the missing
       content in the section where it belongs, faithful to the paper text, at
       the depth the plan implies.
    5. Statements that contradict each other across sections (a number, a
       cause, which experiment a result comes from): keep the one the paper
       text supports.
    6. References to the paper's figure, table, theorem or section numbers:
       rephrase them descriptively.

    Each edit replaces one exact passage. <find> is copied character for
    character from the article (backslashes, $$ and all) and occurs there
    exactly once — a whole sentence or paragraph is safest; <replace> is the
    new text. To insert, set find to the sentence after which the new text goes
    and replace it with that sentence followed by the new text. Never change or
    remove a `## ` heading; a redundant `###` or `####` subsection (a repeated
    caveat, summary or conclusion) may be removed whole, heading included, when
    its point is already made elsewhere. Use at most {max_edits} edits, the most valuable first; if
    nothing needs fixing, output no <edit> blocks. Write new text in {language}
    under the article's rules:
    """
        + MATH_RULES
        + STYLE_RULES
        + """
    Output exactly (repeat the <edit> block per edit; no JSON, no escaping):
    <edit>
    <find>exact passage from the article</find>
    <replace>new text</replace>
    <reason>why</reason>
    </edit>
    <missing_coverage>
    coverage-plan items still missing after your edits, one per line, or "none"
    </missing_coverage>
    """
    )


class PaperFinalizationPrompt(BasePrompt):
    """Writes the TL;DR that opens the review, the closing assessment, the
    translated title used as the post subtitle and the search description."""

    input_variables: list[str] = [
        "explanation",
        "title",
        "glossary",
        "closing_points",
        "abstract",
        "repair_note",
        "language",
    ]
    output_variables: list[str] = [
        "subtitle",
        "description",
        "key_takeaways",
        "closing",
    ]

    system_prompt_template: str = """
    You are a senior AI/ML researcher and technical editor. Given a finished
    section-by-section review of a paper, you write the framing parts a busy
    reader needs: a TL;DR they can scan in a minute and a candid closing
    assessment. You only use facts that appear in the review.
    """

    human_prompt_template: str = (
        """
    Write in {language}. Paper title: {title}

    <review>
    {explanation}
    </review>

    <glossary>
    {glossary}
    </glossary>

    <paper_stated_limitations_and_future_work>
    {closing_points}
    </paper_stated_limitations_and_future_work>

    <abstract>
    {abstract}
    </abstract>

    {repair_note}

    1. Subtitle (inside <subtitle>): the paper title translated into
       {language}, keeping model and method names as they are; if {language}
       is the title's own language, or the translation would equal the title
       (e.g. a bare model name), a short descriptive subtitle instead.

    2. TL;DR (inside <key_takeaways>): exactly four `###` headings, each
       followed by a short answer: 2-4 sentences or 2-4 one-line bullets. Use these questions
       (a Korean post uses them verbatim; any other language translates them):
       ### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?
       ### 이 연구에서 제시하는 새로운 해결 방법은 무엇입니까?
       ### 제안된 방법은 어떻게 구현되었습니까?
       ### 이 연구의 결과가 가지는 의미는 무엇입니까?
       Be concrete: name the core mechanism, and in the last answer give the
       2-3 headline results with the setting that produced them and who
       benefits most; attribute each result to the experiment (model,
       benchmark) the review reports it for. No bullet walls, no restating the
       title. Every number in it is one the paper states, as the paper states
       it (a derived number stays in the body).

    3. Description (inside <description>): the post's search snippet, one plain
       text paragraph of 120-160 characters in {language}: no math, Markdown,
       links or tags. It opens with the model or method name as the paper
       writes it ("Mistral 7B는 …", "DPO는 …"), never with a generic opener
       ("대규모 언어 모델의 발전으로 …"), says what the paper does and its
       headline result, and may end with what the review covers ("… 구조와
       학습 과정을 정리합니다."). Numbers only from the TL;DR above or the
       paper's <abstract>.

    4. Closing assessment (inside <closing>): a `##` heading in {language}
       meaning "Limitations and Takeaways" (Korean: "## 한계와 시사점"), then
       2-3 short paragraphs or a few bullets, each point made once: about
       600-1,200 Hangul syllables in Korean (the same reading time in other
       languages), longer only when the paper has a whole future-directions
       section to cover:
       - strengths that matter in practice;
       - limitations: unstated assumptions, evaluation gaps, cost or
         engineering trade-offs, settings where the method may not hold —
         specific and evidence-based, not generic;
       - practical implications: when to use it and when not;
       - well-known follow-up work only if you are certain it exists, named in
         plain text without links or new numbers; otherwise omit it.
       Cover the paper's own stated limitations and future work listed above
       once each; this is the article's only limitations section. Future work
       comes only from that list; never call something the paper already does
       (an appendix included) future work. Do not
       restate the review's conclusions. A limitation or caveat the review body
       already states is named in one clause at most, never explained again;
       spend the words on what the body has not said.

    Rules for all parts:
    - Use the glossary renderings for its terms, as the review does.
    - Every number must already appear in the review; never add links.
    - Synthesize; do not copy the review's sentences.
    - Never mention the writing process or skipped sections.
    """
        + NUMBER_RULES
        + STYLE_RULES
        + MATH_RULES
        + """
    Output exactly:
    <subtitle>...</subtitle>
    <description>...</description>
    <key_takeaways>
    ### ...
    ...
    </key_takeaways>
    <closing>
    ## ...
    ...
    </closing>
    """
    )
