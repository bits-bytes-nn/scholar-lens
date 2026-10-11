"""Prompts for extracting structured metadata, citations, code, figures, and the table of contents from a paper."""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ..post_conventions import load_local_conventions
from .base import (
    BasePrompt,
)

# Fallback vocabularies until the caller passes the blog's post-conventions
# spec (categories.topics and tags.controlled). Callers override them by passing
# ``category_options`` / ``controlled_tags`` at invoke time.
_CONVENTIONS = load_local_conventions()
DEFAULT_CATEGORY_OPTIONS: tuple[str, ...] = tuple(_CONVENTIONS.topics)
DEFAULT_CONTROLLED_TAGS: tuple[str, ...] = tuple(_CONVENTIONS.tags.controlled)


class AttributesExtractionPrompt(BasePrompt):
    input_variables: list[str] = [
        "text",
        "existing_keywords",
        "category_options",
        "controlled_tags",
    ]
    output_variables: list[str] = [
        "title",
        "seo_title",
        "authors",
        "affiliation",
        "category",
        "keywords",
        "controlled_tags",
    ]

    system_prompt_template: str = """
    You extract publication metadata and blog taxonomy from AI/ML research
    papers: title, authors, publishing organization, one topic category, and a
    few reusable tags. You are precise and conservative.
    """

    human_prompt_template: str = """
    <paper_content>
    {text}
    </paper_content>

    <existing_keywords>
    {existing_keywords}
    </existing_keywords>

    1. TITLE: the exact title as printed on the first page, or "N/A".
    2. SEO_TITLE: only when the title is longer than 70 characters, a version of
       at most 70 characters that keeps the model or method name first (e.g.
       "DeepSeek-Coder: The Rise of Code Intelligence"); otherwise empty.
    3. AUTHORS: author names in order, comma-separated, names only; or "N/A".
    4. AFFILIATION: the ONE organization that published the paper — a company
       lab or a university as commonly named (e.g. "Meta AI", "Google DeepMind",
       "DeepSeek AI", "Stanford University"). No departments, no people, no
       lists. Use the lab most authors belong to; or "N/A".
    5. CATEGORY: exactly one of these values, copied verbatim:
       {category_options}
    6. CONTROLLED_TAGS: 1-3 values from this closed list, copied verbatim,
       comma-separated. Always give at least one — the closest fit when none is
       exact:
       {controlled_tags}
    7. KEYWORDS: at most 5 specific tags for the paper's own contributions
       (1-4 words each, e.g. "Low-Rank Adaptation", "Paged Attention"). Reuse an
       existing keyword whenever one fits; never restate the paper title, never
       use generic terms ("Deep Learning", "Transformer") or a technique the
       paper does not actually use.

    Output the block below exactly once — no notes, no corrections, nothing
    after it:
    <title>...</title>
    <seo_title>...</seo_title>
    <authors>...</authors>
    <affiliation>...</affiliation>
    <category>...</category>
    <controlled_tags>...</controlled_tags>
    <keywords>...</keywords>
    """

    @classmethod
    def get_prompt(cls, enable_prompt_cache: bool = False) -> ChatPromptTemplate:
        return (
            super()
            .get_prompt(enable_prompt_cache)
            .partial(
                category_options=", ".join(DEFAULT_CATEGORY_OPTIONS),
                controlled_tags=", ".join(DEFAULT_CONTROLLED_TAGS),
            )
        )


class CitationExtractionPrompt(BasePrompt):
    input_variables: list[str] = ["text", "existing_citations"]
    output_variables: list[str] = ["citations", "has_more"]

    system_prompt_template: str = """
    You are a world-class academic citation extraction specialist with expertise in bibliographic analysis and pattern
    recognition. Your mission is to achieve 100% accuracy and completeness in extracting ALL academic references from
    research papers, regardless of formatting complexity or citation style variations.

    CORE EXPERTISE:
    • Perfect identification of all academic references without any omissions
    • Flawless handling of inconsistent, non-standard, or discipline-specific citation formats
    • Precise extraction of COMPLETE bibliographic metadata with special attention to FULL TITLES
    • Expert disambiguation of similar references and handling of edge cases
    • Comprehensive coverage of all academic publication types and sources
    • Advanced pattern recognition for multi-line, fragmented, or unusual citation structures

    QUALITY STANDARDS:
    • Zero tolerance for missed references - extract every single academic citation
    • Maintain absolute fidelity to original formatting, capitalization, and special characters
    • Extract COMPLETE TITLES without truncation, abbreviation, or omission
    • Handle complex author names (hyphens, particles, diacritics) with perfect preservation
    • Process citations spanning multiple lines or containing formatting irregularities
    • Exclude only non-academic sources (blogs, news, websites) unless they have formal DOI/citation
    """

    human_prompt_template: str = r"""
    MISSION: Extract ALL bibliographic references from this research paper with surgical precision and complete
    coverage, ensuring FULL TITLE EXTRACTION.

    <paper_content>
    {text}
    </paper_content>

    <existing_citations>
    {existing_citations}
    </existing_citations>

    EXTRACTION PROTOCOL:

    STEP 1: PATTERN ANALYSIS
    • Systematically analyze the paper's citation style and formatting patterns
    • Identify reference section structure, numbering systems, and formatting conventions
    • Note any discipline-specific citation patterns or unusual formatting
    • Identify how titles are formatted (quotes, italics, plain text, etc.)

    STEP 2: COMPREHENSIVE EXTRACTION WITH FULL TITLE FOCUS
    • Extract up to 100 references per batch (processing constraint)
    • Format each reference as a precise tuple: (Authors, Year, Title, arXiv_ID)
    • Authors: First 3 names + "et al." if more (preserve exact formatting: hyphens, particles, diacritics)
    • Year: Publication year as integer, or None if unavailable
    • Title: **COMPLETE FULL TITLE** - This is CRITICAL
    • arXiv_ID: Full ID with "arXiv:" prefix, or None if not applicable

    CRITICAL TITLE EXTRACTION REQUIREMENTS:
    ⚠️  NEVER truncate, abbreviate, or shorten titles
    ⚠️  Extract the ENTIRE title including all subtitles, colons, and descriptive parts
    ⚠️  If a title spans multiple lines, combine all parts into the complete title
    ⚠️  Include all punctuation, special characters, and formatting marks in titles
    ⚠️  Do NOT stop at the first few words - extract the COMPLETE title
    ⚠️  If uncertain about title boundaries, err on the side of including more text rather than less

    TITLE EXTRACTION EXAMPLES (CORRECT):
    ✓ "Attention Is All You Need" (complete)
    ✓ "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (complete with subtitle)
    ✓ "Language Models are Few-Shot Learners" (complete)
    ✓ "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (complete with subtitle)

    TITLE EXTRACTION EXAMPLES (INCORRECT - AVOID):
    ✗ "Attention" (truncated)
    ✗ "BERT" (truncated)
    ✗ "Language Models" (truncated)
    ✗ "An Image" (truncated)

    STEP 3: SOURCE IDENTIFICATION AND QUALITY ASSURANCE
    • ONLY extract from formal reference/bibliography sections, not in-text citation markers

    ⚠️  CRITICAL: DO NOT EXTRACT LATEX CITATION COMMANDS
    LaTeX citation commands are in-text references, NOT bibliographic entries. NEVER extract these:
    • \cite{{author2023title}}
    • \citep{{author2023title}}
    • \citet{{author2023title}}
    • \parencite{{author2023title}}
    • \textcite{{author2023title}}
    • \footcite{{author2023title}}
    • \autocite{{author2023title}}
    • \\cite, \\parencite, \\citep, etc. (with any number of backslashes)
    • Any command starting with backslash(es) followed by "cite" or containing citation keys like "author2023title"
    • Bracketed in-text references like [1], [Smith 2023], (Smith et al., 2023)

    LATEX COMMAND EXAMPLES TO IGNORE:
    ✗ \parencite{{author2023title}}
    ✗ \\parencite{{author2023title}}
    ✗ \cite{{author2023title}}
    ✗ \citep{{author2023title}}
    ✗ [1] or [Author, 2023]

    • Skip citations already in existing_citations list
    • Include ALL academic sources regardless of venue prestige, age, or field
    • Handle edge cases: multi-line citations, page breaks, unusual formatting
    • VERIFY each title is complete and not truncated
    • Mark continuation status: 'y' if more references remain, 'n' if complete

    TARGET PUBLICATION TYPES (include ALL):
    ✓ Peer-reviewed journal articles
    ✓ Conference papers and proceedings
    ✓ Books and edited volumes
    ✓ Book chapters
    ✓ Technical reports and white papers
    ✓ Theses and dissertations
    ✓ Workshop papers
    ✓ Preprints with DOI or arXiv ID
    ✓ Standards and specifications with formal citations
    ✓ Government and institutional reports with formal citations

    RESPONSE FORMAT:
    <citations>
    ("Smith, Johnson, Williams et al.", 2023, "Deep Learning Architectures for Natural Language Processing:
    A Comprehensive Survey and Implementation Guide", "arXiv:2301.12345")
    ("Anderson, Lee", 2022, "Statistical Methods in Machine Learning: Theory, Practice, and Applications", None)
    ("Brown, Taylor, Chen et al.", 2020, "Advances in Computer Vision and Pattern Recognition for Autonomous Systems",
    None)
    </citations>

    <has_more>
    y
    </has_more>

    CRITICAL SUCCESS FACTORS:
    1. COMPLETENESS: Miss zero academic references from formal reference sections
    2. ACCURACY: Preserve exact formatting, spelling, and punctuation from original
    3. FULL TITLES: Extract complete titles without any truncation or abbreviation
    4. PRECISION: Use exact tuple format with proper quotes and commas
    5. THOROUGHNESS: Double-check for citations in unexpected locations (appendices, figure captions, etc.)
    6. CONSISTENCY: Apply uniform standards while respecting original formatting variations
    7. LATEX EXCLUSION: Never extract LaTeX citation commands or in-text references

    SYSTEMATIC VERIFICATION PROCESS:
    1. Identify and map the complete reference section structure
    2. Distinguish between formal bibliography and in-text LaTeX commands
    3. Extract references sequentially in their original order from bibliography only
    4. For each reference, carefully identify the complete title boundaries
    5. Ensure no title is truncated or abbreviated
    6. Cross-check footnotes, endnotes, and any supplementary reference lists
    7. Verify completeness by scanning for citation markers throughout the text
    8. Confirm no references were missed due to formatting irregularities
    9. Confirm no LaTeX commands were mistakenly extracted

    TITLE EXTRACTION VERIFICATION CHECKLIST:
    □ Title includes all words from start to end
    □ Title includes subtitles after colons, dashes, or semicolons
    □ Title includes all descriptive phrases and qualifiers
    □ Title maintains original capitalization and punctuation
    □ No abbreviation or truncation occurred
    □ Multi-line titles are properly combined

    Execute this extraction with the precision of a master bibliographer. Every academic reference matters, and every
    title must be complete and accurate. Remember: extract from bibliography sections only, never from LaTeX commands
    or in-text citations.
    """


class CitationAnalysisPrompt(BasePrompt):
    input_variables: list[str] = ["reference_title", "original_content"]
    output_variables: list[str] = ["analysis"]

    system_prompt_template: str = """
    You are an expert AI/ML research analyst specializing in citation analysis. Your task is to analyze cited papers and
    explain their technical contributions and relevance to the citing work.

    Focus on:
    - Technical accuracy and precision
    - Clear explanation of methodologies
    - Relevance to the citing paper
    - Concise but comprehensive analysis
    """

    human_prompt_template: str = """
    Analyze the relationship between this cited paper and the citing work:

    CITED PAPER TITLE:
    {reference_title}

    CITING WORK CONTEXT:
    {original_content}

    Provide analysis following this structure:

    ## Core Contribution
    - What is the main technical innovation of the cited paper?
    - How does it advance the field?

    ## Technical Details
    - Key algorithms, methods, or architectures
    - Important mathematical formulations (LaTeX between $$ and $$ for both inline and display math; never \\( \\) or single $)
    - Critical experimental findings or theoretical results

    ## Relevance to Citing Work
    - How is this paper being used or referenced?
    - What specific aspects are being built upon or compared against?
    - Why is this citation important for the citing work?

    ## Significance
    - Impact on the field
    - Key advantages or limitations
    - Position relative to other approaches

    Format your response as:
    <analysis>
    [Your structured analysis here using the format above]
    </analysis>

    If the title alone provides insufficient information, focus on what can be reasonably inferred and clearly state any
    limitations in your analysis.
    """


class CitationSummaryPrompt(BasePrompt):
    input_variables: list[str] = ["reference_content", "original_content"]

    system_prompt_template: str = """
    You are an expert research analyst. Create focused, technically precise summaries of academic papers.

    Your tasks:
    - Extract key technical contributions relevant to the citation context
    - Connect methodologies between cited and citing works
    - Provide accurate technical analysis
    - Use precise mathematical notation
    """

    human_prompt_template: str = r"""
    Analyze this reference and create a technical summary focused on its relevance to the citing work.

    REFERENCE TO ANALYZE:
    {reference_content}

    CITING WORK CONTEXT:
    {original_content}

    ANALYSIS REQUIREMENTS:

    1. **Primary Contributions** (start here)
       - Main technical innovations
       - Key algorithms or methods
       - Novel theoretical insights

    2. **Technical Details**
       - Core algorithms and foundations
       - Implementation approaches
       - Experimental methods and results
       - Limitations and constraints

    3. **Mathematical Content**
       - Use LaTeX between $$ and $$ for both inline and display math; never \\( \\) or single $
       - Standard symbols: \alpha, \beta, \sum, \int, etc.
       - Define all variables clearly

    4. **Relevance Analysis**
       - Direct connections to citing work
       - Methodological similarities/differences
       - Technical advantages/disadvantages

    5. **Format Guidelines**
       - Clear section structure
       - Precise technical terminology
       - Consistent mathematical notation
       - Focus on citation relevance

    Provide a concise technical analysis emphasizing direct relevance and accuracy.
    """


class CodeAnalysisPrompt(BasePrompt):
    input_variables: list[str] = ["code"]

    system_prompt_template: str = """
    You are an expert AI/ML code analyzer. Analyze code snippets to create precise technical descriptions for semantic
    search and research paper matching.

    Your role:
    - Identify specific ML/AI algorithms, architectures, and methods
    - Extract key technical implementation details
    - Use standard ML terminology for optimal searchability
    - Connect code to research concepts and paper topics
    - Be concise but comprehensive (50-100 words)
    """

    human_prompt_template: str = """
    Analyze this code and provide a technical description for ML/AI research matching:

    <code>
    {code}
    </code>

    Focus on:
    1. **Algorithm/Method**: What specific ML/AI technique is implemented?
    2. **Architecture**: Model structure, layers, components
    3. **Mathematics**: Key formulas, operations, computations
    4. **Implementation**: Notable design choices, optimizations
    5. **Research Context**: How this relates to academic papers/concepts

    Output: Clear technical summary using standard ML/AI terminology.
    """


class CodebaseSummaryPrompt(BasePrompt):
    input_variables: list[str] = ["codebase"]

    system_prompt_template: str = """
    You are an expert AI/ML code analyzer. Analyze codebases to create technical summaries for research paper matching.

    Your role:
    - Identify core ML/AI algorithms and architectures
    - Extract key mathematical operations and model components
    - Map code to research concepts using standard terminology
    - Focus on research-relevant implementation details
    - Be precise and concise
    """

    human_prompt_template: str = """
    Analyze this codebase and provide a technical summary for research paper relevance assessment:

    <codebase>
    {codebase}
    </codebase>

    Provide:
    1. **Primary Algorithms**: Main ML/AI methods implemented
    2. **Model Architecture**: Key components, layers, mathematical operations
    3. **Data Pipeline**: Processing, training, evaluation components
    4. **Implementation Details**: Notable design choices, optimizations
    5. **Frameworks**: Critical ML/AI libraries and dependencies
    6. **Research Mapping**: How code relates to academic concepts

    Output a clear, structured summary using standard ML/AI terminology.
    Focus on technical aspects that indicate research paper alignment.
    """


class FigureAnalysisPrompt(BasePrompt):
    input_variables: list[str] = ["caption", "image_data"]

    system_prompt_template: str = """
    You are an expert figure analyst for academic research papers. Analyze figures, diagrams, and visualizations to
    extract meaningful technical insights.

    Your task:
    - Determine if the figure is research-relevant (data plots, architectures, results) or non-technical (logos,
    decorative elements, etc.)
    - For research figures: identify visualization type, extract key information, and explain significance
    - For non-technical figures: briefly note what it is
    - Be concise and precise
    """

    human_prompt_template: str = """
    Analyze this figure from a research paper.

    Caption: {caption}

    Provide exactly 3 sentences:

    1. What type of visualization is this? (e.g., line plot, architecture diagram, bar chart, logo, decorative image)
    2. What key information does it show? (specific metrics, components, relationships, or "non-technical content")
    3. What is its research significance? (main finding, contribution, or "not research-relevant")

    If the figure appears to be a logo, header image, or decorative element unrelated to research content, state this
    clearly in sentence 1 and keep sentences 2-3 brief.

    Be specific and technical for research figures. Use standard terminology.
    """
    human_image_prompt_template: str = "data:image/jpeg;base64,{image_data}"

    @classmethod
    def get_prompt(
        cls,
        enable_prompt_cache: bool = False,
    ) -> ChatPromptTemplate:
        if enable_prompt_cache:
            raise ValueError("Prompt caching is not supported for image-based prompts")
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    template=cls.system_prompt_template,
                    input_variables=cls.input_variables,
                ),
                HumanMessagePromptTemplate.from_template(
                    template=[
                        {"text": cls.human_prompt_template},
                        {"image_url": {"url": cls.human_image_prompt_template}},
                    ],
                    input_variables=cls.input_variables,
                ),
            ]
        )
