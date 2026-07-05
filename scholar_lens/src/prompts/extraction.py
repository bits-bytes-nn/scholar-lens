"""Prompts for extracting structured metadata, citations, code, figures, and the table of contents from a paper."""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from .base import (
    BasePrompt,
)


class AttributesExtractionPrompt(BasePrompt):
    input_variables: list[str] = ["text", "existing_keywords"]
    output_variables: list[str] = [
        "title",
        "authors",
        "affiliation",
        "category",
        "keywords",
    ]

    system_prompt_template: str = """
    You are a specialized metadata extraction system for AI/ML research papers. Your task is to accurately identify and
    extract the paper's title and authors plus institutional affiliation, research category, and technical keywords
    representing novel contributions.
    """

    human_prompt_template: str = """
    Analyze this research paper and extract the requested metadata with high precision:

    <paper_content>
    {text}
    </paper_content>

    <existing_keywords>
    {existing_keywords}
    </existing_keywords>

    EXTRACTION REQUIREMENTS:

    0. TITLE & AUTHORS (from the paper's header / first page):
       - TITLE: the paper's exact title as printed. If it genuinely cannot be found, output "N/A".
       - AUTHORS: the author names in order, comma-separated (e.g. "Edward J. Hu, Yelong Shen, Phillip Wallis").
         Names only — no affiliations, superscripts, or emails. If none can be found, output "N/A".

    1. KEYWORDS EXTRACTION (Maximum 10 items):
       - FIRST PRIORITY: Reuse existing keywords EXACTLY as written when they match the paper's content
       - Focus exclusively on novel technical contributions: new algorithms, architectures, methods, or techniques
       - EXCLUDE generic terms: "machine learning", "neural networks", "deep learning", "artificial intelligence"
       - EXCLUDE standard components: "transformer", "CNN", "RNN", "attention mechanism" (unless novel variant)
       - Each keyword must represent a distinct, innovative technical concept introduced or significantly advanced
       - Use precise Title Case formatting
       - Prioritize methodological innovations over application domains

    2. AUTHOR AFFILIATION:
       - Extract the primary institutional affiliation of the FIRST AUTHOR only
       - Use the complete, official English name of the institution
       - Include specific department, lab, or research center if prominently mentioned
       - Format as a single, clear institutional identifier

    3. RESEARCH CATEGORY (Select exactly ONE):
       Choose the most specific category that best represents the paper's primary technical contribution.
       Output the category label EXACTLY as written below — no extra words, notes, or parentheticals.
       - Computer Vision
       - Language Models
       - Speech Processing
       - Multimodal Learning
       - Image/Video Generation
       - Training & Inference Optimization
       - Reinforcement Learning
       - Retrieval Augmented Generation
       - Recommendation Systems
       - Time Series Analysis
       - Other

       Category scope notes (do NOT append these to the output):
       - "Retrieval Augmented Generation" also covers knowledge graphs, ontologies,
         and graph/knowledge-based retrieval or querying.

    OUTPUT FORMAT (Follow exactly):
    <title>
    [The paper's exact title, or "N/A"]
    </title>

    <authors>
    [Comma-separated author names in order, or "N/A"]
    </authors>

    <affiliation>
    [First author's primary institution with department/lab if notable]
    </affiliation>

    <category>
    [Single most specific category from the provided list]
    </category>

    <keywords>
    [Comma-separated list of novel technical contributions, maximum 10 items]
    </keywords>
    """


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
    - Important mathematical formulations (use LaTeX: \\( ... \\) for inline, $$ ... $$ for display; never single-dollar $ ... $ for inline)
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
       - Use LaTeX notation: \\( ... \\) for inline, $$ ... $$ for display; never single-dollar $ ... $ for inline
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


class TableOfContentsPrompt(BasePrompt):
    input_variables: list[str] = ["paper_content"]

    system_prompt_template: str = """
    You are an expert academic document analyzer. Your task is to create structured table of contents
    from research papers with accurate section hierarchy and meaningful content summaries.

    Key principles:
    - Extract sections in order as they appear
    - Maintain exact hierarchical relationships
    - Generate concise but informative summaries
    - Use consistent, simple XML structure
    - Handle edge cases gracefully
    """

    human_prompt_template: str = """
    Analyze the following research paper and create a structured table of contents.

    <paper_content>
    {paper_content}
    </paper_content>

    ## INSTRUCTIONS

    ### 1. SECTION EXTRACTION
    - Identify ALL section headers with their hierarchical levels (1, 2, 3, 4+)
    - Preserve original numbering and titles EXACTLY as they appear
    - Include standard sections: Abstract, Introduction, Methods, Results, Discussion, Conclusion, References
    - Include any additional sections: Related Work, Experiments, Limitations, Appendices, etc.
    - Handle unnumbered sections by assigning appropriate hierarchy level

    ### 2. CONTENT SUMMARIZATION
    For each section WITHOUT subsections:
    - Write 1-2 clear, informative sentences
    - Include specific technical details when available (methods, results, metrics)
    - Focus on what the section actually contributes
    - Use technical terminology appropriately
    - Avoid generic phrases like "This section discusses..."

    For sections WITH subsections:
    - Provide brief overview of what the main section covers
    - Let subsections contain the detailed summaries

    ### 3. VISUAL ELEMENTS (Optional)
    Only include if clearly mentioned:
    - Figures: Figure 1, Fig. 2, etc.
    - Tables: Table 1, Tab. 2, etc.
    - Use empty tags if none found: <figures></figures>
    - Each item as separate element: <figure>Figure 1</figure>

    ## OUTPUT FORMAT

    Use this XML structure with elements (not attributes) to preserve all data:

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <table_of_contents>
        <section>
            <level>1</level>
            <number>1</number>
            <title>Introduction</title>
            <summary> Introduces a novel transformer-based architecture for document classification achieving 94.2%
            accuracy on benchmark datasets. Establishes the research gap and outlines key contributions including a new
            attention mechanism.</summary>
            <figures>
                <figure>Figure 1</figure>
                <figure>Figure 2</figure>
            </figures>
            <tables>
                <table>Table 1</table>
            </tables>
            <subsections>
                <section>
                    <level>2</level>
                    <number>1.1</number>
                    <title>Research Motivation</title>
                    <summary>Identifies limitations in existing approaches with 15-20% error rates on multilingual
                    datasets and motivates the need for cross-lingual document understanding.</summary>
                    <figures></figures>
                    <tables></tables>
                    <subsections></subsections>
                </section>
                <section>
                    <level>2</level>
                    <number>1.2</number>
                    <title>Contributions</title>
                    <summary>Lists three main contributions: novel attention mechanism, multilingual training strategy,
                    and comprehensive evaluation on 5 language datasets.</summary>
                    <figures></figures>
                    <tables></tables>
                    <subsections></subsections>
                </section>
            </subsections>
        </section>

        <section>
            <level>1</level>
            <number>2</number>
            <title>Related Work</title>
            <summary>
                Reviews transformer architectures and document classification methods, comparing 15 baseline approaches
                and identifying gaps in cross-lingual performance.
            </summary>
            <figures></figures>
            <tables>
                <table>Table 2</table>
            </tables>
            <subsections></subsections>
        </section>
    </table_of_contents>
    ```

    ## QUALITY GUIDELINES

    ### Essential Requirements:
    1. **Accuracy**: Section numbers and titles must match source exactly
    2. **Completeness**: Include every section, don't skip any
    3. **Hierarchy**: Maintain correct parent-child relationships
    4. **Clarity**: Summaries should be informative and specific
    5. **XML Structure**: Use elements not attributes, include all required tags

    ### CRITICAL - XML Output Rules:
    - Output raw XML tags directly: <section>, <level>, <title>, etc.
    - NEVER escape XML tags as HTML entities like &lt;section&gt; or &lt;level&gt;
    - NEVER mix escaped and unescaped tags
    - Special characters in TEXT CONTENT should be escaped: & → &amp;, < → &lt;, > → &gt;
    - But XML TAGS themselves must remain unescaped: <section> NOT &lt;section&gt;

    ### Best Practices:
    - If section numbering is inconsistent, do your best to determine hierarchy from formatting
    - If no clear subsections exist, use empty <subsections></subsections>
    - Include quantitative results when mentioned (percentages, metrics, sample sizes)
    - Mention specific algorithms, datasets, or methodologies when present
    - Keep summaries concise but informative (1-2 sentences typically)
    - Always include all required elements even if empty

    ### Error Handling:
    - If uncertain about hierarchy, err on the side of flatter structure
    - If figures/tables are unclear, use empty tags: <figures></figures>
    - If section title is ambiguous, use the text as closely as possible
    - Always include all required elements: level, number, title, summary, figures, tables, subsections

    ## PROCESSING STEPS

    1. **First Pass**: Scan for all section headers and establish hierarchy
    2. **Second Pass**: Extract content for each section
    3. **Third Pass**: Identify visual elements if clearly referenced
    4. **Final Pass**: Generate XML with proper nesting and all required elements

    Generate ONLY the XML output. No additional text or explanations."""
