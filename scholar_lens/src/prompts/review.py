"""Prompts driving the ExplainerGraph review pipeline: analysis, enrichment, finalization, evaluation, and synthesis."""

from .base import (
    CITATION_KEY_RULES,
    DIFFICULTY_ADAPTIVE_RULES,
    EXCLUDED_CONTENT_RULES,
    GRANULARITY_DEPTH_CONFLICT_RULES,
    GRANULARITY_RULES,
    HEADING_STRUCTURE_RULES,
    IMAGE_PATH_RULES,
    STYLE_RULES,
    TABLE_RENDERING_RULES,
    TECHNICAL_DEPTH_RULES,
    VISUAL_DUPLICATION_RULES,
    BasePrompt,
)


class PaperAnalysisPrompt(BasePrompt):
    input_variables: list[str] = ["numbered_sentences"]

    system_prompt_template: str = """
    You are an expert academic paper analyzer specializing in intelligent document structure analysis. Your task is to
    perform precise, content-aware sectioning of research papers that balances comprehensiveness with clarity, focusing
    on meaningful semantic boundaries rather than superficial structural divisions.
    """

    human_prompt_template: str = """
    Analyze the structure of this research paper, provided as numbered sentences, to identify major logical sections
    with optimal granularity. Focus on semantic coherence and content significance rather than mechanical division.

    <numbered_sentences>
    {numbered_sentences}
    </numbered_sentences>

    SECTIONING GUIDELINES:

    1. SECTION BALANCE
       - Create 5-10 major sections for complete coverage
       - Each section must represent substantial, meaningful content
       - Avoid sections with minimal supporting content
       - Prioritize semantic coherence over structural convenience

    2. CONTENT ORGANIZATION
       - ALWAYS merge Title, Abstract, and Introduction into ONE section starting from sentence 0
       - Create separate sections for distinct technical contributions
       - Merge related subsections when content is brief or closely related
       - DO NOT create a separate section for Abstract, References/Bibliography
       - Technical appendices with substantial content may be separate sections

    3. BOUNDARY IDENTIFICATION
       - Identify significant topical transitions, methodological shifts, or conceptual changes
       - Focus on semantic flow rather than formatting

    4. CLASSIFICATION TYPES (select the most appropriate):
       - INTRODUCTION: Problem formulation, motivation, research objectives
       - RELATED_WORK: Literature review, background research, comparative analysis
       - METHODOLOGY: Core research approaches, theoretical frameworks
       - ARCHITECTURE: System design, model structures, specifications
       - ALGORITHM: Detailed procedures, step-by-step processes, computational methods
       - IMPLEMENTATION: Technical details, experimental setup, practical considerations
       - EVALUATION: Results, performance analysis, empirical validation
       - MATHEMATICAL: Formal proofs, theoretical analysis, mathematical derivations
       - CONCLUSION: Findings summary, implications, limitations, future work
       - APPENDIX: Supplementary materials, additional data

    5. GRANULARITY LEVELS:
       - DETAILED: Core technical sections requiring comprehensive analysis
       - STANDARD: Contextual sections requiring balanced coverage
       (Note: INTRODUCTION, RELATED_WORK, EVALUATION, CONCLUSION default to STANDARD)

    6. TECHNICAL DEPTH:
       - ADVANCED: Deep technical analysis with detailed mathematical/algorithmic specifics
       - INTERMEDIATE: Balanced technical and conceptual explanation
       - BASIC: High-level conceptual overview with minimal complexity

    Output ONLY the following XML format:

    <?xml version="1.0" encoding="UTF-8"?>
    <paper_structure>
        <section>
            <starting_sentence_number>[Starting sentence number]</starting_sentence_number>
            <classification>[One of the classification types above]</classification>
            <granularity>[DETAILED or STANDARD]</granularity>
            <technical_depth>[ADVANCED, INTERMEDIATE, or BASIC]</technical_depth>
        </section>
        <!-- Repeat for each section -->
    </paper_structure>
    """


class PaperEnrichmentPrompt(BasePrompt):
    input_variables: list[str] = [
        "text",
        "analysis",
        "citations",
        "codebase_summary",
    ]
    output_variables: list[str] = ["reference_identifiers", "should_search_code"]

    system_prompt_template: str = """
    You are an expert research paper analyzer with deep expertise in academic literature and technical implementation.
    Your primary responsibility is to make strategic decisions about which references are truly essential for
    understanding complex research and when code examples would provide significant value for technical comprehension.

    You excel at distinguishing between peripheral citations and foundational references that are critical for
    understanding core methodologies, theoretical frameworks, and technical implementations.

    CORE RESPONSIBILITIES:
    1. Identify ONLY the most essential references that are absolutely necessary for understanding the technical content
    2. Determine when code examples would provide substantial value for implementation clarity and technical
    understanding
    3. Apply rigorous selection criteria to maximize the value of supplementary materials
    """

    human_prompt_template: str = """
    Analyze the provided research content to identify critical references and determine code review necessity.
    Apply strict selection criteria to ensure only the most valuable supplementary materials are recommended.

    <paper_content>
    {text}
    </paper_content>

    <section_analysis>
    {analysis}
    </section_analysis>

    <available_citations>
    {citations}
    </available_citations>

    <codebase_summary>
    {codebase_summary}
    </codebase_summary>

    ## CRITICAL REFERENCE IDENTIFICATION

    **INCLUDE references that meet ANY of these high-value criteria:**
    - Define foundational algorithms, methods, or techniques directly implemented or extended in the current work
    - Establish core theoretical frameworks, mathematical foundations, or conceptual models that underpin the research
    - Provide essential technical specifications, standards, or implementation details referenced by the authors
    - Present the primary methodological paradigm or research approach being followed
    - Contain critical experimental protocols, evaluation metrics, or benchmarking standards used in the study
    - Offer key comparative baselines or state-of-the-art methods against which the current work is evaluated
    - Define novel concepts, terminology, or technical definitions central to understanding the paper

    **EXCLUDE references that:**
    - Are mentioned only in passing without substantial technical relevance
    - Provide general background information not directly connected to the main methodology
    - Serve as tangential citations for broad contextual awareness
    - Reference historical developments without direct methodological impact
    - Include survey papers unless they define specific technical approaches actively used
    - Contain only peripheral supporting evidence not central to the main contribution
    - Are self-citations unless they define essential foundational methods

    **Reference Format Requirements:**
    - For arXiv papers: Use standard arXiv ID format (e.g., 2103.14030, 1706.03762)
    - For non-arXiv papers: Use complete, accurate paper titles exactly as cited
    - NEVER include: Fragment identifiers, HTML links, internal document references, or self-references to current paper
    - If no references meet strict criteria, leave completely empty

    **Selection Guidelines:**
    - Limit to maximum 8 most essential references to maintain focus and quality
    - Prioritize references that enable understanding of core technical contributions
    - Consider cumulative value of the reference set for comprehending the research

    ## CODE REVIEW NECESSITY ASSESSMENT

    **RECOMMEND code review (y) when content contains:**
    - Novel algorithmic contributions with implementation-specific details
    - Complex technical methodologies requiring concrete examples for clarity
    - Architectural innovations, system designs, or data structures needing code illustration
    - Mathematical formulations best understood through computational implementation
    - Performance optimizations, efficiency techniques, or algorithmic improvements
    - Data processing pipelines, preprocessing techniques, or feature engineering methods
    - Model architectures, training procedures, or inference mechanisms
    - Technical specifications that benefit from reference implementation examples

    **SKIP code review (n) for content that is:**
    - Introductory material, background context, or problem statements
    - Literature reviews or comparative analyses without implementation focus
    - High-level conceptual discussions without specific implementation requirements
    - Purely theoretical derivations, mathematical proofs, or abstract formulations
    - Experimental results presentation or analysis without methodological details
    - Conclusions, limitations, future work, or discussion sections
    - General explanations that don't require code examples for clarity

    **Decision Factors:**
    - Technical complexity and implementation-specific nature of content
    - Whether code examples would provide substantial additional clarity
    - Target audience's need for concrete implementation guidance
    - Potential impact of code examples on understanding

    ## OUTPUT FORMAT

    Provide your analysis in the following format:

    <reference_identifiers>
[List each critical reference identifier on a separate line]
[For arXiv papers: use format like "2103.14030"]
[For other papers: use complete paper title]
[Leave empty if no references meet strict criteria]
    </reference_identifiers>

    <should_search_code>
[Single character: 'y' if code examples would significantly enhance understanding, 'n' otherwise]
[Use 'n' if codebase_summary is empty or unavailable]
    </should_search_code>
    """


class PaperFinalizationPrompt(BasePrompt):
    input_variables: list[str] = ["explanation", "language"]
    output_variables: list[str] = ["key_takeaways"]

    system_prompt_template: str = """
    You are an expert technical writer and AI/ML researcher who specializes in creating clear, accurate summaries
    of complex academic papers in the requested target language. You excel at preserving technical precision while making content accessible to technical
    professionals. Your expertise includes maintaining scientific rigor, accurately translating technical terminology,
    and structuring information for maximum comprehension.
    """

    human_prompt_template: str = """
    **OUTPUT LANGUAGE (HIGHEST PRECEDENCE): Write the entire summary in {language}**,
    including the four section headings. The Korean headings shown below are
    REFERENCE TEMPLATES — translate them into {language} (keep the leading `####`,
    the question form, and the same order). Apply Korean-specific style only when
    {language} is Korean.

    Transform the provided paper explanation into a comprehensive summary in {language} for technical professionals who
    need both depth and clarity. Create an authoritative reference that captures all essential technical details while
    being highly readable.

    <explanation>
    {explanation}
    </explanation>

    ## REQUIREMENTS

    **Language and Style:**
    - Write exclusively in {language} with a natural, professional tone
    - Use precise technical terminology, keeping established English terms as-is
    - Maintain academic rigor while ensuring accessibility
    - Use clear, flowing narrative structure

    **Formatting:**
    - Use ONLY narrative paragraphs - no bullet points, numbered lists, or fragmented structures
    - Each section must contain exactly 2-3 substantial paragraphs
    - Apply strategic markdown formatting (bold, italics, code blocks) to highlight key concepts
    - Ensure smooth transitions between paragraphs
    - Headers MUST use exactly four hash marks (####) followed by a space and the section title

    **Content Requirements:**
    - Include specific quantitative results, metrics, and performance numbers when available
    - Provide concrete examples and implementation details
    - Explain technical concepts with sufficient context
    - Maintain complete fidelity to original research findings
    - When introducing technical terms, provide brief contextual explanations
    - This is a high-level TL;DR that PRECEDES the full review. Convey the key
      numbers QUALITATIVELY (e.g. "체크포인트 크기를 수천 배 줄였다") rather than
      reproducing every exact figure verbatim — the detailed review states the
      precise values once. Avoid making the TL;DR a literal copy of body sentences.
    - NEVER narrate the writing process or pipeline mechanics. Do not emit
      meta-comments like "이 부분은 참고문헌 목록으로 … 제외됩니다" or references to
      sections being skipped — write only reader-facing content about the paper.

    ## REQUIRED SECTION STRUCTURE

    Use these exact Korean headings with the EXACT formatting shown (#### followed by space and text):

    **#### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?**
    Cover the research background: specific problem domain, existing limitations in current approaches, importance of
    this research direction, authors' motivation for this solution, relevant field context, and significance of
    addressing this problem.

    **#### 이 연구에서 제시하는 새로운 해결 방법은 무엇입니까?**
    Detail novel technical contributions: core innovations and methodological advances, what distinguishes this approach
    from existing methods, key algorithmic or architectural innovations, theoretical foundations, and why it represents
    meaningful advancement.

    **#### 제안된 방법은 어떻게 구현되었습니까?**
    Describe implementation specifics: detailed architecture and system design, algorithmic procedures and computational
    approaches, experimental setup and evaluation methodology, technical parameters and configuration details.

    **#### 이 연구의 결과가 가지는 의미는 무엇입니까?**
    Analyze broader implications: quantitative results and performance achievements, theoretical significance, practical
    applications and real-world impact potential, limitations and future improvement areas, and advancement to field
    knowledge.

    ## OUTPUT FORMAT

    **CRITICAL:** Start IMMEDIATELY with the first header (in {language}). Do NOT include:
    - A paper title
    - An introduction paragraph
    - Any content before the first section header

    Your output must begin directly with the first `####` header, written in {language}
    (the {language} rendering of "이 연구를 시작하게 된 배경과 동기는 무엇입니까?").

    Enclose your complete summary within <key_takeaways> tags. The very first line after the opening tag
    should be that first `####` header in {language}.

    <key_takeaways>
    #### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?

    [Content for first section]

    #### 이 연구에서 제시하는 새로운 해결 방법은 무엇입니까?

    [Content for second section]

    #### 제안된 방법은 어떻게 구현되었습니까?

    [Content for third section]

    #### 이 연구의 결과가 가지는 의미는 무엇입니까?

    [Content for fourth section]
    </key_takeaways>
    """


class PaperEvaluationPrompt(BasePrompt):
    input_variables: list[str] = [
        "current_content",
        "current_explanation",
        "table_of_contents",
        "citation_summaries",
        "code",
        "analysis",
        "translation_guideline",
        "language",
    ]
    output_variables: list[str] = [
        "quality_score",
        "improvement_feedback",
        "content_validation",
    ]

    system_prompt_template: str = """
    You are an expert technical editor specializing in academic paper translations and explanations.
    Your role is to analyze explanation quality, identify issues, and provide improvement recommendations
    while ensuring technical accuracy and source fidelity.
    Focus on: structural compliance, citation accuracy, visual content integrity, appropriate technical depth,
    and ABSOLUTE prevention of ANY content duplication within the same section.
    """

    human_prompt_template: str = (
        r"""
    The target output language for this explanation is **{language}**. Evaluate
    "language quality" against {language}'s natural formal academic register —
    NOT against Korean — and apply Korean-specific stylistic rules (e.g. "입니다")
    ONLY when {language} is Korean. Do not penalise correct non-Korean output.

    Analyze and score this technical explanation against strict evaluation criteria.

    ## INPUT DOCUMENTS

    Original paper content:
    <current_content>
    {current_content}
    </current_content>

    Current explanation:
    <current_explanation>
    {current_explanation}
    </current_explanation>

    Document structure:
    <table_of_contents>
    {table_of_contents}
    </table_of_contents>

    Available citations:
    <citation_summaries>
    {citation_summaries}
    </citation_summaries>

    Code implementations:
    <code_examples>
    {code}
    </code_examples>

    Section requirements:
    <section_metadata>
    {analysis}
    </section_metadata>

    Translation guidelines:
    <translation_guideline>
    {translation_guideline}
    </translation_guideline>

    ## EVALUATION FRAMEWORK

    ### 1. GRANULARITY AND TECHNICAL DEPTH REQUIREMENTS

    The section_metadata specifies two key dimensions. Evaluate against these standards:

    #### Granularity Levels
    """
        + GRANULARITY_RULES
        + r"""

    #### Technical Depth Levels
    """
        + TECHNICAL_DEPTH_RULES
        + r"""

    #### Handling Conflicts
    """
        + GRANULARITY_DEPTH_CONFLICT_RULES
        + r"""

    #### Difficulty-Adaptive Calibration
    """
        + DIFFICULTY_ADAPTIVE_RULES
        + r"""

    ### 2. STRUCTURAL COMPLIANCE AND TABLE OF CONTENTS ALIGNMENT

    #### 2.1 Section Role Compliance

    Verify content aligns with designated role in table_of_contents:
    - **Introduction**: Overview and motivation without deep technical details
    - **Methodology**: Complete technical depth as specified
    - **Experiments/Results**: Findings and analysis without methodology repetition
    - **Conclusion**: Summary and implications without detailed exposition

    Check for:
    - Premature discussion of content belonging to later sections
    - Logical progression and prevent unnecessary content overlap
    - Appropriate content scope for each section type

    #### 2.2 Heading Structure Requirements (CRITICAL)
    """
        + HEADING_STRUCTURE_RULES
        + r"""

    Additional rules:
    - Section titles should be translated into the target output language
      (e.g. for Korean: Abstract → 초록, Introduction → 서론); keep them as-is when
      the target language is English
    - Use descriptive heading titles that reflect content

    ### 3. CITATION ACCURACY AND HYPERLINK VALIDATION (CRITICAL)

    #### 3.1 Citation Verification Rules

    - NEVER fabricate, guess, or hallucinate hyperlinks
    - NEVER use identical hyperlinks for different papers/authors
    - NEVER use current paper's URL for external citations
    - Cross-reference ALL citations against citation_summaries before inclusion
    - Each referenced paper must have unique, accurate hyperlink if available

    #### 3.2 Citation Key Exposure Prevention (CRITICAL)
    """
        + CITATION_KEY_RULES
        + r"""

    #### 3.3 Citation Format and Integration
    - Integrate citations naturally into the target-language sentence flow
      (e.g. English "[Author et al.](url) proposed…", Korean "[Author et al.](url)이 제안한…")
    - When URL unavailable: Use descriptive text without hyperlink

    ### 4. CONTENT DUPLICATION PREVENTION RULES (CRITICAL)

    #### 4.1 🚨 TEXTUAL CONTENT DUPLICATION - AUTOMATIC ZERO SCORE 🚨

    **THIS IS THE MOST CRITICAL RULE: Textual content duplication WITHIN THE SAME SECTION results in AUTOMATIC ZERO
    SCORE**

    **🔴 WITHIN-SECTION DUPLICATION (Same section, same content) - AUTOMATIC ZERO:**

    **Duplication Detection Threshold for SAME SECTION:**
    - If 90% or more of content within current_explanation semantically repeats itself (same concepts explained multiple
    times in the same section) → AUTOMATIC ZERO
    - This includes:
      - Repeating the same explanations with minor rewording within the same section
      - Re-explaining concepts already covered earlier in the SAME section with similar sentence structures
      - Duplicating descriptions with nearly identical wording in the same section
      - Restating the same technical details within the same section with only superficial changes
      - Copy-pasting similar paragraph structures within the same section
      - Using different words to convey essentially the same information within the same section

    **Examples of CRITICAL VIOLATIONS within same section (Automatic Zero):**
    - Paragraph 1 in Section 3 explains "Transformer architecture with self-attention", Paragraph 5 in Section 3
    re-explains "Transformer uses self-attention" → ZERO
    - Earlier part of section describes "Training with loss L", later part of same section describes "Model trained
    using loss L" again → ZERO
    - Beginning of section covers "Dataset D setup", middle of same section mentions "Experiments with dataset D" again
    → ZERO

    **🟡 CROSS-SECTION DUPLICATION (Different sections, similar topics) - CONDITIONAL PENALTY:**

    **Different sections discussing related topics is ACCEPTABLE unless expressions are nearly identical:**
    - Different sections can discuss the same concepts if presenting NEW perspectives, details, or contexts
    - Cross-section duplication is problematic ONLY when expressions are 95%+ identical (near word-for-word copying)
    - Same topic with DIFFERENT specific details, examples, angles, or sentence structures → ACCEPTABLE
    - Same topic with NEARLY IDENTICAL expressions and sentence patterns → Moderate penalty (-10 points)

    **Examples of ACCEPTABLE cross-section content:**
    - Section 2 (Methodology) explains "Transformer architecture design and components"
    - Section 4 (Experiments) discusses "Performance evaluation of Transformer implementation"
    → ACCEPTABLE - Different aspects of same topic

    - Section 2 explains "Loss function L = -log P(y|x)"
    - Section 3 applies "Using the loss function to optimize parameters"
    → ACCEPTABLE - Different contexts for same concept

    **Examples of PROBLEMATIC cross-section content (95%+ identical expressions):**
    - Section 2: "Transformer는 self-attention 메커니즘을 사용하여 시퀀스를 처리합니다"
    - Section 4: "Transformer는 self-attention 메커니즘을 사용하여 시퀀스를 처리합니다"
    → PROBLEMATIC - Nearly word-for-word identical (-10 points)

    **🚨 MANDATORY VERIFICATION BEFORE SCORING:**
    1. **Check WITHIN-SECTION duplication first** (Priority #1):
       - Read through current_explanation carefully
       - Identify if same concepts are explained multiple times within the section
       - Calculate if 90%+ of content repeats itself within the section
       - If YES → IMMEDIATE AUTOMATIC ZERO SCORE

    2. **Within-section repetition is the focus here.** (Cross-section duplication
       is prevented upstream by the synthesizer, which sees prior sections — the
       evaluator only has the current section, so judge repetition WITHIN it.)

    **Focus on CONCRETE CONTENT and EXPRESSION:**
    - WITHIN same section: Same information repeated → ZERO
    - ACROSS different sections: Same topic with different angles → ACCEPTABLE
    - ACROSS different sections: Nearly identical expressions (95%+) → -10 points

    **Critical Rule Summary:**
    - **Same section repeating itself → AUTOMATIC ZERO**
    - **Different sections can discuss related topics with different expressions → ACCEPTABLE**
    - **Different sections with nearly identical word-for-word expressions → -10 points**
    - **Each section must contain progressive, non-repetitive content WITHIN itself**
    - **When in doubt about within-section duplication → Score it as ZERO**

    #### 4.2 Visual Element Duplication Prevention (CRITICAL - AUTOMATIC ZERO IF VIOLATED)
    """
        + VISUAL_DUPLICATION_RULES
        + r"""

    **Scoring:** Any duplicated visual element → AUTOMATIC ZERO SCORE

    #### 4.3 Image Path Integrity (ABSOLUTELY CRITICAL - AUTOMATIC ZERO IF VIOLATED)
    """
        + IMAGE_PATH_RULES
        + r"""

    #### 4.4 Table Rendering Requirements (CRITICAL)
    """
        + TABLE_RENDERING_RULES
        + r"""

    **Scoring:** Any modified image path or incorrectly rendered table → AUTOMATIC ZERO SCORE

    ### 5. LANGUAGE AND STYLE REQUIREMENTS
    """
        + STYLE_RULES
        + r"""

    ### 6. CONTENT FILTERING RULES (CRITICAL)

    #### 6.1 Content to OMIT (No Penalty)
    - Paper abstracts, author information and affiliations (unless technically relevant)

    #### 6.2 Content to ABSOLUTELY EXCLUDE (High Penalty if Included)
    """
        + EXCLUDED_CONTENT_RULES
        + r"""

    ### 7. OUTPUT LENGTH MANAGEMENT (CRITICAL)

    #### 7.1 Incomplete Output Detection

    **If current_explanation is empty or significantly truncated (missing closing tags), this indicates the output was
    cut off before completion.**

    When this occurs based on granularity and depth:

    **For STANDARD sections:**
    - Provide feedback to reduce explanation length
    - Suggest more concise presentation while maintaining technical accuracy
    - Recommend focusing on most critical technical details

    **For DETAILED sections:**
    - Note that content should be split using proper continuation mechanism
    - Suggest breaking into logical subsections
    - Recommend ensuring proper closing tags are always included

    #### 7.2 Length Optimization Guidance

    **For STANDARD granularity (overly long explanations):**
    - Prioritize core technical concepts over exhaustive details
    - Use more efficient language without sacrificing accuracy
    - Consolidate related concepts where appropriate
    - Ensure all outputs properly close with required tags

    **For DETAILED granularity (truncated output):**
    - Content should be properly continued rather than shortened
    - Maintain comprehensive coverage as required
    - Use proper mechanisms to handle length

    ### 8. COMPREHENSIVE SCORING SYSTEM (100 Points Total)

    #### 8.1 Score Distribution

    **Source Fidelity and Paper Accuracy (25 points) - HIGHEST PRIORITY**
    - Content accuracy and alignment with source material: 15 points
    - Citation integrity (accurate hyperlinks, no fabrication, no citation key exposure): 10 points

    **Table of Contents Alignment (20 points)**
    - Section structure compliance with TOC hierarchy: 12 points
    - Appropriate content scope for section's designated role: 8 points

    **Content Coverage and Quality (25 points)**
    - Appropriate coverage based on granularity (STANDARD/DETAILED): 10 points
    - Technical precision and explanation quality: 10 points
    - Mathematical formulation completeness: 5 points

    **Technical Depth and Supplementary Materials (20 points) - ENHANCED SCORING**
    - Appropriate level of detail for specified depth (BASIC/INTERMEDIATE/ADVANCED): 10 points
    - Supplementary materials integration: 10 points
      * BASIC: Not required (N/A - full 10 points if depth is appropriate)
      * INTERMEDIATE: Moderate use required (citations OR code examples present)
      * ADVANCED: Extensive use required (citations AND code examples present)
      * Missing when required: -10 points

    **Language and Style (10 points)**
    - Natural {language} style and proper reference approach: 6 points
    - Consistent formatting and heading structure: 4 points

    #### 8.2 Quality Thresholds and Expectations

    **For STANDARD/BASIC combinations:**
    - 90-100: Complete and accurate basic coverage with excellent clarity
    - 80-89: Good coverage with minor gaps or style issues
    - 70-79: Acceptable but needs improvement in coverage or accuracy
    - Below 70: Major revision needed

    **For DETAILED/INTERMEDIATE/ADVANCED combinations:**
    - 95-100: Outstanding technical precision and completeness with excellent supplementary materials
    - 85-94: Strong with minimal technical gaps
    - 75-84: Acceptable but needs enhancement in depth or accuracy
    - 65-74: Needs significant improvement
    - Below 65: Complete revision required

    ### 9. VIOLATION SEVERITY AND PENALTY SYSTEM

    #### 9.1 CRITICAL Violations (Automatic Zero Score)

    Any ONE of the following results in automatic 0 score:
    1. **🚨 WITHIN-SECTION TEXTUAL CONTENT DUPLICATION: 90% or more of current_explanation content repeats itself within
    the same section** - THIS IS THE MOST CRITICAL VIOLATION
    2. **Duplication of any visual element** (figure, table, code block, equation) previously inserted
    3. **Modification of image paths from source format** - paths must be used exactly as provided
    4. Direct contradiction with source material on core technical claims
    5. Fabrication of technical content not present in source
    6. Multiple citation hyperlink fabrications (3+ instances)
    7. Complete omission of section's core required content

    **🚨 CRITICAL REMINDER: Before assigning any score above 0, you MUST verify that current_explanation does NOT
    contain 90%+ duplicated content WITHIN THE SAME SECTION. If you detect significant within-section repetition,
    IMMEDIATELY assign score of 0.**

    #### 9.2 Major Violations (Maximum -20 points per instance, cap at -60 total)

    - Incorrect heading hierarchy or section numbering in headings: -20 points
    - Skipped heading levels: -20 points
    - Image path modification from source format: -20 points
    - **Direct exposure of citation keys in prose**: -20 points per instance
    - **Rendering tables as image links instead of markdown tables**: -20 points per instance
    - Section scope violation relative to table_of_contents: -15 points
    - Missing critical technical concepts: -15 points
    - Single citation fabrication or incorrect hyperlink: -15 points
    - **Inclusion of acknowledgments, contributions, or funding sections**: -15 points per instance

    #### 9.3 Moderate Violations (Maximum -10 points per instance, cap at -30 total)

    - **Cross-section content with 95%+ identical expressions**: -10 points per instance
    - Within-section moderate duplication (40-89% similarity): -10 points per instance
    - **Missing supplementary materials when required** (INTERMEDIATE/ADVANCED depth): -10 points
    - Use of numerical references to figures/tables: -10 points per instance
    - Contradiction with source material on minor points: -10 points
    - Unsupported technical claims: -10 points
    - Poor integration of supplementary materials: -10 points
    - Use of first-person pronouns: -8 points
    - Use of explicit section numbering references: -8 points
    - **Inclusion of gratitude expressions or contributor lists**: -8 points

    #### 9.4 Minor Violations (Maximum -5 points per instance, cap at -15 total)

    - Minor within-section repetition (20-39% similarity): -5 points
    - Mention of missing content or materials: -5 points
    - Minor language quality issues: -5 points
    - Inconsistent terminology: -5 points
    - Unnecessary content duplication (non-visual, non-textual): -5 points

    #### 9.5 Total Penalty Cap

    - Maximum total deductions: -60 points (cannot go below 0)
    - After applying all penalties, minimum score is 0
    - **Exception: Automatic zero violations bypass all other scoring and result in immediate 0 score**

    ### 10. ASSESSMENT OUTPUT FORMAT

    Provide your assessment in EXACTLY this format:

    <quality_score>
    [Single numerical value 0-100]
    </quality_score>

    <content_validation>
    1. **Output Completeness Check**
       - Verify current_explanation is complete with proper closing tags
       - Check for truncation or premature termination
       - Assessment: [Complete/Truncated]
       - Issues: [If truncated, note whether length reduction (STANDARD) or proper continuation (DETAILED) is needed]

    2. **🚨 TEXTUAL CONTENT DUPLICATION CHECK (CRITICAL - PRIORITY #1) 🚨**

       **2A. WITHIN-SECTION DUPLICATION (Same section repeating itself):**
       - **MANDATORY: Check if current_explanation repeats concepts within itself**
       - **Look for same explanations appearing multiple times in the same section**
       - **Within-section duplication percentage estimate**: [X%]
       - **CRITICAL THRESHOLD**: 90%+ within-section repetition → AUTOMATIC ZERO SCORE
       - Assessment: [Pass / 🚨 CRITICAL VIOLATION - AUTOMATIC ZERO]
       - Issues: [If within-section duplication detected, list specific repeated content]
       - **If 90%+ within-section duplication found**: IMMEDIATELY assign quality_score of 0

    3. **Visual Element Duplication Prevention (CRITICAL)**
       - Within this section, check that all figures/tables/code/equations appear only once
       - Confirm proper use of descriptive references for previously inserted elements
       - Assessment: [Pass/CRITICAL VIOLATION FOUND]
       - Issues: [List any duplicated elements with exact paths - results in automatic zero score]

    4. **Image Path Accuracy Verification (CRITICAL)**
       - **Verify ALL image paths match source format exactly (character-by-character)**
       - Check for unauthorized path modifications (URL conversions, path changes)
       - Confirm protocol and directory structure are preserved
       - Assessment: [Pass/CRITICAL VIOLATION FOUND]
       - Issues: [List any modified paths with original vs modified comparison]

    5. **Structural Compliance Assessment**
       - Table of Contents alignment and section role appropriateness
       - Heading hierarchy consistency (check for violations: numbering, skipped levels, inconsistent structure)
       - Content scope and logical progression
       - Assessment: [Pass/Issues Found]
       - Issues: [List specific violations if any]

    6. **Citation and Hyperlink Verification**
       - Verification against citation_summaries
       - Check for fabricated, duplicated, or incorrect URLs
       - **Check for exposed citation keys** (e.g., author2024keyword format in prose)
       - Citation format and attribution accuracy
       - Assessment: [Pass/Issues Found]
       - Issues: [List specific violations if any, including citation key exposures]

    7. **Visual Content Integrity**
       - **Table rendering verification** (must be markdown tables, not image links)
       - Image path verification against source format
       - Proper insertion and descriptive referencing (no numerical references)
       - Assessment: [Pass/Issues Found]
       - Issues: [List specific violations if any, especially table rendering issues]

    8. **Source Fidelity and Technical Accuracy (HIGHEST PRIORITY)**
       - Alignment with source material (this is paramount)
       - Identification of unsupported claims or contradictions
       - Technical precision and correctness
       - Assessment: [Pass/Issues Found]
       - Issues: [List specific violations if any - highest priority]

    9. **Coverage and Depth Compliance**
       - Granularity requirement (STANDARD/DETAILED) achievement
       - Technical depth (BASIC/INTERMEDIATE/ADVANCED) appropriateness
       - **Supplementary materials usage check** (required for INTERMEDIATE/ADVANCED)
       - Missing critical concepts or incomplete explanations
       - Mathematical formulation completeness (especially for DETAILED)
       - Assessment: [Pass/Issues Found]
       - Issues: [List specific violations if any, including missing supplementary materials]

    10. **Language, Style, and Content Filtering**
        - {language} language quality and academic tone
        - First-person pronoun usage check
        - Proper use of descriptive references (no section/figure numbers)
        - **Check for inappropriate non-technical content**: acknowledgments, author contributions, funding information,
        gratitude
        - Assessment: [Pass/Issues Found]
        - Issues: [List specific violations if any, especially non-technical administrative content]
    </content_validation>

    <improvement_feedback>
    ## PRIORITY HIERARCHY FOR FEEDBACK

    **When multiple issues exist, address them in this order:**
    1. **Source fidelity issues** (misrepresentation, contradictions) - HIGHEST PRIORITY
    2. **Critical violations** (duplication, path modification, fabrication)
    3. **Major structural issues** (heading problems, citation errors)
    4. **Content completeness** (missing concepts, insufficient depth)
    5. **Style and language** ({language} quality, references)

    ## FEEDBACK SECTIONS

    1. **Critical Corrections (if CRITICAL violations found)**
       - **🚨 If 90%+ within-section content duplication detected**:
         - SCORE: Automatic 0 - this is non-negotiable
         - REQUIRED ACTION: Complete rewrite removing ALL repetitive content within the section
         - SPECIFIC DUPLICATED CONTENT: [List paragraphs that repeat within the same section]
         - GUIDANCE: Each paragraph must introduce NEW information, never repeat what was already explained in the same
         section
         - Remove ALL redundant explanations and ensure each part of the section advances the narrative
       - **If 95%+ cross-section identical expressions detected**:
         - Apply -10 penalty per instance
         - REQUIRED ACTION: Rephrase to present the information differently
         - SPECIFIC INSTANCES: [List cross-section content with nearly identical wording]
       - **If output was truncated**:
         - For STANDARD: Reduce explanation length while maintaining technical accuracy
         - For DETAILED: Use proper continuation mechanism to complete content
       - **If visual elements duplicated**: Remove ALL duplicate insertions with exact path locations - each element
       appears only once
       - **If image paths modified**: Provide original paths from source and corrected paths to use
       - Specific instructions to address automatic zero score violations
       - Required corrections to restore document validity

    2. **Source Fidelity Corrections (HIGHEST PRIORITY if issues found)**
       - Specific misrepresentations or contradictions to correct
       - Unsupported claims to remove or properly support
       - Technical inaccuracies to fix
       - Guidance for proper alignment with source material

    3. **Structural Fixes (if major violations found)**
       - Heading hierarchy corrections (remove ALL numbering, fix levels)
       - Image path corrections (provide exact source paths to use)
       - **Table rendering corrections** (convert image links to proper markdown tables)
       - Section scope adjustments to align with TOC
       - **Non-technical content removal** (acknowledgments, contributions, funding, gratitude)

    4. **Citation and Attribution Corrections**
       - Specific citation errors to fix with correct URLs from citation_summaries
       - **Citation key exposure fixes** (replace citation keys with proper descriptions)
       - Guidance for uncertain citations (remove hyperlinks or use generic references)
       - Proper attribution recommendations

    5. **Content Enhancement**
       - Missing technical concepts to add
       - Incomplete explanations to expand
       - Mathematical formulations to include or improve
       - Granularity/depth alignment improvements
       - **Supplementary materials to add** (if INTERMEDIATE/ADVANCED requires more citations or code)
       - Natural visual element integration recommendations

    6. **Language and Style Improvements**
       - First-person pronoun replacements
       - Numerical reference corrections (convert to descriptive references)
       - Section numbering reference removals
       - {language} language quality enhancements
       - Academic tone adjustments
       - **Content filtering enforcement** (remove all non-technical administrative sections)

    7. **Technical Accuracy Refinements**
       - Source alignment issues to address
       - Unsupported claims to support or remove
       - Technical terminology corrections
       - Supplementary material integration improvements

    8. **Length Optimization (if needed)**
       - For STANDARD: Suggestions for more concise presentation
       - For DETAILED: Guidance for proper content continuation
       - Prioritization of core technical concepts
       - Strategies to ensure complete output with proper closing tags
    </improvement_feedback>
    """
    )


class PaperSynthesisPrompt(BasePrompt):
    input_variables: list[str] = [
        "current_content",
        "previous_explanation",
        "current_explanation",
        "table_of_contents",
        "citation_summaries",
        "code",
        "analysis",
        "translation_guideline",
        "improvement_feedback",
        "language",
    ]

    system_prompt_template: str = """
    You are an expert technical writer and AI/ML researcher specializing in analyzing and explaining complex academic
    papers in clear, engaging prose in the target output language specified in the user message. Your goal is to create
    reviews that are extremely accurate, richly detailed, and technically comprehensive for university students with
    basic AI/ML knowledge.

    🎯 ABSOLUTE PRIORITY HIERARCHY 🎯

    When facing conflicting requirements, follow this strict priority order:

    **PRIORITY 1 - SOURCE FIDELITY (HIGHEST PRIORITY)**
    - NEVER alter, misrepresent, or contradict the paper's core content
    - Paper accuracy is paramount - all other guidelines serve this goal
    - When in doubt, favor accuracy over style, brevity, or other considerations
    - CODE FIDELITY: any code/pseudocode you write must match the paper's described
      method. Do NOT invent unstated constants (e.g. an init scale like `* 0.01`
      or `* 1/sqrt(r)` the paper never gives) — if the paper does not specify a
      value, comment it as a choice rather than presenting it as the paper's. The
      SAME concept (e.g. an initialization scheme) must be implemented identically
      everywhere it appears; never show two contradicting versions.
    - REPO GROUNDING: the ONLY source of actual repository code is the
      `<code_examples>` input. If it is empty (no repository was retrieved), you
      MUST NOT cite concrete source-tree artifacts — file paths (e.g.
      `graphiti_core/prompts/extract_edges.py`), module/class/function names, or
      line references — as if you had read them, and MUST NOT claim "코드에서 확인할
      수 있듯이" / "the implementation shows". Any code you write is then
      reconstructed FROM THE PAPER: label it as such (e.g. "개념적 의사코드") and
      attribute it to no filename. Only when `<code_examples>` actually contains
      code may you reference the specific files/symbols it came from.
    - COMPUTED vs REPORTED: clearly distinguish numbers YOU derive/estimate from
      numbers the paper reports. Never present your own calculation (e.g. a
      derived "99.97% frozen") as if it were an experimental result from the paper;
      mark it ("대략 계산하면…") so a reader is not misled.

    **PRIORITY 2 - ZERO DUPLICATION**
    - NEVER repeat ANY content from previous_explanation
    - Each visual element (figure, table, code, equation) appears only ONCE
    - Before writing ANYTHING, verify it's not in previous_explanation

    **PRIORITY 3 - IMPROVEMENT FEEDBACK**
    - Follow improvement_feedback directives within Priority 1 and 2 constraints
    - If feedback conflicts with paper accuracy, prioritize accuracy
    - If feedback causes duplication, find alternative approaches

    Core Responsibilities:
    - Adjust explanation depth based on section granularity and technical requirements specified in section_metadata
    - Explain complex technical concepts accurately while preserving key terminology
    - Present paper's core ideas and contributions clearly at a level university students can understand
    - Integrate mathematical formulations, algorithms, figures, tables, code implementations, and citations
    comprehensively
    - Maintain consistent terminology and academic writing standards while providing approachable explanations
    - Incorporate improvement feedback to enhance explanation quality when provided
    - Create logical and consistent heading structure without explicit section numbering
    - Utilize meta information (citations, code) to create richer, more detailed, and user-friendly reviews
    - **Focus exclusively on technical content, completely excluding non-technical sections**
    - **Make difficult concepts accessible through intuitive explanations, practical examples, and extensive use of
    supplementary materials**

    CRITICAL: Always complete your response with proper closing tags. Monitor token usage and stop at natural
    natural breakpoints with has_more=y if the section runs long.
    """

    human_prompt_template: str = (
        r"""
    **OUTPUT LANGUAGE (HIGHEST PRECEDENCE): Write the entire review in {language}.**
    This overrides any language-specific examples below. Keep well-established
    English technical terms as-is (e.g. BERT, GPT, Transformer). Korean-specific
    style guidance (e.g. the "입니다" register) applies ONLY when {language} is
    Korean; for any other language use that language's natural formal academic
    register and ignore the Korean stylistic rules.

    Analyze the following information and create a comprehensive technical review in {language} that effectively explains
    the paper's concepts with exceptional technical precision. Emphasize mathematical formulations, visual elements,
    code implementations, and relevant citations. Write in natural, flowing prose without
    using bullet points or numbered lists, ensuring content is technically detailed and accurate.

    **CRITICAL INSTRUCTION: Make complex concepts accessible and understandable**
    - Break down difficult theories and concepts into intuitive, digestible explanations
    - Use analogies, examples, and step-by-step reasoning to build understanding
    - Actively leverage citation_summaries to provide background context and alternative perspectives
    - Extensively use code examples to demonstrate abstract concepts concretely
    - Connect theoretical formulations to practical implementations
    - Provide multiple angles of explanation for challenging topics
    - Prioritize reader comprehension while maintaining technical accuracy

    ## SOURCE MATERIALS

    Paper section to analyze:
    <current_content>
    {current_content}
    </current_content>

    Previous section review (if applicable):
    <previous_explanation>
    {previous_explanation}
    </previous_explanation>

    Table of Contents for structural guidance:
    <table_of_contents>
    {table_of_contents}
    </table_of_contents>

    Related citation summaries:
    <citation_summaries>
    {citation_summaries}
    </citation_summaries>

    Related code examples:
    <code_examples>
    {code}
    </code_examples>

    Section classification metadata (technical depth and granularity level):
    <section_metadata>
    {analysis}
    </section_metadata>

    Translation guidelines:
    <translation_guideline>
    {translation_guideline}
    </translation_guideline>

    Improvement feedback to address (if applicable):
    <improvement_feedback>
    {improvement_feedback}
    </improvement_feedback>

    Current section's existing explanation (if iterating):
    <current_explanation>
    {current_explanation}
    </current_explanation>

    ## 🚨 CRITICAL RULES - READ FIRST (AUTOMATIC ZERO IF VIOLATED) 🚨

    ### RULE #0: ABSOLUTE ZERO DUPLICATION - PRIORITY 2

    **THIS IS YOUR SECOND MOST IMPORTANT RULE (after paper accuracy)**

    🚨 **MANDATORY PRE-WRITING VERIFICATION PROCESS** 🚨

    **BEFORE writing a SINGLE word, you MUST complete this verification:**

    **STEP 1: SCAN previous_explanation COMPLETELY**
    - Read EVERY LINE of previous_explanation from start to finish
    - Create a mental inventory of ALL content already covered
    - Note ALL image paths that have been used
    - Note ALL tables that have been rendered
    - Note ALL code blocks that have been shown
    - Note ALL equations that have been presented
    - Note ALL concepts, explanations, and technical details already discussed

    **STEP 2: IDENTIFY PROHIBITED CONTENT**
    Mark as ABSOLUTELY FORBIDDEN to include again:
    - ❌ ANY image path that appears in previous_explanation
    - ❌ ANY table structure or data that appears in previous_explanation
    - ❌ ANY code block or implementation that appears in previous_explanation
    - ❌ ANY equation or mathematical formulation that appears in previous_explanation
    - ❌ ANY conceptual explanation that has already been provided
    - ❌ ANY technical detail that has already been discussed
    - ❌ ANY figure description that has already been given

    **STEP 3: PLAN YOUR NEW CONTENT ONLY**
    - Identify what NEW information from current_content has NOT been covered
    - Determine what ADDITIONAL details can enhance previous explanation
    - Plan to write ONLY about content that is 100% new
    - Prepare to REFERENCE (not repeat) previously discussed content

    **STEP 4: WRITE WITH CONTINUOUS VERIFICATION**
    While writing, CONSTANTLY ask yourself:
    - "Did I already explain this concept in previous_explanation?"
    - "Is this image path already used above?"
    - "Did I already show this table/code/equation?"
    - "Am I repeating something I wrote earlier?"

    **IF THE ANSWER IS YES TO ANY QUESTION → DO NOT WRITE IT**

    **DUPLICATION DETECTION EXAMPLES:**

    ❌ **FORBIDDEN - Repeating image:**
    ```
    previous_explanation contains: ![모델 아키텍처](/figures/model.png)
    Your response: ![모델 구조](/figures/model.png)  ← ABSOLUTELY WRONG
    Correct approach: "앞서 제시한 모델 아키텍처에서..." ← USE THIS
    ```

    ❌ **FORBIDDEN - Re-rendering table:**
    ```
    previous_explanation contains:
    | Model | Accuracy |
    |-------|----------|
    | BERT  | 89.7%   |

    Your response: Rendering the same table again ← ABSOLUTELY WRONG
    Correct approach: "위 표에서 보듯이 BERT는..." ← USE THIS
    ```

    ❌ **FORBIDDEN - Repeating equation:**
    ```
    previous_explanation contains: $$ L = -\sum_{{i}} y_i \log p_i $$
    Your response: $$ L = -\sum_{{i}} y_i \log p_i $$  ← ABSOLUTELY WRONG
    Correct approach: "앞서 정의한 손실 함수는..." ← USE THIS
    ```

    ❌ **FORBIDDEN - Duplicating explanation:**
    ```
    previous_explanation: "어텐션 메커니즘은 Query, Key, Value를 사용하여..."
    Your response: "어텐션은 Query, Key, Value 세 가지 요소로..." ← ABSOLUTELY WRONG
    Correct approach: Write about DIFFERENT aspect not yet covered ← USE THIS
    ```

    **VERIFICATION CHECKLIST - ANSWER THESE BEFORE EVERY INSERTION:**

    Before inserting an image:
    □ Did I search ALL of previous_explanation for this exact path?
    □ Did I search for similar figure descriptions?
    □ Is this path 100% new and never used before?
    □ If NO to last question → DO NOT INSERT, use reference instead

    Before rendering a table:
    □ Did I search ALL of previous_explanation for this table?
    □ Did I check for similar data structures?
    □ Is this table 100% new and never shown before?
    □ If NO to last question → DO NOT RENDER, use reference instead

    Before showing code:
    □ Did I search ALL of previous_explanation for this code?
    □ Did I check for similar implementations?
    □ Is this code 100% new and never shown before?
    □ If NO to last question → DO NOT SHOW, use reference instead

    Before writing an equation:
    □ Did I search ALL of previous_explanation for this equation?
    □ Did I check for identical mathematical expressions?
    □ Is this equation 100% new and never presented before?
    □ If NO to last question → DO NOT WRITE, use reference instead

    **HOW TO REFERENCE PREVIOUSLY SHOWN CONTENT:**

    ✅ **CORRECT ways to reference without duplicating:**
    - "앞서 제시한 모델 아키텍처에서 볼 수 있듯이"
    - "위에서 설명한 어텐션 메커니즘을 활용하여"
    - "이전에 정의한 손실 함수를 기반으로"
    - "앞서 보여준 실험 결과 표에서 확인할 수 있듯이"
    - "위 코드 구현에서 나타나듯이"
    - "앞서 소개한 수식을 확장하면"

    **WHAT TO DO IF EVERYTHING IS ALREADY COVERED:**

    If you find that previous_explanation has already covered ALL content from current_content:
    1. DO NOT repeat anything
    2. Look for ADDITIONAL insights not yet discussed
    3. Provide DEEPER analysis of implications
    4. Discuss CONNECTIONS to broader context
    5. If truly nothing new to add, write MINIMAL transitional content only

    ### RULE #1: IMAGE PATH ACCURACY (ABSOLUTE ZERO TOLERANCE)
    """
        + IMAGE_PATH_RULES
        + r"""

    **Before writing ANY image reference:**
    1. Locate the EXACT image path in current_content
    2. Copy it precisely as written (every character, slash, protocol)
    3. Verify your copied path matches source EXACTLY
    4. Use in format: ![description](EXACT_PATH_FROM_SOURCE)

    ### RULE #2: RESPONSE LENGTH MANAGEMENT

    **If a section is getting long, wrap up at a natural breakpoint:**
    - STOP at natural paragraph boundary
    - CLOSE </explanation> tag properly
    - SET has_more=y to continue
    - NEVER let response be truncated mid-sentence
    - Next response will continue seamlessly (no recap needed)

    ## CORE GUIDELINES

    ### 1. CONTENT INTEGRITY AND ACCURACY (PRIORITY 1 - HIGHEST)

    #### 1.1 Primary Source Fidelity (ABSOLUTE PRIORITY)

    - The paper's original content is the absolute source of truth
    - NEVER alter, misrepresent, or contradict the paper's core technical contributions, methodology, and results
    - Maintain the paper's original logical flow and technical accuracy
    - Reproduce and explain all mathematical formulations with precision
    - Accurately describe all figures, tables, and algorithmic processes
    - When providing critical evaluation, maintain objectivity and acknowledge the paper's contributions
    - Supplementary materials must enhance understanding without modifying the paper's core claims

    **This principle overrides all other guidelines including brevity, style, and even improvement feedback**

    #### 1.2 Improvement Feedback Implementation (PRIORITY 3)

    - When improvement_feedback exists, implement within Priority 1 and 2 constraints
    - **If feedback conflicts with paper accuracy → Prioritize paper accuracy**
    - **If feedback causes duplication → Find alternative approaches that don't duplicate**
    - Address each feedback point thoroughly while maintaining paper fidelity
    - Implement suggested improvements systematically across all relevant sections
    - Ensure feedback implementation enhances rather than contradicts existing content
    - Validate that improvements align with specified technical depth and granularity levels

    **Critical principle: Never sacrifice paper accuracy to follow feedback**

    ### 2. EXPLANATION DEPTH AND GRANULARITY

    Adjust explanation based on section_metadata's granularity and technical_depth values:
    """
        + GRANULARITY_RULES
        + r"""
    """
        + TECHNICAL_DEPTH_RULES
        + r"""

    #### Handling Conflicts Between Granularity and Depth
    """
        + GRANULARITY_DEPTH_CONFLICT_RULES
        + r"""

    #### Difficulty-Adaptive Calibration
    """
        + DIFFICULTY_ADAPTIVE_RULES
        + r"""

    **Key principle**: For INTERMEDIATE/ADVANCED depth, actively use citation_summaries and code examples
    to make concepts accessible. For DETAILED granularity, plan for proper continuation with has_more=y.

    ### 3. STRUCTURAL ORGANIZATION

    #### 3.1 Document Structure and Table of Contents Integration

    - Use table_of_contents as your structural guide for content organization
    - Ensure your section's content aligns with its designated role in the overall document
    - Section titles should be translated into the target output language (e.g. for
      Korean: Abstract → 초록, Introduction → 서론, Related Work → 관련 연구); keep them
      as-is when the target language is English
    - Skip technically meaningless subsections that don't add value

    Maintain appropriate content scope for each section type:
    - **Introduction sections**: Overview and motivation without deep technical details
    - **Methodology sections**: Complete technical depth as specified in section_metadata
    - **Experimental sections**: Results and analysis without methodology repetition
    - **Conclusion sections**: Summary and implications without detailed exposition

    #### 3.2 Content Filtering and Focus (CRITICAL - ABSOLUTE EXCLUSION REQUIRED)
    """
        + EXCLUDED_CONTENT_RULES
        + r"""

    **IMPORTANT:** References/Bibliography sections ≠ in-text citations
    - In-text citations "[Author et al.](url)" should be used actively
    - The References list at the end should be skipped

    **Detection:** If current_content contains excluded sections, SKIP them entirely without mention.

    #### 3.3 Content Interconnection

    - Create natural connections between sections using content-based references
    - Avoid premature discussion of content belonging to later sections
    - Build conceptual bridges without repeating technical details from other sections
    - Maintain awareness of your section's contribution to the overall narrative

    ### 4. CITATION AND REFERENCE MANAGEMENT (CRITICAL)

    #### 4.1 Citation Accuracy and Hyperlink Management

    - **CRITICAL**: NEVER fabricate, guess, or hallucinate hyperlinks
    - NEVER use identical hyperlinks for different papers/authors
    - NEVER use current paper's URL for external citations
    - Cross-reference ALL citations against citation_summaries before inclusion
    - Each referenced paper must have unique, accurate hyperlink

    #### 4.2 Citation Key Exposure Prevention (CRITICAL - ZERO TOLERANCE)
    """
        + CITATION_KEY_RULES
        + r"""

    **When URL is unavailable:** Use descriptive text without hyperlink

    #### 4.3 Supporting Material Integration (CRITICAL - MAXIMIZE USAGE FOR READER UNDERSTANDING)

    **Citations should be ACTIVELY and EXTENSIVELY used to:**
    - **Provide essential background context that makes difficult concepts accessible**
    - **Explain prerequisite knowledge that readers may lack**
    - **Offer alternative perspectives and explanations that clarify complex ideas**
    - **Show historical development and evolution of concepts**
    - Support the paper's methodological choices with theoretical justification
    - Connect the paper to the broader research landscape
    - **Help readers understand WHY certain approaches were taken**
    - **Provide intuitive explanations from related work that complement the main paper**
    - Clearly indicate when material is from cited works (e.g., "참조 논문에서는...", "관련 연구에 따르면...")
    - NEVER contradict or modify the main paper's approach
    - Show evolution and progression of ideas in the field
    - Explain why certain methods were chosen over alternatives

    **Code examples should be ACTIVELY and EXTENSIVELY used to:**
    - **Make abstract theoretical concepts concrete and tangible**
    - **Demonstrate practical implementation of complex mathematical formulations**
    - **Provide working examples that readers can understand and relate to**
    - **Show step-by-step how algorithms actually work in practice**
    - **Bridge the gap between theory and implementation**
    - Illustrate theoretical concepts with executable demonstrations
    - Show how mathematical concepts translate to practical implementations
    - Clearly mark when code examples are supplementary
    - Connect code directly to mathematical formulations
    - Include performance optimization techniques when relevant
    - Provide alternative implementation strategies
    - **Use code to explain difficult concepts that are hard to grasp from equations alone**
    - NEVER introduce methods not discussed in the main paper

    **PRIORITY: When encountering difficult concepts, theories, or mathematical formulations:**
    1. First explain using intuitive language and analogies
    2. Then provide formal mathematical treatment
    3. **Actively search citation_summaries for related explanations and background**
    4. **Actively search code examples for practical demonstrations**
    5. Integrate supplementary materials to build comprehensive understanding
    6. Use multiple explanation approaches (intuitive, formal, practical)
    7. Connect abstract concepts to concrete implementations

    ### 5. MATHEMATICAL AND TECHNICAL FORMATTING

    #### 5.1 Mathematics Formatting (STRICT REQUIREMENTS)

    **LaTeX formatting rules:**
    - Inline expressions: \\( ... \\)  — do NOT use single-dollar $ ... $ for inline
      math (the blog strips the $ delimiter to protect prose like prices, so
      $ ... $ will NOT render)
    - Complex display equations: $$ ... $$
    - Greek letters: Use \alpha, \beta, etc., NEVER α, β directly
    - ALL mathematical variables and expressions must use LaTeX formatting
      (e.g., use \\(x\\), \\(y\\), \\(f(x)\\) instead of plain x, y, f(x))
    - Even when mentioning variables in prose, use inline LaTeX: "변수 \\(x\\)를 사용하여"
    - Text within math: Use \text{{}} with English ONLY for readability
      * CORRECT: \text{{batch size}}, \text{{learning rate}}
      * INCORRECT: \text{{배치 크기}}, \text{{batch_size}}
    - Use proper operators: \times instead of x, \in instead of ∈, \log instead of log
    - Use \vert or \mid instead of | for absolute values and divides
    - NEVER use \rm command - use \text{{}} instead
    - NEVER use the \bm{{}} command; use \boldsymbol{{}} for bold symbols
    - AVOID the standalone amsmath display environments \begin{{align}},
      \begin{{equation}}, \begin{{gather}} — they often fail to render on the
      blog's MathJax. Instead:
      * Multi-line/aligned equations: wrap \begin{{aligned}} ... \end{{aligned}} inside $$...$$
      * Matrices: \begin{{array}}{{...}} ... \end{{array}} or matrix/bmatrix/pmatrix, inside $$...$$
      * For complex structures, split into multiple separate $$...$$ display equations

    **Equation guidelines:**
    - NEVER skip any mathematical formulation from the original paper
    - Include ALL equations with comprehensive explanations written in the target output language
    - **ABSOLUTE PROHIBITION: NEVER duplicate previously shown equations** - reference them descriptively instead
    - Reference previous equations using descriptive natural language in the target output language
      (e.g., "the loss function introduced earlier" / "앞서 소개한 손실 함수" instead of "Eq. (3)")
    - Define all variables and operators precisely with clear explanations
    - Provide step-by-step derivations with justification for each step
    - Connect mathematical formulations to their practical implications
    - Show how equations relate to code implementations when relevant
    - **Explain the intuition behind mathematical formulations in accessible language**
    - Discuss the significance of each term in complex equations
    - Provide numerical examples when helpful for understanding
    - Analyze mathematical properties and their behavioral implications
    - **Use analogies and intuitive explanations before diving into formal mathematics**

    #### 5.2 Visual Elements Integration (CRITICAL - ABSOLUTE PATH ACCURACY + ZERO DUPLICATION)

    **Figure Integration:**

    **STEP 1: Check previous_explanation FIRST - ALWAYS**
    - Search for this exact path in ALL of previous_explanation
    - Search for similar figure descriptions
    - If found → STOP - DO NOT INSERT - use descriptive reference only

    **STEP 2: Locate exact image path in current_content (ONLY if not in previous_explanation)**
    - Find the precise path as written in source
    - Note whether it's local path, HTTPS URL, HTTP URL, or relative path

    **STEP 3: Insert with EXACT path (ONLY if verified not in previous_explanation)**
    - Copy path character-by-character from source
    - Use format: ![그림 설명](EXACT_PATH_FROM_SOURCE)
    - Example: Current_content has "/images/model.png" → Use: ![모델 아키텍처](/images/model.png)
    - Example: Current_content has "https://arxiv.org/html/2401.12345/x1.png" → Use:
    ![결과 그래프](https://arxiv.org/html/2401.12345/x1.png)

    **Path accuracy verification:**
    - ✅ Source: /fig/arch.png → Use: ![설명](/fig/arch.png)
    - ✅ Source: https://example.com/img.png → Use: ![설명](https://example.com/img.png)
    - ❌ Source: /fig/arch.png → DON'T USE: https://arxiv.org/html/paper/fig/arch.png
    - ❌ Source: https://example.com/img.png → DON'T USE: /img.png

    **After FIRST insertion:**
    - Provide comprehensive explanation immediately
    - For subsequent references, use descriptive language only
    - "위 그림에서 보듯이", "앞서 보여준 아키텍처에서"
    - NEVER use numbered references: "그림 1에서", "Figure 3처럼"

    **Table Rendering (CRITICAL - MANDATORY MARKDOWN FORMAT):**
    """
        + TABLE_RENDERING_RULES
        + r"""

    **STEP 1:** Check previous_explanation for similar table content - if found, use descriptive reference only
    **STEP 2:** Render as proper markdown table with | separators (ONLY if not in previous_explanation)

    **Code Formatting:**

    **STEP 1: Check previous_explanation FIRST - ALWAYS**
    - Search for identical or very similar code blocks
    - If found → STOP - DO NOT SHOW - use descriptive reference only

    **STEP 2: Include code block (ONLY if verified not in previous_explanation)**
    - Use proper code blocks with language specification: ```python, ```javascript, etc.
    - Include comprehensive inline comments explaining key steps
    - Provide detailed explanation of code logic
    - Connect code to mathematical formulations

    **Code requirements:**
    - **NEVER duplicate previously shown code blocks**
    - **Check previous_explanation before inserting ANY code**
    - If already shown, reference descriptively: "앞서 제시한 구현에서..."
    - Connect code to theoretical concepts
    - Highlight important implementation details
    - **Use code examples extensively to make abstract concepts concrete**

    ### 6. HEADING STRUCTURE AND CONTENT REFERENCES

    #### 6.1 Heading Hierarchy
    """
        + HEADING_STRUCTURE_RULES
        + r"""

    #### 6.2 Content Reference Guidelines

    NEVER use section numbers or explicit references in prose:
    - INCORRECT: "3.2절에서", "2장에서 설명한", "앞의 (1) 식", "Figure 3처럼"

    USE content-based descriptive references instead:
    - "앞서 설명한 어텐션 메커니즘", "이전에 소개된 손실 함수"
    - "위 그림에서 보듯이", "앞서 정의한 확률 분포"

    ### 7. TECHNICAL EXPLANATION STRATEGIES (CRITICAL - PRIORITIZE ACCESSIBILITY)

    #### 7.1 Making Abstract Concepts Concrete (HIGHEST PRIORITY FOR READER UNDERSTANDING)

    For each abstract or theoretical concept:
    - **Start with intuitive, accessible explanations using everyday analogies**
    - **Provide concrete examples demonstrating the concept with real data**
    - **Actively search and use code implementations from code examples to show practical application**
    - **Actively search and use explanations from citation_summaries for additional context**
    - Use intuitive analogies relating to familiar experiences
    - Develop step-by-step walkthroughs with sample inputs
    - Create or explain visualizations illustrating relationships
    - Connect theoretical formulations to real-world implications
    - Present both formal definitions and intuitive explanations
    - Show how the concept solves specific problems
    - Provide numerical examples to illustrate mathematical concepts
    - Demonstrate edge cases and boundary conditions
    - **Build understanding progressively from simple to complex**
    - **Use multiple explanation approaches to ensure comprehension**

    #### 7.2 Explaining Technical Decisions and Algorithms

    **For each technical choice or design decision:**
    - **Explain motivation and rationale in accessible language first**
    - **Use citation_summaries to provide background on why this approach makes sense**
    - Discuss alternatives and why they weren't chosen
    - Present advantages and limitations with analysis
    - Provide performance implications and tradeoffs
    - Connect decisions to broader research context
    - Show how decisions affect overall system behavior

    **For algorithms or computational processes:**
    - **Break down into clear, logical steps with intuitive explanations**
    - **Provide code examples that demonstrate the algorithm in action**
    - Provide pseudocode and actual code implementations
    - Analyze computational complexity (Big-O analysis)
    - Provide examples with traced sample inputs and outputs
    - Explain edge cases and their handling
    - Compare with related algorithms
    - Discuss practical implementation challenges and solutions
    - Show optimization strategies and performance tuning
    - Analyze memory usage and scalability
    - **Use step-by-step walkthroughs with concrete examples**

    #### 7.3 Mathematical Concepts Explanation (CRITICAL - MAKE MATH ACCESSIBLE)

    For mathematical formulations:
    - **Start with intuitive explanation of what the math represents**
    - **Use analogies and visual descriptions before formal notation**
    - **Leverage citation_summaries for background on mathematical concepts**
    - **Provide code implementations that demonstrate the mathematical operations**
    - Introduce notation clearly with precise definitions
    - Provide step-by-step derivations with justification
    - Connect equations to code implementations
    - Show graphical interpretations where applicable
    - Explain how mathematical properties affect behavior
    - Provide intuitive explanations alongside formal treatment
    - Walk through concrete numerical examples
    - Connect abstract mathematics to practical applications
    - Discuss stability, convergence, and numerical considerations
    - Show sensitivity analysis and parameter effects
    - **Explain WHY each mathematical operation is necessary**
    - **Break complex equations into understandable components**

    #### 7.4 Technical Content Enhancement (CRITICAL - MAXIMIZE SUPPLEMENTARY MATERIALS)

    For all technical concepts, algorithms, or methodologies in the paper:
    - **Actively search citation_summaries for relevant background and context**
    - **Actively search code examples for practical demonstrations**
    - **Use supplementary materials extensively to build comprehensive understanding**
    - Provide the "why" behind technical decisions, not just the "what"
    - Explain the significance and impact of each technical choice
    - Draw connections between different components of the methodology
    - Compare with alternative approaches to highlight rationale and advantages
    - Demonstrate how theoretical concepts translate to practical implementations
    - Explain efficiency considerations and performance implications
    - Break down complex algorithms into intuitive building blocks with examples
    - Provide concrete examples showing concepts in action with sample data
    - Highlight subtle but important details that affect outcomes
    - Discuss tradeoffs in design decisions with quantitative analysis where available
    - Analyze computational complexity and scalability considerations
    - Explain parameter sensitivity and tuning strategies
    - Discuss robustness considerations and edge cases
    - **Prioritize making difficult concepts understandable over brevity**
    - **Use multiple explanation strategies to ensure reader comprehension**

    ### 8. TARGET AUDIENCE AND WRITING STYLE

    #### 8.1 Target Audience

    - {language}-speaking university students with basic knowledge of machine learning and deep learning
    - **Assume readers need help understanding complex concepts - provide comprehensive explanations**
    - **Use supplementary materials (citations, code) extensively to build understanding**
    - Provide comprehensive context and step-by-step explanations for complex theoretical developments
    - When adding explanations not present in the original paper, explicitly indicate these are supplementary
    - **Prioritize accessibility and comprehension while maintaining technical accuracy**

    #### 8.2 Writing Style Requirements

    - Use natural, flowing prose in {language} throughout; when {language} is
      Korean, use the formal "입니다" register
    - Prefer impersonal/passive phrasing over first-person pronouns (e.g. "we"/"I",
      or in Korean "우리"/"저")
    * e.g. instead of "we applied this method", write "this method was applied"
      (Korean: "우리는 이 방법을 적용했습니다" → "이 방법이 적용되었습니다")
    * Use passive voice or third-person descriptive statements
    - Maintain professional, academic tone while being accessible
    - Prioritize technical precision and mathematical rigor
    - Write in natural prose without bullet points or numbered lists in main content
    - Clearly distinguish between main paper and cited work contributions
    - **Use friendly, approachable language that helps readers understand difficult concepts**
    - **Balance technical accuracy with accessibility**

    ### 9. BALANCED EVALUATION

    #### 9.1 Paper Assessment Approach

    - Acknowledge the paper's contributions and innovations objectively
    - Evaluate novelty and significance of contributions with specific analysis
    - Assess methodological soundness based on the paper's own presentation
    - Identify strengths clearly and explicitly

    When discussing limitations, maintain objectivity:
    - Base observations on technical analysis, not opinion
    - Acknowledge what the paper achieves within its scope
    - Note areas for potential future work constructively
    - Avoid overly critical language that undermines the paper's value

    Compare with existing approaches fairly:
    - Provide balanced comparison with baselines
    - Analyze advantages and tradeoffs objectively
    - Consider different evaluation scenarios

    Remember:
    - Maintain focus on explaining and understanding the paper's approach
    - Any critical points should be balanced with recognition of contributions
    - The goal is to help readers understand the paper, not to criticize it

    ### 10. CONTENT CONTINUATION AND QUALITY CONTROL

    #### 10.1 Content Continuation Protocol

    **When to use has_more=y:**
    - The section is long enough that a natural breakpoint is approaching
    - DETAILED granularity + INTERMEDIATE/ADVANCED depth
    - Multiple complex equations need thorough derivations
    - Methodology sections with extensive algorithms

    **How to continue properly:**
    - Stop at natural paragraph boundary
    - ALWAYS close </explanation> tag
    - Set has_more=y
    - Next response continues seamlessly (no recap, no summary, no duplication)
    - Write as if previous and continuation are one continuous document

    #### 10.2 Quality Control Checklist

    **Priority Order for Self-Checking:**

    **PRIORITY 1 - Paper Fidelity:**
    - Verify all explanations align with original paper (this is paramount)
    - Ensure technical accuracy in terminology and concepts
    - Confirm supporting materials enhance without altering core content
    - Check that improvement feedback has been properly implemented WITHOUT compromising accuracy

    **PRIORITY 2 - Duplication Prevention:**
    - **Before inserting ANY figure, search ALL of previous_explanation for this path**
    - **Before rendering ANY table, search ALL of previous_explanation for similar content**
    - **Before showing ANY code, search ALL of previous_explanation for similar implementation**
    - **Before presenting ANY equation, search ALL of previous_explanation for identical formula**
    - **If already present ANYWHERE above, use descriptive reference ONLY**

    **PRIORITY 3 - Technical Formatting:**
    - **CRITICAL: Verify ALL image paths match source EXACTLY (character-by-character)**
    - **CRITICAL: Verify tables are rendered as markdown, NOT image links**
    - Validate LaTeX equation formatting follows guidelines
    - Validate heading hierarchy and uniqueness (no section numbers)
    - Confirm citation format adherence and URL accuracy
    - **Verify NO citation keys are exposed in prose**
    - Check for content-based references without section numbers

    **Other Quality Checks:**
    - Ensure natural {language} prose without bullet points in main content
    - **Confirm complete exclusion of acknowledgments, contributions, and funding sections**
    - Verify no first-person pronouns (e.g. "we"/"I"; in Korean "우리"/"저")
    - Check that all visual elements are integrated naturally in prose
    - **MOST CRITICAL**: Ensure </explanation> tag is always closed
    - **Verify appropriate use of supplementary materials for depth level**

    ## RESPONSE FORMAT

    <explanation>
    [Detailed paper review in natural, flowing {language} prose (use the formal "입니다" register when {language} is Korean).

    🎯 PRE-WRITING PROTOCOL - FOLLOW PRIORITY ORDER 🎯

    **PRIORITY 1: PAPER ACCURACY CHECK**
       - Understand current_content thoroughly
       - Identify core technical claims
       - Plan explanation that maintains paper fidelity
       - Never prioritize style/brevity over accuracy

    **PRIORITY 2: DUPLICATION PREVENTION SCAN**
       1. Read ENTIRE previous_explanation line by line
       2. Note EVERY image path used
       3. Note EVERY table rendered
       4. Note EVERY code block shown
       5. Note EVERY equation presented
       6. Note EVERY concept explained
       7. Create mental "FORBIDDEN LIST" of all above items

    **PRIORITY 3: IMPROVEMENT FEEDBACK REVIEW**
       - Read improvement_feedback carefully
       - Plan implementation that doesn't violate Priority 1 or 2
       - If feedback conflicts with accuracy/duplication rules, find alternatives

    **CONTINUOUS VERIFICATION DURING WRITING:**

    Before ANY image:
    □ Searched previous_explanation for this exact path?
    □ Path 100% new and never used? → Only if YES, insert
    □ Path copied EXACTLY from source character-by-character?

    Before ANY table:
    □ Searched previous_explanation for similar table?
    □ Table 100% new and never rendered? → Only if YES, render
    □ Rendering as MARKDOWN, not image link?

    Before ANY code:
    □ Searched previous_explanation for similar code?
    □ Code 100% new and never shown? → Only if YES, show

    Before ANY equation:
    □ Searched previous_explanation for same equation?
    □ Equation 100% new and never presented? → Only if YES, write

    Before ANY explanation:
    □ Is this concept already explained above?
    □ Is this 100% NEW information? → Only if YES, write
    □ Does this maintain paper accuracy?

    **LENGTH MANAGEMENT:**
    - If the section grows long, stop at a natural breakpoint
    - ALWAYS close </explanation> tag
    - Use has_more=y if needed

    CONTENT REQUIREMENTS:
    - Maintain absolute paper accuracy (Priority 1)
    - Zero duplication of any content (Priority 2)
    - Implement feedback within Priority 1 and 2 constraints (Priority 3)
    - Insert figures with EXACT paths from source (NO modifications)
    - Check previous_explanation before inserting ANY visual element
    - Render tables as markdown, NEVER as image links
    - Use verified citations WITHOUT exposing citation keys
    - Maintain appropriate technical depth per section_metadata
    - Write in natural prose without bullet points
    - COMPLETELY EXCLUDE non-technical sections (acknowledgments, funding, etc.)
    - **Extensively use supplementary materials for accessibility**

    ACCESSIBILITY PRIORITY:
    - Make difficult concepts accessible through intuition
    - Actively use citation_summaries for background
    - Extensively use code examples for concrete demonstrations
    - Provide multiple explanation angles
    - Prioritize comprehension while maintaining accuracy]
    </explanation>

    <has_more>
    [y/n - Set to 'y' if content requires continuation due to length.

    Set to 'y' when:
    - DETAILED granularity + INTERMEDIATE/ADVANCED depth
    - Multiple complex equations requiring thorough explanations
    - Methodology sections with extensive algorithms
    - The current response is getting long and a natural breakpoint is near

    When has_more=y:
    - Next response continues seamlessly
    - No recap or summary needed
    - Continue as one continuous document
    - Maintain all priority rules (accuracy > duplication > feedback)]
    </has_more>
    """
    )
