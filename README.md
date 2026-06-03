## 👩‍🏫 SCHOLAR-LENS

AI-powered research assistant that turns papers and documentation into publishable
content: in-depth **paper reviews**, concise **paper summaries**, and self-study
**technical guides** — driven from the CLI, AWS Batch, or a Slack bot.

![Sample](./assets/sample.png)

### ✨ Features

- **Three artifact types**: in-depth paper *reviews* (multi-stage LangGraph),
  concise paper *summaries* (five-section format with figures/tables/math/code),
  and technical *guides/tutorials* generated from documentation URLs.
- **Any AI/ML paper**: accepts an arXiv ID **or** an arbitrary paper PDF URL
  (non-PDF URLs are rejected); arXiv is no longer required.
- **AI-Powered Analysis**: Amazon Bedrock (Claude, incl. Opus 4.8) for multi-stage
  understanding, citation analysis, and figure interpretation.
- **Code Integration**: GitHub repository analysis with semantic search via FAISS.
- **Slack Paper Bot**: mention/DM the bot with a paper id/URL or doc URLs; it
  parses intent with an LLM and dispatches a Batch job, then reports back.
- **Blog publishing**: every artifact can be uploaded to S3 and opened as a PR
  against a configured github.io blog (category-aware Jekyll front matter).
- **Scalable Infrastructure**: AWS Batch for containerized job execution.

### 🏗️ Architecture

> 📖 **Full line-by-line technical documentation:** [`assets/tech-doc.md`](./assets/tech-doc.md)
> — every module, the end-to-end data flow for all three artifact types, config
> schema, model usage, infrastructure, and diagrams.

#### Core Components
- **PaperSource** (`paper_source.py`): resolves an arXiv ID or paper PDF URL to a
  source; validates that URL sources actually serve a PDF.
- **ArxivHandler** (`arxiv_handler.py`): Paper metadata and content retrieval
- **Parser** (`parser.py`): HTML/PDF parsing with figure extraction
- **ContentExtractor** (`content_extractor.py`): Structured content and citation extraction
- **CodeRetriever** (`code_retriever.py`): Repository cloning and semantic code analysis
- **CitationSummarizer** (`citation_summarizer.py`): Reference paper analysis
- **ExplainerGraph** (`explainer.py`): Multi-stage LangGraph workflow for paper reviews
- **PaperSummarizer** (`summarizer.py`): five-section paper summaries
- **TechGuideGenerator** (`tech_guide.py`) + **WebResearcher** (`web_research.py`):
  research URLs (+ sub-pages + web search) and write a technical guide
- **Publisher** (`publisher.py`): artifact-agnostic S3 upload + blog PR
- **Paper Bot** (`slack/`): LLM intent parsing → AWS Batch dispatch

#### Infrastructure
- **AWS Batch**: Containerized job execution with ECS
- **Amazon Bedrock**: Claude models for analysis and generation
- **S3**: Paper storage and asset management
- **SSM Parameter Store**: Configuration management

### 🛠️ Tech Stack

- Python 3.12+, AWS CDK, Docker
- Amazon Bedrock, LangChain, LangGraph
- PyMuPDF, BeautifulSoup4, FAISS
- Pydantic validation, YAML configuration

### 📋 Configuration

Copy `scholar_lens/configs/config-template.yaml` to
`scholar_lens/configs/config.yaml` and edit it. Abbreviated example:

```yaml
resources:
  project_name: scholar-lens
  stage: dev
  profile_name: your-profile
  default_region_name: ap-northeast-2
  bedrock_region_name: us-west-2
  s3_bucket_name: your-bucket
  email_address: your-email@example.com
  github:
    enabled: true
    repo_name: owner/owner.github.io

paper:
  citation_extraction_model_id: anthropic.claude-sonnet-4-6
  table_of_contents_model_id: anthropic.claude-sonnet-4-6

explanation:
  paper_analysis_model_id: anthropic.claude-sonnet-4-6
  paper_synthesis_model_id: anthropic.claude-opus-4-8   # newest Claude Opus

summary:
  summary_model_id: anthropic.claude-opus-4-8

tech_guide:
  writing_model_id: anthropic.claude-opus-4-8

output_language: Korean
```

See [`assets/tech-doc.md`](./assets/tech-doc.md) §5 for the full config schema.

### 🚀 Usage

#### Infrastructure Deployment
```bash
# Deploy infrastructure
python scripts/deploy_infra.py
```

#### Development
```bash
# Install dependencies (including dev/test tools)
poetry install --with dev

# Set up environment
cp .env.template .env
# Edit .env with your configuration

# Paper REVIEW — --source accepts an arXiv ID or a paper PDF URL
python scholar_lens/main.py --source 2312.11805 --parse-pdf True
python scholar_lens/main.py --source https://openreview.net/pdf?id=XXXX

# Paper SUMMARY (concise five-section format)
python scholar_lens/main.py --source 2312.11805 --mode summarize

# Technical GUIDE / tutorial from documentation URLs
python scholar_lens/tech_guide_main.py \
  --urls https://docs.framework.io/start https://docs.framework.io/api \
  --search-queries "framework.io best practices"

# Submit a batch job (review or summarize)
python scripts/run_batch.py --source 2505.09388 --mode review \
  --repo-urls https://github.com/org/repo

# Run the Slack Paper Bot (Socket Mode; needs SLACK_BOT_TOKEN/SLACK_APP_TOKEN)
python -m scholar_lens.slack.bot
```

#### Testing & Quality
```bash
# Run the test suite (mocks all AWS/network — no credentials or cost)
poetry run pytest tests/ --cov=scholar_lens --cov-report=term-missing

# Lint & format
poetry run ruff check scholar_lens scripts tests
poetry run black --check scholar_lens scripts tests

# Validate infrastructure synthesis (no AWS account needed)
poetry run python scripts/ci_synth_check.py
```

CI (`.github/workflows/ci.yml`) runs lint/format, the pytest matrix
(Python 3.12 & 3.13), and a CDK synth check on every push and PR.
