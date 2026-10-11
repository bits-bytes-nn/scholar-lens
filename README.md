<div align="center">

# Scholar-Lens

**Turn AI/ML papers and documentation into blog-ready posts: in-depth reviews, concise summaries, and hands-on technical guides.**

Runs locally or on AWS Batch, driven by a CLI that people and agents (OpenClaw, Claude Code, Codex) share. Powered by Claude on Amazon Bedrock.

[![CI](https://github.com/bits-bytes-nn/scholar-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/scholar-lens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock%20(Claude)-green)

[한국어 README](./README.ko.md) · [Design reference](./docs/design.md)

![Sample](./assets/sample.png)

</div>

## What it does

| Artifact | Input | Output |
| --- | --- | --- |
| **Review** | arXiv id or paper PDF URL (+ optional code repos) | Section-by-section review for a general DS/MLE reader: easy parts short, hard parts built up from intuition to math to a worked example, with figures, tables, a TL;DR and a limitations section |
| **Summary** | arXiv id or paper PDF URL | A TL;DR plus five short sections (motivation, idea, implementation, results, significance and limits) |
| **Tech guide** | One or more documentation URLs | A source-grounded tutorial, researched across sub-pages and web search |

Every post is Jekyll Markdown that follows the target blog's conventions, and is checked before it is saved: a post that breaks a rule is rejected instead of published. Posts go to a local file, optionally S3, and optionally a pull request on the blog repository.

## How it works

![Generation pipelines](./docs/diagrams/pipeline.png)

- **Review:** a planner reads the whole paper and writes a coverage plan (sections, must-cover items, difficulty, prerequisites, concept ownership, figures, glossary). Each section is written from its brief, checked against it, and revised until nothing is missing; one editing pass then fixes terminology, repetition and gaps across the article.
- **Summary:** one long-context call over the parsed paper.
- **Guide:** research → relevance gate → outline → per-section write, evaluate and fact-check.
- **Publishing:** math and Markdown normalization, figure localization, the blog-conventions gate, then S3 and a GitHub PR on a fixed branch per post.

![AWS architecture](./docs/diagrams/aws-architecture.png)

See [docs/design.md](./docs/design.md) for every stage, rule and configuration key.

## Quick start

Prerequisites:

- Python 3.12+ and [Poetry](https://python-poetry.org/)
- An AWS profile with Bedrock access to the configured models in `bedrock_region_name` (defaults: Claude Opus 5.5, Sonnet 5.5, Haiku 5.5)
- `UPSTAGE_API_KEY` for PDF parsing (non-arXiv sources and the HTML fallback)
- Optional: `TAVILY_API_KEY` or `BRAVE_API_KEY` for guide web research, `GITHUB_TOKEN` for blog PRs

```bash
git clone https://github.com/bits-bytes-nn/scholar-lens.git && cd scholar-lens
poetry install --with dev
cp scholar_lens/configs/config-template.yaml scholar_lens/configs/config.yaml   # edit it
cp .env.template .env                                                           # fill in keys
poetry run python -m scholar_lens summarize 2106.09685 --no-pr
```

## Configuration

`scholar_lens/configs/config.yaml` (gitignored) is validated strictly: unknown keys are errors. The template lists every key with its default. The ones you usually set:

- `resources`: `profile_name`, `bedrock_region_name`, `s3_bucket_name` (needed for Batch and `status`), `email_address` (SNS notifications)
- `resources.github`: `enabled`, `repo_name` (e.g. `owner/owner.github.io`), `base_branch`, `branch_prefix`, and the category labels per artifact type
- per-stage `*_model_id`, `thinking_effort`, quality thresholds and optional `max_total_tokens` cost caps
- `output_language` (default `Korean`)

## CLI

```bash
poetry run python -m scholar_lens review 2106.09685 --repo-url https://github.com/microsoft/LoRA
poetry run python -m scholar_lens summarize https://openreview.net/pdf?id=nZeVKeeFYf9 --no-pr
poetry run python -m scholar_lens guide https://docs.vllm.ai/en/latest/ --search-query "vLLM paged attention"
poetry run python -m scholar_lens review 2106.09685 --remote     # submit to AWS Batch
poetry run python -m scholar_lens status <job-id> --wait         # poll a remote job
```

| Option | Commands | Effect |
| --- | --- | --- |
| `--repo-url URL` | review, summarize | Ground the text in the paper's code (repeatable) |
| `--parse-pdf` | review, summarize | Parse the PDF even when arXiv HTML exists |
| `--search-query Q` | guide | Extra web-search query (repeatable) |
| `--no-subpages` | guide | Do not follow in-scope sub-pages |
| `--no-pr` | review, summarize, guide | Save (and upload) the post but open no PR |
| `--remote` | review, summarize, guide | Submit to AWS Batch and return the job id |
| `--wait`, `--timeout S` | status | Poll until the job finishes or `S` seconds pass |

Every command prints one JSON object on stdout; logs go to stderr. A run returns a `JobResult` (`status`, `title`, `markdown_path`, `s3_uri`, `pr_url`, `warnings`, `violations`, `error`, `estimated_cost_usd`, …); `--remote` returns the `job_id`; `status` returns the Batch state and, once finished, the stored result. Exit codes: `0` succeeded or submitted, `1` failed (including a post rejected by the conventions gate), `2` rejected input (not a PDF, unsafe URL, unknown arXiv id, non-technical pages), `3` still running (`status`, or `status --wait` timed out).

## Agents (OpenClaw, Claude Code, Codex)

The same CLI is the agent interface. The skill `~/.openclaw/workspace/skills/scholar-lens/SKILL.md` tells an agent how to call it, when to use `--remote`, and that `--no-pr` is the default unless the user asked to publish. A CLI plus skill was chosen over an MCP server because runs take minutes to an hour: `--remote` + `status` already gives an asynchronous contract, and the skill directory is shared with Claude Code and Codex.

## Blog conventions gate

The blog owns its rules in `_data/post_conventions.yml`; `assets/post_conventions.yml` is a copy (CI fails when they drift, and runs read the blog's base-branch copy when GitHub is enabled). Before a post is saved, `scholar_lens/src/post_validator.py` checks it: front matter (`paper_author`, `subtitle`, `seo_title`, categories, cover, tags), headings (`## TL;DR`, no H1), `$$` math, table shape, image locations, leaked generator tags. Errors stop publishing (the post is kept as `*.rejected.md` next to the work files and listed in `violations`); warnings become a checklist in the PR body.

## Deploy to AWS

Needs Node.js, Docker and the AWS CDK CLI in addition to the above.

```bash
cdk bootstrap                                    # once per account/region
cdk deploy                                       # cdk.json runs: poetry run python scripts/deploy_infra.py
poetry run python scripts/put_secrets.py         # write .env secrets to SSM SecureString
```

The stack runs one Batch job definition (`python3 -m scholar_lens worker --request Ref::request`) on on-demand and spot compute, writes results to `s3://<bucket>/<prefix>/results/<job_id>.json`, and notifies SNS on success and failure. `cdk synth`/`diff` never touch secrets.

## Development

```bash
poetry run pytest                                # unit and pipeline tests, no AWS or network
poetry run ruff check scholar_lens scripts tests
poetry run black --check scholar_lens scripts tests
poetry run mypy scholar_lens scripts
poetry run python scripts/ci_synth_check.py      # CDK synth with a fake account
```

## License

See [LICENSE](./LICENSE).
