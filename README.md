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
| **Tech guide** | Documentation URLs and/or reference material (papers, design docs, deep-dive posts) | A source-grounded guide covering installation and usage, how the internals work and why, what each major feature means, and how to fit the tool into a larger system; researched across sub-pages and web search |

Every post is Jekyll Markdown that follows the target blog's conventions, and is checked before it is saved: a post that breaks a rule is rejected instead of published. Posts go to a local file, optionally S3, and optionally a pull request on the blog repository.

## How it works

![Generation pipelines](./docs/diagrams/pipeline.png)

- **Review:** a planner reads the whole paper and writes a coverage plan (sections, must-cover items, difficulty, prerequisites, concept ownership, figures, glossary). Each section is written from its brief, checked against it, and revised until nothing is missing; one editing pass then fixes terminology, repetition and gaps across the article.
- **Summary:** one long-context call over the parsed paper.
- **Guide:** research (the docs, the reference material, and web searches for usage, internals and design, and production integration) → relevance gate → outline → per-section write, evaluate and fact-check. Official docs decide commands, flags and defaults; papers, design docs and reputable write-ups may ground concepts, internals and architecture.
- **Publishing:** math and Markdown normalization, copying the paper's own figures into the blog (any other image stays a link to its source), the blog-conventions gate, then S3 and a GitHub PR on a fixed branch per post.

![AWS architecture](./docs/diagrams/aws-architecture.png)

See [docs/design.md](./docs/design.md) for every stage, rule and configuration key.

## Quick start

Prerequisites:

- Python 3.12+ and [Poetry](https://python-poetry.org/)
- An AWS profile with Bedrock access to the configured models in `bedrock_region_name` (defaults: Claude Opus 5.5, Sonnet 5.5, Haiku 5.5)
- `UPSTAGE_API_KEY` for PDF parsing (non-arXiv sources and the HTML fallback)
- Optional: `TAVILY_API_KEY` or `BRAVE_API_KEY` for guide web research, `GITHUB_TOKEN` for blog PRs, `CROSSREF_MAILTO` (Crossref and OpenAlex polite pools), `OPENALEX_API_KEY` (raises OpenAlex's keyless budget of about 100 searches per day per IP) and `SEMANTIC_SCHOLAR_API_KEY` (adds Semantic Scholar to citation lookups; skipped without a key)

```bash
git clone https://github.com/bits-bytes-nn/scholar-lens.git && cd scholar-lens
poetry install --with dev
cp scholar_lens/configs/config-template.yaml scholar_lens/configs/config.yaml   # edit it
cp .env.template .env                                                           # fill in keys
poetry run python -m scholar_lens summarize 2106.09685 --no-pr
```

## Configuration

`scholar_lens/configs/config.yaml` (gitignored) is validated strictly: unknown keys are errors. The template lists every key with its default. Without the file a local run uses those defaults (no S3 bucket, GitHub off) and your default AWS credentials, so it still calls Bedrock and is billed. The ones you usually set:

- `resources`: `profile_name`, `bedrock_region_name`, `s3_bucket_name` (needed for `cdk deploy` and Batch; without it `status` shows the Batch state but no stored result), `email_address` (SNS notifications)
- `resources.github`: `enabled`, `repo_name` (e.g. `owner/owner.github.io`), `base_branch`, `branch_prefix`, and the category labels per artifact type
- per-stage `*_model_id`, `thinking_effort`, quality thresholds and `max_total_tokens` cost caps (defaults: review 8M, summary 1M, guide 3M tokens)
- `output_language` (default `Korean`)

## CLI

```bash
poetry run python -m scholar_lens review 2106.09685 --repo-url "https://github.com/microsoft/LoRA"
poetry run python -m scholar_lens summarize "https://openreview.net/pdf?id=nZeVKeeFYf9" --no-pr
poetry run python -m scholar_lens guide "https://docs.vllm.ai/en/latest/getting_started/quickstart.html" \
  "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html" \
  --reference "https://arxiv.org/abs/2309.06180" \
  --reference "https://blog.vllm.ai/2025/01/27/v1-alpha-release.html"
poetry run python -m scholar_lens review 2106.09685 --remote     # submit to AWS Batch
poetry run python -m scholar_lens status <job-id> --wait         # poll a remote job
```

| Option | Commands | Effect |
| --- | --- | --- |
| `--repo-url URL` | review, summarize | Ground the text in the paper's code (repeatable) |
| `--parse-pdf` | review, summarize | Parse the PDF even when arXiv HTML exists |
| `--reference URL` | guide | Reference material to build on (repeatable, next to zero or more positional docs URLs, before or after them; a URL given as both counts as docs, listed once): a paper, design doc, deep-dive post or talk page as an HTML page or an arXiv link (other PDFs are not read); always used and listed, never treated as official docs |
| `--search-query Q` | guide | Web-search query (repeatable); replaces the queries the guide would plan itself |
| `--no-subpages` | guide | Do not follow in-scope sub-pages |
| `--no-pr` | review, summarize, guide | Save (and upload) the post but open no PR |
| `--remote` | review, summarize, guide | Submit to AWS Batch and return the job id |
| `--wait`, `--timeout S` | status | Poll until the job finishes or `S` seconds pass |

Every command prints one JSON object on stdout (a run's `JobResult`, the `job_id` for `--remote`, the Batch state and stored result for `status`); logs go to stderr. Exit codes: `0` succeeded or submitted, `1` failed, `2` rejected input, `3` still running, `4` the command itself failed ([details](./docs/design.md#21-명령)). Quote URLs: in zsh, `?` in an unquoted URL is a glob and the command fails.

## Agents (OpenClaw, Claude Code, Codex)

The same CLI is the agent interface. The skill `~/.openclaw/workspace/skills/scholar-lens/SKILL.md` tells an agent how to call it, when to use `--remote`, and that `--no-pr` is the default unless the user asked to publish. A CLI plus skill was chosen over an MCP server because runs take minutes to an hour: `--remote` + `status` already gives an asynchronous contract, and the skill directory is shared with Claude Code and Codex.

## Blog conventions gate

Before a post is saved it is checked against the blog's `_data/post_conventions.yml` (copied to `assets/post_conventions.yml`): errors stop publishing and keep the post as `*.rejected.md`, warnings become a PR checklist ([rules](./docs/design.md#6-블로그-규칙과-검증)).

## Deploy to AWS

Needs Node.js, Docker and the AWS CDK CLI in addition to the above, and an existing S3 bucket named in `resources.s3_bucket_name` (the stack does not create it; use one dedicated to Scholar Lens, since the job role may list the whole bucket). Pass the same profile as `resources.profile_name`:

```bash
cdk bootstrap --profile <profile_name>           # once per account/region
cdk deploy --profile <profile_name>              # cdk.json runs: poetry run python scripts/deploy_infra.py
poetry run python scripts/put_secrets.py         # write .env secrets to SSM SecureString
poetry run python scripts/put_secrets.py github-token   # after rotating one secret
```

Batch jobs use the `config.yaml` baked into the image at the last deploy, so re-run `cdk deploy` after editing it ([stack details](./docs/design.md#10-인프라)).

The stack's hourly cost alarm covers Batch runs only: local runs send no CloudWatch metrics, so check their `estimated_cost_usd` in the printed result ([alarms](./docs/design.md#11-관측성)).

## Development

```bash
poetry run pytest tests/ -W ignore::DeprecationWarning --cov=scholar_lens --cov=scripts --cov-fail-under=85   # as in CI; no AWS or network
poetry run ruff check scholar_lens scripts tests
poetry run black --check scholar_lens scripts tests
poetry run mypy scholar_lens scripts
poetry run python scripts/ci_synth_check.py      # CDK synth with a fake account
actionlint                                       # lint the GitHub workflows
```

## Changelog

- After upgrading, redeploy (`cdk deploy`) before using `--remote`: the Batch image runs the code of the last deploy, and an older worker rejects a request with a field it does not know (such as `--reference`).
- Guides go deeper and broader: usage, how the internals work and why, and how to fit the tool into a larger system. An outline-level section plan assigns each concept to one owning section, the evaluator's blocking issues (factual errors, contradictions, unsupported flags or defaults) force a revision, and `tech_guide.max_revision_attempts` now defaults to one revision. A run costs more than before.
- `guide --reference URL` (repeatable) adds reference material (papers, design docs, deep-dive posts) next to the docs URLs, and a guide can be built from references alone. Docs URLs may now come before or after the options.
- New guide defaults: `tech_guide.fetch_top_results` is now 12 (was 4) and `tech_guide.max_sections` caps the outline. A `config.yaml` copied from an older template keeps its old values; edit them to pick up the new defaults.
- Paper posts carry `paper_doi` front matter (the arXiv DOI for arXiv papers, else the paper's DOI when known); the blog looks the citation count up on Semantic Scholar.
- Citation lookups: OpenAlex follows Crossref in the chain (keyless use allows about 100 searches per day per IP; set the optional `OPENALEX_API_KEY` to raise it), Crossref picks the matching record among its top five hits, and Semantic Scholar is queried only when `SEMANTIC_SCHOLAR_API_KEY` is set.
- Removed `citations.citation_analysis_model_id`: delete it from an existing `config.yaml`, or loading fails with `extra_forbidden`.
- Only the paper's own figures are copied into the blog (arXiv HTML figures, figures recovered from the arXiv source, figures cropped from the PDF). Any other image (docs sites, blogs, reference material) stays a link to its source and is listed on the PR checklist as `remote image not localized`. A paper figure that cannot be stored blocks the post; the run result's `markdown_path` and `s3_uri` then point to the rejected draft.
- ar5iv is never used: an ar5iv link fails validation, and an arXiv figure that comes back as ar5iv's "NO IMAGE AVAILABLE" placeholder is taken from the paper's arXiv source (`/e-print`).
- A `glightbox` link must open its own image (`href` equal to the inner `src`), or the post is rejected.
- Paper posts carry a generated `description` (120-160 characters of plain text that open with the model or method name and are unique on the blog). Its numbers are checked against the TL;DR and the abstract, and the TL;DR's numbers against the paper; a mismatch feeds the review's repair attempt or becomes a PR note, and only a description that is not unique on the blog blocks the post. Scaled and combined Korean numbers are compared as quantities (`12만 8천` matches `128K`); a `만` read as the particle "only" (`5만 사용`, `10만큼`) is not a unit.
- Rerunning a published post lists on the PR the hand edits made to it or its images since it was generated, so they can be carried over.
- Reference and source lists are `[title](url)` lines, one per URL; a guide's sources use the fetched page titles.
- Guide section lengths count prose only: code, table rows and images are excluded.
- One terminology rule in every prompt: the translation guideline's rendering wins for each term it lists (guides now receive the guideline too); other terms are translated where natural, with the English in parentheses at first use. The lint flags any 하다/되다 form glued to an English word (`prefill하면서`, `swap-in되어`).

## License

See [LICENSE](./LICENSE).
