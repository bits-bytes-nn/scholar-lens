<div align="center">

# Scholar-Lens

**Turn AI/ML papers and documentation into blog-ready posts: in-depth reviews, concise summaries, hands-on technical guides, and surveys of a research topic's papers.**

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
| **Paper survey** | A research topic (+ optional required papers, reference material to consult and years), then the plan file | `survey plan`: an editable JSON plan of the topic's major papers (OpenAlex, arXiv) in chronological lineages with roles, reasons and citation links. `survey write`: a long-form post in the style of Lilian Weng's blog: each lineage's papers explained and connected, with the papers' figures and key equations, a timeline and a comparison table, insights and author-year citations |

Every post is Jekyll Markdown that follows the target blog's conventions, and is checked before it is saved: a post that breaks a rule is rejected instead of published. Posts go to a local file, optionally S3, and optionally a pull request on the blog repository.

## How it works

![Generation pipelines](./docs/diagrams/pipeline.png)

- **Review:** a planner reads the whole paper and writes a coverage plan (sections, must-cover items, difficulty, prerequisites, concept ownership, figures, glossary). Each section is written from its brief, checked against it, and revised until nothing is missing; one editing pass then fixes terminology, repetition and gaps across the article.
- **Summary:** one long-context call over the parsed paper.
- **Guide:** research (the docs, the reference material, and web searches for usage, internals and design, and production integration) → relevance gate → outline → per-section write, evaluate and fact-check. Official docs decide commands, flags and defaults; papers, design docs and reputable write-ups may ground concepts, internals and architecture.
- **Survey plan:** the user's reference material (blog posts, survey articles, paper pages) is read once; a scope call proposes lineages, search queries and landmark titles from the topic and excerpts of that material; OpenAlex and arXiv searches (arXiv's metadata checked via DataCite, Semantic Scholar counts with a key) build a candidate pool, joined by the papers the references are or link to (arXiv and doi.org links, `arXiv:` ids), that code merges, links and ranks; a selection call picks the papers within the configured bounds, and code validates the choice before the plan is saved.
- **Survey write:** the (edited) plan is validated again and hand-added papers are completed from DataCite/OpenAlex; each paper becomes a cached card (one call over a core paper's arXiv HTML full text or a supporting paper's abstract) whose numbers, equation and figures code checks against the paper; an outline call plans the sections (introduction, one per lineage, optional cross-cutting ones, comparison, insights) whose lengths code derives from the paper counts; each section is written, evaluated, revised once and fact-checked against its cards like a guide section; the sections that frame the field (introduction, cross-cutting, comparison, insights) also get excerpts of the plan's reference articles as explanatory material (never a source of paper facts; a view is attributed to its article by name); code renders `[@id]` citation keys as author-year links and builds the timeline, comparison table, reference list and a further-reading list of the reference articles.
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
- per-stage `*_model_id`, `thinking_effort`, quality thresholds and `max_total_tokens` cost caps (defaults: review 8M, summary 1M, guide 3M, survey plan 0.3M, survey write 4.5M tokens via `survey.write_max_total_tokens`)
- `survey`: `min_papers`/`max_papers` (6/25), `candidate_pool` (60); for the write stage `chars_per_core_paper`/`chars_per_supporting_paper` (3,000/800) within `min_total_chars`/`max_total_chars` (20,000/70,000) and `card_source_chars`
- `resources.github.survey_category` (`Paper Surveys`): must be a type in the blog's post conventions, or `survey write` stops before any model call
- `output_language` (default `Korean`)

## CLI

```bash
poetry run python -m scholar_lens review 2106.09685 --repo-url "https://github.com/microsoft/LoRA"
poetry run python -m scholar_lens summarize "https://openreview.net/pdf?id=nZeVKeeFYf9" --no-pr
poetry run python -m scholar_lens guide "https://docs.vllm.ai/en/latest/getting_started/quickstart.html" \
  "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html" \
  --reference "https://arxiv.org/abs/2309.06180" \
  --reference "https://blog.vllm.ai/2025/01/27/v1-alpha-release.html"
poetry run python -m scholar_lens survey plan "preference optimization for LLM alignment" \
  --paper 2305.18290 --since 2017 \
  --reference "https://huggingface.co/blog/rlhf" \
  --reference "https://arxiv.org/abs/2312.14925"                # writes papers/survey-<slug>/plan.json
poetry run python -m scholar_lens survey write \
  --plan papers/survey-preference-optimization-for-llm-alignment/plan.json --no-pr
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
| `--paper ID_OR_URL` | survey plan | A paper the survey must include: arXiv id or URL, or DOI (repeatable) |
| `--reference URL` | survey plan | Reference material to consult (repeatable): a blog post, survey or review article, or paper page as an HTML page or an arXiv link. Read once: excerpts shape the scope, and the papers it is or links to (arXiv abs/pdf/html and doi.org links, `arXiv:<id>` in its text) join the candidates, flagged `[REFERENCED by N]`; the number of references naming them breaks ties among candidates with equal topic evidence (in-pool citations and search hits), and they are not required. An unreadable reference is a plan note, not a failure. The plan keeps the references (`references[]`); `survey write` reads the articles again for the introduction, cross-cutting, comparison and insights sections and lists them under `### 참고 자료` / `### Further reading` |
| `--since YEAR`, `--until YEAR` | survey plan | Only papers from these years (required papers excepted) |
| `--plan PATH_OR_S3_URI` | survey write | The plan file: a local path or `s3://<bucket>/<key>` in the configured bucket; with `--remote` a local plan is uploaded to `<s3_prefix>/surveys/<slug>/plan-<sha256>.json` first |
| `--no-pr` | review, summarize, guide, survey write | Save (and upload) the post but open no PR |
| `--remote` | review, summarize, guide, survey plan, survey write | Submit to AWS Batch and return the job id |
| `--wait`, `--timeout S` | status | Poll until the job finishes or `S` seconds pass |

`survey plan` writes no post: its `JobResult` carries `plan_path` (local `papers/survey-<slug>/plan.json`; empty for a `--remote` run, whose container is gone) and `s3_uri` (its S3 copy, the one to pass to `survey write --plan` after a remote run). Edit the plan (drop, add or move papers, change roles, rename lineages) and pass it to `survey write --plan`; a rerun of the plan stage keeps a plan that differs as `plan.<UTC time>.json`. A paper added by hand needs only `id` (arXiv id or URL, or DOI), `lineage` and `role`; the write stage fills in its title, authors, year and URL. A reference added by hand needs only `url` (its `kind`, `paper` for an arXiv or doi.org link and `article` otherwise, follows from it); only articles are read by the write stage. A plan needs at least one `core` paper. Paper cards are cached (local `papers/survey-cards/`, S3 `<s3_prefix>/surveys/cards/`) by paper, card schema, model, language and a `-<variant>` digest of the card prompt and `card_source_chars` (changing either writes the cards again), so a rerun or another survey of the same papers reuses them. The plan's fields and the write stage are described in the [design reference](./docs/design.md#43-논문-서베이).

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

- `survey plan --reference URL` (repeatable; `reference_urls` on survey_plan requests) is back, and now used: the plan stage reads each reference for the scope call and adds the papers it is or links to as candidates (a tie-break after topic evidence in the rank, `[REFERENCED by N]` in the shortlist); the plan stores them as `references[]` (`schema_version` 3; version 1 and 2 plans still load); the write stage uses the reference articles as explanatory material in the sections that frame the field and lists them as further reading after `### References`. Redeploy before using it with `--remote`.

- New `survey write --plan <file>` (kind `survey_write`): turns a (possibly edited) plan into a `Paper Surveys` post with cached paper cards, an outline, guide-style section writing and checks, deterministic timeline and comparison tables, the papers' figures and author-year citations. It needs the blog's post conventions to list the `Paper Surveys` type (`resources.github.survey_category`); until the blog has that type on `main`, the bundled `assets/post_conventions.yml` stays as it is (the copy check would fail) and the tests use conventions with the type added. The guide's evaluate-and-revise loop and grounding retry are now shared functions (`tech_guide.revise_section`, `checked_grounding`), with unchanged behaviour. Redeploy before using it with `--remote`.
- New `survey plan "<topic>"` (kind `survey_plan`): finds a topic's major papers and saves an editable plan (`plan_path`, `s3_uri`). Without `SEMANTIC_SCHOLAR_API_KEY` citation counts are OpenAlex's, which undercount arXiv-era ML papers, and few citation links are known. Redeploy before using it with `--remote`.
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
