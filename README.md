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
| **Review** | arXiv id or paper PDF URL, optional code repos | A section-by-section review for data scientists and ML engineers: TL;DR, intuition before math, worked examples, figures, limitations |
| **Summary** | arXiv id or paper PDF URL | A TL;DR and five short sections: background, method, implementation, results, significance and limits |
| **Tech guide** | Documentation URLs, optional reference material | A grounded guide to usage, internals and architecture, and fitting the tool into a larger system |
| **Paper survey** | A research topic, then the edited plan | `survey plan`: an editable list of the topic's major papers by lineage. `survey write`: a long-form survey with figures, equations, a timeline and a comparison table |

Every post is Jekyll Markdown in the blog's conventions and is checked before it is saved: a post that breaks a rule is rejected, not published. Posts go to a local file, optionally S3 and a pull request on the blog repository.

## How it works

![Generation pipelines](./docs/diagrams/pipeline.png)

Guides, surveys and (optionally) reviews and summaries are written by one evidence-ledger engine:

1. Reading agents record evidence from each source, quoted verbatim and checked in code.
2. An outline gives each fact to one section, in the post kind's order.
3. Each section is drafted from its evidence and checked claim by claim; unsupported sentences are fixed or cut.
4. Whole-post checks (consistency, repetition, coverage, code that runs as pasted), then the TL;DR, closing and description.

Reviews and summaries can also use the sections engine (the default), which plans the paper's coverage and writes and revises each section from its brief. Publishing normalizes math and Markdown, copies the paper's own figures, runs the blog-conventions gate and opens a PR on a fixed branch per post.

![AWS architecture](./docs/diagrams/aws-architecture.png)

[docs/design.md](./docs/design.md) (Korean) describes every stage and rule; [docs/engine.md](./docs/engine.md) the engine.

## Quick start

Prerequisites:

- Python 3.12+ and [Poetry](https://python-poetry.org/)
- An AWS profile with Bedrock access to the configured Claude models
- `UPSTAGE_API_KEY` for PDF parsing
- Optional: `TAVILY_API_KEY` or `BRAVE_API_KEY` (guide web research), `GITHUB_TOKEN` (blog PRs), `CROSSREF_MAILTO`, `OPENALEX_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY` (citation lookups)

```bash
git clone https://github.com/bits-bytes-nn/scholar-lens.git && cd scholar-lens
poetry install --with dev
cp scholar_lens/configs/config-template.yaml scholar_lens/configs/config.yaml   # edit it
cp .env.template .env                                                           # fill in keys
poetry run python -m scholar_lens summarize 2106.09685 --no-pr
```

## Usage

```bash
poetry run python -m scholar_lens review 2106.09685 --repo-url "https://github.com/microsoft/LoRA"
poetry run python -m scholar_lens summarize "https://openreview.net/pdf?id=nZeVKeeFYf9" --no-pr
poetry run python -m scholar_lens guide "https://docs.vllm.ai/en/latest/getting_started/quickstart.html" \
  --reference "https://arxiv.org/abs/2309.06180"
poetry run python -m scholar_lens survey plan "preference optimization for LLM alignment" --paper 2305.18290
poetry run python -m scholar_lens survey write --plan papers/survey-<slug>/plan.json
poetry run python -m scholar_lens review 2106.09685 --remote     # run on AWS Batch
poetry run python -m scholar_lens status <job-id> --wait
poetry run python -m scholar_lens publish papers/<dir>/<post>.rejected.md   # after editing a rejected post
```

- Each command prints one JSON result on stdout; logs go to stderr.
- `--no-pr` saves the post without opening a PR; `--remote` submits the run to AWS Batch.
- A rerun with the same inputs resumes from the stages it saved; `--fresh` recomputes everything.
- Quote URLs: in zsh an unquoted `?` is a glob.

[docs/usage.md](./docs/usage.md) lists every option, the survey plan format, resuming and republishing, exit codes and configuration keys.

Agents (OpenClaw, Claude Code, Codex) call the same CLI through the skill in `~/Projects/agent-config/skills/scholar-lens/SKILL.md`.

## Configuration

`scholar_lens/configs/config.yaml` (gitignored) is validated strictly, and the template lists every key with its default. The ones you usually set:

- `resources`: AWS profile, Bedrock region, S3 bucket, notification email
- `resources.github`: blog repository, branch prefix, `author_email` (needed to regenerate a published post in place)
- Model ids, thinking effort and a cost ceiling per artifact (`max_cost_usd`)
- `explanation.engine`, `summary.engine`: `sections` (default) or `evidence`
- `output_language` (default `Korean`)

## Deploy to AWS

Needs Node.js, Docker, the AWS CDK CLI and an existing S3 bucket named in `resources.s3_bucket_name`:

```bash
cdk bootstrap --profile <profile_name>       # once per account and region
cdk deploy --all --profile <profile_name>
poetry run python scripts/put_secrets.py     # .env secrets to SSM
```

Batch jobs use the `config.yaml` baked into the image at the last deploy; redeploy after editing it. See [docs/usage.md](./docs/usage.md#deploy-to-aws) for tracing, alarms and the stack.

## Development

```bash
poetry run pytest tests/          # no AWS or network
poetry run ruff check scholar_lens scripts tests
poetry run black --check scholar_lens scripts tests
poetry run mypy scholar_lens scripts
poetry run lint-imports           # layer contracts
```

CI runs these plus coverage, pip-audit, a CDK synth check and a Docker build. Changes are listed in [CHANGELOG.md](./CHANGELOG.md).

## License

See [LICENSE](./LICENSE).
