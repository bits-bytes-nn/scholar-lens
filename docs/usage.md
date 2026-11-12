# Usage reference

Every CLI option, the survey plan file, resuming and republishing, outputs and exit codes, configuration and deployment details. The [README](../README.md) has the short version; [design.md](./design.md) (Korean) describes every stage and rule.

## Options

| Option | Commands | Effect |
| --- | --- | --- |
| `--repo-url URL` | review, summarize | Ground the text in the paper's code (repeatable) |
| `--parse-pdf` | review, summarize | Parse the PDF even when arXiv HTML exists |
| `--reference URL` | guide | Reference material to build on (repeatable, next to zero or more positional docs URLs, before or after them; a URL given as both counts as docs, listed once): a paper, design doc, deep-dive post or talk page as an HTML page or an arXiv link (other PDFs are not read); always used and listed, never treated as official docs |
| `--search-query Q` | guide | Web-search query (repeatable); replaces the queries the guide would plan itself |
| `--no-subpages` | guide | Do not follow in-scope sub-pages |
| `--fresh-sources` | guide | Gather the sources again instead of reusing the corpus a run with the same inputs saved within `tech_guide.source_snapshot_ttl_days` (default 7) |
| `--paper ID_OR_URL` | survey plan | A paper the survey must include: arXiv id or URL, or DOI (repeatable) |
| `--reference URL` | survey plan | Reference material to consult (repeatable): a blog post, survey or review article, or paper page as an HTML page or an arXiv link. Read once: excerpts shape the scope, and the papers it is or links to (arXiv abs/pdf/html and doi.org links, `arXiv:<id>` in its text) join the candidates, flagged `[REFERENCED by N]`; the number of references naming them breaks ties among candidates with equal topic evidence (in-pool citations and search hits), and they are not required. An unreadable reference is a plan note, not a failure. The plan keeps the references (`references[]`); `survey write` reads the articles again as explanatory material for the outline and the introduction, cross-cutting and insights sections, and lists them under `### 참고 자료` / `### Further reading` |
| `--since YEAR`, `--until YEAR` | survey plan | Only papers from these years (required papers excepted) |
| `--plan PATH_OR_S3_URI` | survey write | The plan file: a local path or `s3://<bucket>/<key>` in the configured bucket; with `--remote` a local plan is uploaded to `<s3_prefix>/surveys/<slug>/plan-<first 16 hex digits of its sha256>.json` first |
| `--no-pr` | review, summarize, guide, survey write, publish | Save (and upload) the post but open no PR |
| `--fresh` | review, summarize, guide, survey write | Compute every stage again: ignore the saved stage checkpoints and the evidence cache (both are still written), and for a guide gather the sources again (implies `--fresh-sources`) |
| `--remote` | review, summarize, guide, survey plan, survey write | Submit to AWS Batch and return the job id |
| `--wait`, `--timeout S` | status | Poll until the job finishes or `S` seconds pass (default 14400, four hours) |
| `--since D`, `--until D`, `--by kind\|stage\|model\|day`, `--billed` | costs | Group the Batch runs that finished in `[since, until)` (UTC days) by kind, stage, model or day from their result manifests, with what failed runs spent; `--billed` adds Cost Explorer's cost of resources tagged `ProjectName=<resources.project_name>` (the stack and the per-kind Bedrock inference profiles); activate `ProjectName` (and `Kind`) as cost allocation tags in Billing first |

`survey plan` writes no post: its `JobResult` carries `plan_path` (local `papers/survey-<slug>/plan.json`; empty for a `--remote` run, whose container is gone) and `s3_uri` (its S3 copy, the one to pass to `survey write --plan` after a remote run). Edit the plan (drop, add or move papers, change roles, rename lineages) and pass it to `survey write --plan`; a rerun of the plan stage keeps a plan that differs as `plan.<UTC time>.json`. A paper added by hand needs only `id` (arXiv id or URL, or DOI), `lineage` and `role`; the write stage fills in its title, authors, year and URL. A reference added by hand needs only `url` (its `kind`, `paper` for an arXiv or doi.org link and `article` otherwise, follows from it); only articles are read by the write stage. A plan needs at least one `core` paper. Each paper's evidence is cached (local `papers/survey-evidence/`, S3 `<s3_prefix>/surveys/evidence/`) by paper, text hash, reader version, model, language, role and source kind, so a rerun or another survey of the same papers reuses it. The plan's fields and the write stage are described in the [design reference](./design.md#43-논문-서베이).

## Resume and publish

The evidence engines save each stage's output under a hash of its inputs: the key targets, the outline, every accepted section, the whole-post result, the TL;DR and (reviews, summaries) the subtitle and description. A stage's key covers the ledger, the post kind, the settings the stage reads, the system prompts it uses (a prompt change invalidates it) and the output of the stage before it, so changing an input recomputes that stage and the ones after it only. A rerun with the same inputs therefore resumes by itself after a failure, an interruption or a cost-ceiling stop, loading the saved stages without model calls; a Batch retry (the job definition retries only when its host went away) resumes the same way from S3, where the checkpoints live next to the evidence cache (`<s3_prefix>/<papers|guides|surveys>/checkpoints/`). A loaded section is not checked again; its PR notes, blocking problems and claim-check verdicts come back with it. Pass `--fresh` to recompute everything. `--fresh` applies to the evidence engines only (guides, surveys, and reviews and summaries with `engine: evidence`); under the default `sections` engine for reviews and summaries it does nothing.

Every run saves its publish request (everything but the post text: title, branch key, PR title and body, blocking problems and review notes, the post's file name) as `<name>.request.json` next to the saved post or the rejected `*.rejected.md`, and in S3 next to its `posts/` or `rejected/` copy (a rejected post's images go to `rejected/blog-images/<slug>/`). `publish <markdown_path|s3_uri>` reads the post, possibly edited, and that request back, runs the publisher's post-processing and blog checks again and opens or updates the PR on the run's branch; `--no-pr` checks and saves it only. It runs locally (no `--remote`) and exits like a run: `0`, `1` when the post still breaks a rule (kept as `*.rejected.md` again), `2` when there is no such post or no request next to it, `5` when the PR failed. A request with a field this version does not know is refused. The run's blocking problems come back with the request and still block the PR; after fixing them in the post, `publish --resolved` turns them into checklist items to confirm instead. A run's result lists them in `blocking` (and, marked, first in `warnings`); a `--no-pr` run with blocking problems still succeeds.

Every command prints one JSON object on stdout (a run's `JobResult`, the `job_id` for `--remote`, the Batch state and stored result for `status`); logs go to stderr. Exit codes: `0` succeeded or submitted, `1` failed, `2` rejected input, `3` still running, `4` the command itself failed, `5` the artifact was saved but the requested blog PR failed (the result is `failed` with `pr_error`, `markdown_path` and `s3_uri`; also from `publish`); the Batch `worker` exits `10` (failed) or `11` (rejected) after sending its own notification, `4` when that notification could not be sent, and `143` when Batch stopped it ([details](./design.md#21-명령)). Quote URLs: in zsh, `?` in an unquoted URL is a glob and the command fails.

## Configuration

`scholar_lens/configs/config.yaml` (gitignored) is validated strictly: unknown keys are errors. The template lists every key with its default. Without the file a local run uses those defaults (no S3 bucket, GitHub off) and your default AWS credentials, so it still calls Bedrock and is billed. The ones you usually set:

- `resources`: `profile_name`, `bedrock_region_name`, `s3_bucket_name` (needed for `cdk deploy --all` and Batch; without it `status` shows the Batch state but no stored result), `email_address` (SNS notifications)
- `resources.github`: `enabled`, `repo_name` (e.g. `owner/owner.github.io`), `base_branch`, `branch_prefix`, `author_email` (the tool's commit identity; required for regeneration: without it a rerun replaces only a post with the same slug and leaves its older branch copies), and the category labels per artifact type
- per-stage `*_model_id`, `thinking_effort`, quality thresholds and cost caps: `max_total_tokens` for reviews and summaries on the `sections` engine and for survey plans (defaults 8M, 1M, 0.3M tokens), and for every engine a ceiling on the run's estimated cost (`max_cost_usd`: `tech_guide` $60, `explanation` $40, `summary` $12; `survey.write_max_cost_usd` $60; `null` for none). The ceiling counts the current attempt's spend only: stages a rerun or a Batch retry loads from its checkpoints cost nothing and are not counted again, so a run stopped at the ceiling and resumed may spend up to the ceiling once more on the stages left
- `explanation.engine`, `summary.engine`: `sections` (default) or `evidence`, with `reader_model_id` and `min_total_chars`/`max_total_chars` (review 25,000/55,000, summary 2,500/4,500)
- Evidence cache: the evidence engines store what their reading agents recorded per source, locally under `papers/paper-evidence/`, `papers/guide-evidence/` and `papers/survey-evidence/` and in S3 under `<s3_prefix>/papers/evidence/`, `<s3_prefix>/guides/evidence/` and `<s3_prefix>/surveys/evidence/`, keyed by the source text's hash, reader version, reading model and language. A rerun reuses it; cached items that fail today's checks are dropped, and a source is read again when none pass. The same stores hold the stage checkpoints under `checkpoints/` (see [Resume and publish](#resume-and-publish)); `--fresh` ignores both for one run ([details](./design.md#45-근거-장부-엔진-공통))
- `tech_guide.run_examples` (default `false`): run the guide's Python examples in an AgentCore Code Interpreter sandbox and fix a block that fails as pasted; `run_block_timeout_s` (60) and `run_timeout_s` (600, at most 840: one synchronous Code Interpreter call may take 15 minutes) bound a block and the program, `code_interpreter_id` picks the interpreter of a local run (a Batch job uses the stack's)
- `survey`: `min_papers`/`max_papers` (6/25), `candidate_pool` (60); for the write stage `chars_per_core_paper`/`chars_per_supporting_paper` (3,000/800) within `min_total_chars`/`max_total_chars` (20,000/70,000)
- `resources.github.survey_category` (`Paper Surveys`): must be a type in the blog's post conventions, or `survey write` stops before any model call
- `output_language` (default `Korean`)

## Deploy to AWS

Needs Node.js, Docker and the AWS CDK CLI in addition to the above, and an existing S3 bucket named in `resources.s3_bucket_name` (the stack does not create it; use one dedicated to Scholar Lens, since the job role may list the whole bucket). Pass the same profile as `resources.profile_name`:

```bash
cdk bootstrap --profile <profile_name>           # once per account/region
cdk deploy --all --profile <profile_name>        # cdk.json runs: poetry run python scripts/deploy_infra.py
poetry run python scripts/put_secrets.py         # write .env secrets to SSM SecureString
poetry run python scripts/put_secrets.py github-token   # after rotating one secret
```

Batch jobs use the `config.yaml` baked into the image at the last deploy, so re-run `cdk deploy --all` after editing it ([stack details](./design.md#10-인프라)).

Batch jobs do not trace to LangSmith, even when tracing is on in the shell that deploys. To trace them, set `SCHOLAR_LENS_BATCH_TRACING=true` when running `cdk deploy --all`; the jobs then get `LANGCHAIN_TRACING_V2=true` and their prompts, which hold paper and blog text, go to LangSmith.

The stack's hourly cost alarm covers Batch runs only: local runs send no CloudWatch metrics, so check their `estimated_cost_usd` in the printed result ([alarms](./design.md#11-관측성)).
