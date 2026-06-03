# Scholar-Lens Quality Rubric

A static, file:line-evidence rubric used by the LLM-judge to gate each phase of
the Scholar-Lens overhaul. Every deduction must cite concrete `file:line`
evidence. Scores are 0–10 per criterion.

**Scoring anchors**
- **10** — Exemplary; a senior staff engineer would approve as-is.
- **9** — Excellent; only cosmetic nits.
- **8** — Good; 1–2 substantive but non-blocking issues.
- **6–7** — Acceptable; several real issues.
- **≤5** — Needs rework.

**Gate target:** every criterion ≥ 8.5; none < 8.0.

---

## Phase 1 — Refactoring, Clean Code & AWS Well-Architected
- **A. Clean code & architecture** — SOLID, cohesion, sensible abstractions, no
  god-functions, clear naming.
- **B. No hardcoding / overfitting** — model IDs, thresholds, category/cover
  mappings live in config/constants; no locale-specific output hacks.
- **C. Latest LLM models** — newest Claude (incl. Opus 4.8) wired consistently;
  capability flags (1M context, thinking, caching) internally consistent.
- **D. Dead code removed** — no unused symbols, stale TODOs, unreachable branches.
- **E. Source generality** — accepts arXiv IDs *and* arbitrary paper PDF URLs;
  rejects non-PDF URLs; no hidden arXiv coupling.
- **F. AWS Well-Architected security** — least-privilege IAM (no `*FullAccess`),
  encryption (KMS), log retention, scoped egress, honest secret handling.

## Phase 2 — Test Suite & CI/CD
- **T1. Coverage breadth** — the right modules are tested (pure/core logic);
  LLM-orchestration may be out of scope if documented.
- **T2. Test depth** — edge cases, error paths, boundaries, behavior not impl.
- **T3. Isolation** — no live AWS/network; moto/responses; deterministic.
- **T4. CI pipeline** — ruff + black + mypy + pytest matrix + infra synth; mypy
  is a real (passing) gate, not a silent no-op.
- **T5. Ergonomics** — fixtures, fast (<60s), no inter-test coupling.

## Phase 3 — New Agents (Summary / Tech-Guide / Slack Bot)
- **S1. Requirement coverage** — 5-section summary with visuals; tech-guide
  research→relevance-gate→synopsis→sections, rejects non-technical URLs; Slack
  bot parses intent and dispatches review/summarize/guide; blog PR generalized.
- **S2. Architecture & reuse** — reuses Bedrock factory/prompts/parsers; shared
  Publisher; pluggable web-search; no layering violations / circular imports.
- **S3. Correctness & robustness** — no misrouting; reject-on-bad-input paths;
  Batch parameter contracts correct end-to-end.
- **S4. Tests** — meaningful behavior coverage of every new module.
- **S5. Clean code & safety** — mypy/ruff clean; no secret mishandling.

## Phase 4 — Documentation, Prompts & Diagrams

### D1. tech-doc.md depth & line-by-line accuracy (0–10)
- Documents the **whole** system: every module's responsibility, the end-to-end
  data flow for all three artifact types, config schema, model usage, infra.
- "Line-by-line" means the prose ties explanations to concrete code locations
  (`module.py` / functions / `file:line`), not vague summaries.
- **Must be the single source** for this content (lives only in
  `assets/tech-doc.md`; not duplicated across scattered docs).
- 10 = a new engineer could understand and modify any subsystem from this doc
  alone; 6 = major subsystems glossed; ≤5 = drifts from the actual code.

### D2. tech-doc.md correctness & code-sync (0–10)
- Every claim matches the current code: module names, function names, config
  keys, model IDs, CLI flags, SSM params, IAM scoping. No drift, no invented
  APIs. The judge spot-checks ≥10 claims against `file:line`.

### D3. Prompt & algorithm quality (0–10)
- Review/summary prompts realize the brief: **easy parts kept brief, hard parts
  explained accessibly with background knowledge and visual cues** (figures,
  tables, analogies). No overfitting to a single paper/locale.
- The explanation algorithm (analyze→enrich→synthesize→reflect) is coherent and
  the prompt changes are grounded, not cosmetic.

### D4. Excalidraw diagrams (0–10)
- Hand-drawn style (sketch fonts, rough shapes) per project convention.
- Cover the pipeline/workflows: paper review, paper summary, tech-guide, Slack
  dispatch. Accurate to the code (node names match modules/functions).
- Embedded/linked from `tech-doc.md`.

### D5. draw.io AWS architecture (0–10)
- AWS architecture **only** (not workflows). Correct service icons (Batch, ECS,
  Bedrock, S3, SNS, SSM, KMS, CloudWatch, VPC).
- Accurate to the CDK in `scripts/deploy_infra.py`: split IAM roles, two job
  definitions, encrypted SNS, scoped egress, log retention.
- Embedded/linked from `tech-doc.md`.

### D6. README & docs consistency (0–10)
- README is accurate and current: features, all three artifact types, the Slack
  bot, build/test/deploy instructions. No contradictions with tech-doc.md or
  the code.
