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
- **C. Latest LLM models** — current Claude models (Opus/Sonnet/Haiku 5.5) wired consistently;
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

## Phase 3 — Artifact Agents and Interfaces
- **S1. Requirement coverage** — review, summary (five sections with visuals)
  and tech guide (research → relevance gate → synopsis → sections → evaluate →
  ground; rejects non-technical URLs); one CLI (`scholar-lens
  review|summarize|guide|status`) that runs locally or submits to Batch; blog
  PR publishing shared by all three.
- **S2. Architecture & reuse** — reuses the Bedrock factory, shared prompt rule
  blocks and parsers; one Publisher; pluggable web search; no layering
  violations or circular imports.
- **S3. Correctness & robustness** — reject-on-bad-input paths; the Batch job
  request contract is the same JSON model the CLI builds.
- **S4. Tests** — meaningful behavior coverage of every new module.
- **S5. Clean code & safety** — mypy/ruff clean; no secret mishandling.

## Phase 4 — Documentation, Prompts & Diagrams

### D1. Design-doc depth (0–10)
- `docs/design.md` documents the whole system: each module's responsibility,
  the data flow for all three artifact types, config schema, model usage,
  infra. Explanations point to modules and functions (symbol anchors, not
  line numbers). README and design doc do not duplicate each other.

### D2. Design-doc correctness (0–10)
- Every claim matches the current code: module and function names, config
  keys, model IDs, CLI flags, SSM params, IAM scoping. The judge spot-checks at
  least ten claims against the code.

### D3. Prompt & algorithm quality (0–10)
- Prompts realize the brief: easy parts brief, hard parts layered (intuition →
  formalism → example → figure), added context grounded and labelled, one
  consistent set of shared rules (depth, visuals, math, style) across review,
  summary and guide. No contradictions, no overfitting to one paper or locale.
- The review workflow (plan → enrich → write ⇄ check → edit → finalize)
  covers every planned must-cover item, revises the best draft against the
  latest feedback rather than rewriting blind, and applies whole-article edits
  only where they match exactly.

### D4. Excalidraw diagrams (0–10)
- Hand-drawn style; cover the paper review, paper summary and tech-guide
  flows; node names match modules/functions; linked from the design doc.

### D5. draw.io AWS architecture (0–10)
- AWS architecture only, with real service icons; accurate to
  `scripts/deploy_infra.py` (Batch job definition, IAM roles, encrypted SNS,
  scoped egress, log retention); linked from the design doc.

### D6. README & docs consistency (0–10)
- README is accurate and current: features, all three artifact types, the CLI
  and the OpenClaw skill, build/test/deploy steps. No contradictions with the
  design doc or the code.
