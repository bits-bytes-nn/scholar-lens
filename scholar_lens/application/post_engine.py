"""The one post engine (docs/engine.md, 4 and 8 E1).

key targets → read → outline (repaired in code, asked again with its
problems, every key target given evidence) → wrong evidence left out → each
section drafted from its evidence and what earlier sections taught, checked
and fixed → the whole-post steps (consistency, repetition, coverage) → the
TL;DR last → assembly. The engine knows no post kind: what differs comes
from the kind's ``PostSpec`` (data) and the small hooks below, which a
kind's engine overrides. The shared loop pieces
(reading agent, claim check, fix) are ``EvidenceWriter``'s. Ports only: the
model, the evidence cache and the checkpoint store are injected.

Checkpoints (docs/engine.md, 4.9; :mod:`.checkpoints`): with a store, the
key targets, the outline (after the wrong evidence, other-SDK evidence and
empty sections are left out), every accepted section (after
``verify_and_fix``), the whole-post result and the TL;DR are saved, each
under a hash of the stage's inputs chained to the output before it, and a
rerun loads every stage whose key exists instead of calling the model. A
loaded section is not checked again. The writer state later stages read is
restored as follows:

- persisted: the notes, blocking problems and claim-check verdicts each
  stage added (:class:`~.checkpoints.WriterState`, replayed in order);
- persisted and re-derived: a section's registered checks (the verifier
  set whole-post fixes run) are rebuilt in code from the outline, the
  ledger and the earlier sections exactly as the writer built them, and
  the source passages they read from the section's saved draft;
- re-derived: the key targets and source names (from the loaded stages).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from scholar_lens.application.checkpoints import (
    CHECKPOINT_VERSION,
    Checkpoint,
    Checkpoints,
    input_hash,
)
from scholar_lens.application.evidence_writer import (
    EDIT_ANSWER,
    EvidenceWriter,
    PostFinding,
    PostReview,
    SectionChecks,
    WriterPrompts,
    all_or_none,
    block,
    cached_evidence,
    cached_notes,
    dump_read,
    evidence_lines,
    with_canonical_evidence,
)
from scholar_lens.application.ports import (
    AgentLimits,
    CheckpointStore,
    LanguageModel,
    ModelRole,
    ProgramRuns,
    RoleSpec,
    SandboxUnavailable,
    TextStore,
)
from scholar_lens.domain.coverage import (
    CoverageTarget,
    ExtractedTargets,
    coverage_problems,
    document_targets,
    extraction_documents,
    extraction_text,
    headline_evidence,
    headline_problems,
    repair_coverage,
    target_lines,
    targets_from,
    unassigned,
    uncited,
    unsatisfiable,
    unused_source_sections,
    without_evidence,
)
from scholar_lens.domain.evidence import (
    Evidence,
    EvidenceLedger,
    SourceDocument,
    same_fact_duplicates,
)
from scholar_lens.domain.evidence_draft import (
    cited_ids,
    copied_sentences,
    free_numbers,
    named_versions,
    prose_length,
    prose_sentences,
    section_heading,
    section_numbers,
    section_problems,
)
from scholar_lens.domain.glossary import glossary_view
from scholar_lens.domain.lengths import char_factor
from scholar_lens.domain.post_context import (
    WrittenSection,
    previous_sections,
    taught_list,
)
from scholar_lens.domain.post_spec import (
    PostOutline,
    PostSection,
    PostSpec,
    depth_targets,
    evidence_policy,
    other_sdk,
    sentence_title_problems,
)
from scholar_lens.domain.program import program_problems, python_code_problems
from scholar_lens.domain.program_run import (
    ProgramRun,
    judge_run,
    program_blocks,
    program_script,
    read_outcomes,
    run_mark,
    run_requirements,
)
from scholar_lens.domain.prompts.rules import (
    FIX_ANSWER,
    TARGETS_SYSTEM,
    consistency_system,
    repetition_system,
)
from scholar_lens.domain.usage_context import usage_stage
from scholar_lens.domain.verifiers import Problem, VerifierSet, as_problems

logger = logging.getLogger(__name__)

StageT = TypeVar("StageT")


@dataclass(frozen=True)
class ReadJob:
    """One reading agent's work: a source (or part of one), the cache key
    of its evidence, the reader's prompt and limits, the evidence kinds
    it is asked again for when it skips them, and the low end of the
    evidence count its prompt asks for (a reader that records fewer is
    asked again for the rest)."""

    document: SourceDocument
    key: str
    prompt: str
    limits: AgentLimits
    required: tuple[str, ...] = ()
    minimum: int = 0


def reader_key(job: ReadJob, prompt: str, reader: RoleSpec | None = None) -> str:
    """The cache key of a job's evidence: the job's key (the source, the
    reader version and model) with a digest of everything else the reader
    is given: its prompt (the topic, the key targets, the counts asked
    for), its limits, the kinds it is asked again for, the count below
    which it is continued, and the reading role's model (its id, effort,
    thinking switch and fallback)."""
    digest = input_hash(
        prompt,
        job.limits.turns,
        job.limits.total_tokens,
        list(job.required),
        job.minimum,
        reader,
    )[:12]
    stem = job.key.removesuffix(".json")
    return f"{stem}-r{digest}.json"


# Makes a read job inside the reading gate (a survey fetches the paper's
# text first); None when there is nothing to read.
ReadStart = Callable[[], Awaitable[ReadJob | None]]


@dataclass(frozen=True)
class PostPrompts:
    """The outline's system prompt and the kind's parts of the whole-post
    reads, which ``prompts.rules`` composes with the shared checks."""

    outline: str
    # What the post is, without an article ("technical guide").
    subject: str
    # The kind's own checks of the consistency and the repetition read.
    consistency: str = ""
    repetition: str = ""
    # The kind's additions to the code checks (read when the spec's
    # whole-post steps trace code).
    code_trace: str = ""
    # Appended to every whole-post read (source rules, security).
    tail: str = ""


@dataclass(frozen=True)
class RunSettings:
    """The run's values the engine reads (from a kind's settings)."""

    language: str
    concurrency: int
    outline_repairs: int
    min_total_chars: int
    max_total_chars: int


@dataclass(frozen=True)
class SectionContext[OutlineT, SectionT]:
    """What one section's prompt, checks and review see."""

    outline: OutlineT
    ledger: EvidenceLedger
    section: SectionT
    own: list[Evidence]
    earlier_ids: list[str]
    earlier: list[Evidence]
    written: Sequence[tuple[SectionT, str]]
    glossary: Mapping[str, str]
    target: int
    reference: str


class SavedSection(BaseModel):
    """An accepted section's checkpoint: the section, and the draft its
    checks were registered with (the source passages they read follow
    from it)."""

    markdown: str
    draft: str


def _fingerprint(document: SourceDocument) -> str:
    return input_hash(document.model_dump(mode="json"))


def _with_title(markdown: str, title: str) -> str:
    """``markdown``'s first ``##`` heading set to ``title`` (a function
    replacement: the title is text, not a template, "$\\mathcal{O}(n)$"
    holds escapes)."""
    return re.sub(
        r"^## .*$", lambda _m: f"## {title}", markdown.lstrip(), count=1, flags=re.M
    )


class PostEngine[OutlineT: PostOutline, SectionT: PostSection, ResultT](EvidenceWriter):
    """The generic flow over a ``PostSpec``; a kind's engine supplies its
    prompts and overrides the hooks it needs."""

    def __init__(
        self,
        model: LanguageModel,
        spec: PostSpec,
        prompts: WriterPrompts,
        post_prompts: PostPrompts,
        run: RunSettings,
        *,
        outline_schema: type[OutlineT],
        store: TextStore | None = None,
        checkpoints: CheckpointStore | None = None,
        max_fix_rounds: int,
        roles: Mapping[ModelRole, RoleSpec] | None = None,
        program_runs: ProgramRuns | None = None,
    ) -> None:
        super().__init__(
            model, prompts, max_fix_rounds=max_fix_rounds, limits=spec.limits
        )
        self.spec = spec
        # Running the post's code in a sandbox (with the spec's code trace;
        # None: the code is read, not run).
        self.program_runs = program_runs
        # Why the stage being computed must not be saved ("" to save it).
        self._unsaved = ""
        # The model each role runs on (id, effort, thinking, fallback): part
        # of the key of every stage the role produces.
        self.roles: Mapping[ModelRole, RoleSpec] = dict(roles or {})
        self.post_prompts = post_prompts
        self.run = run
        self.outline_schema = outline_schema
        self.store = store
        self.checkpoints = (
            Checkpoints(checkpoints, spec.name) if checkpoints is not None else None
        )
        # The output digest of the last stage run or loaded: the next
        # stage's key includes it (a recomputed stage invalidates the rest).
        self._upstream = ""
        # Numbers a sentence may state without evidence.
        self.free = free_numbers()
        # What the post must state (set before reading: the readers record
        # its evidence first).
        self.targets: list[CoverageTarget] = []
        trace = spec.post_passes.code_trace
        # The system prompts of the whole-post reads, by step name.
        self.pass_systems = {
            "consistency": consistency_system(
                post_prompts.subject,
                post_prompts.consistency,
                code_trace=trace,
                trace_extra=post_prompts.code_trace,
                tail=post_prompts.tail,
            ),
            "repetition": repetition_system(
                post_prompts.subject, post_prompts.repetition, tail=post_prompts.tail
            ),
        }

    # ---- checkpoints ------------------------------------------------------------

    async def staged(
        self,
        stage: str,
        inputs: Sequence[object],
        compute: Callable[[], Awaitable[StageT]],
        *,
        dump: Callable[[StageT], Any],
        load: Callable[[Any], StageT],
    ) -> StageT:
        """``compute``'s result, or the one saved for the same inputs.

        Without a checkpoint store the stage simply runs. With one, its key
        hashes the checkpoint version, the kind, ``stage``, the previous
        stage's output and ``inputs``; a saved checkpoint is loaded (and the
        writer-state changes it recorded replayed), else the stage runs and
        is saved with what it added to the writer's state. A checkpoint
        ``load`` cannot use is run again.

        Args:
            stage: The stage name (a key segment).
            inputs: What decides the stage's output besides the previous
                stage: its prompts, system prompts and settings.
            compute: Runs the stage.
            dump: The result as JSON-ready data.
            load: The result from that data (may register derived state);
                raises ``ValueError``/``TypeError``/``KeyError`` when the
                data does not fit.

        Returns:
            The stage's result.
        """
        if self.checkpoints is None:
            return await compute()
        key = self.checkpoints.key(
            stage,
            input_hash(
                CHECKPOINT_VERSION, self.spec.name, stage, self._upstream, *inputs
            ),
        )
        saved = self.checkpoints.load(key)
        if saved is not None:
            try:
                value = load(saved.value)
            except (ValueError, TypeError, KeyError, ValidationError) as e:
                logger.warning(
                    "Checkpoint %s does not fit (%s); running again.", key, e
                )
            else:
                self.replay_state(saved.state)
                self._upstream = input_hash(key, saved.value)
                logger.info("Stage %s loaded from its checkpoint %s.", stage, key)
                return value
        mark = self.state_mark()
        self._unsaved = ""
        value = await compute()
        data = dump(value)
        if self._unsaved:
            # A stage that could not do all it should (a sandbox that did
            # not run) runs again on the next attempt.
            logger.info("Stage %s not saved: %s.", stage, self._unsaved)
            self._upstream = input_hash(key, json.loads(json.dumps(data)))
            return value
        self.checkpoints.save(
            key,
            Checkpoint(
                version=CHECKPOINT_VERSION,
                stage=stage,
                value=data,
                state=self.state_since(mark),
            ),
        )
        # Through JSON, as a load would see it.
        self._upstream = input_hash(key, json.loads(json.dumps(data)))
        return value

    def role_inputs(self, *roles: ModelRole) -> list[object]:
        """The models of ``roles`` as checkpoint inputs (a role without a
        model stays None): a changed model, effort or fallback misses
        the key of every stage that role produces."""
        return [[role, self.roles.get(role)] for role in roles]

    def writing_inputs(self) -> list[object]:
        """What decides every section-level stage besides its prompt: the
        system prompts of drafting, checking, review and fixes (a prompt
        change invalidates), the answer formats, the writing and checking
        models, the loop settings and the kind's settings."""
        p = self.prompts
        return [
            self.role_inputs("write", "evaluate"),
            input_hash(
                p.writer, p.claim_check, p.review, p.fix, EDIT_ANSWER, FIX_ANSWER
            ),
            self.run,
            self.limits,
            self.max_fix_rounds,
            self.free,
            self.checkpoint_inputs(),
        ]

    def checkpoint_inputs(self) -> object:
        """The kind's settings and run inputs its sections' checks and
        prompts read beyond what the prompt shows (JSON-ready; see
        :func:`~.checkpoints.input_hash`)."""
        return {}

    # ---- key targets ------------------------------------------------------------

    async def key_targets(
        self, documents: Sequence[SourceDocument]
    ) -> list[CoverageTarget]:
        """The post's key targets from ``documents`` as the spec's coverage
        names them (docs/engine.md, 4.6): one structured call over the
        documents it extracts from, then one target per document of a
        ``per_document`` authority (code). An extraction that gives no
        valid answer leaves a note, and the post goes on with the targets
        code derives. A checkpointed stage: keyed by the documents, the
        coverage spec, the extraction's prompts and the prompt head."""
        return await self.staged(
            "targets",
            [
                TARGETS_SYSTEM,
                self.role_inputs("evaluate"),
                self.prompt_head(),
                self.spec.coverage,
                self.limits.coverage_source_chars,
                [_fingerprint(d) for d in documents],
            ],
            lambda: self._extract_targets(documents),
            dump=lambda targets: [t.model_dump(mode="json") for t in targets],
            load=lambda data: [CoverageTarget.model_validate(t) for t in data],
        )

    async def _extract_targets(
        self, documents: Sequence[SourceDocument]
    ) -> list[CoverageTarget]:
        spec = self.spec.coverage
        read = extraction_documents(spec, documents)
        targets: list[CoverageTarget] = []
        if read:
            low, high = spec.count
            kinds = "\n".join(
                f"- {k.name}: {k.meaning}"
                + (
                    f" (the first {k.headline} of this kind are the post's headline)"
                    if k.headline
                    else ""
                )
                for k in spec.kinds
            )
            prompt = "\n\n".join(
                [
                    self.prompt_head(),
                    f"List {low}-{high} key targets.",
                    block("kinds", kinds),
                    block(
                        "sources",
                        extraction_text(spec, read, self.limits.coverage_source_chars),
                    ),
                ]
            )
            with usage_stage("coverage"):
                answer = await self._evaluated(
                    "the key-target extraction",
                    role="evaluate",
                    system=TARGETS_SYSTEM,
                    prompt=prompt,
                    schema=ExtractedTargets,
                )
            if answer is not None:
                targets = targets_from(answer, spec, [d.paper_id for d in read])
                if len(targets) < low:
                    logger.info(
                        "The extraction gave %d key targets (asked %d-%d).",
                        len(targets),
                        low,
                        high,
                    )
        targets += document_targets(spec, documents, start=len(targets) + 1)
        if targets:
            logger.info(
                "Key targets: %s",
                "; ".join(f"{t.id} {t.statement[:80]}" for t in targets),
            )
        return targets

    def targets_block(self) -> str:
        """The key targets as a prompt block ("" without targets)."""
        if not self.targets:
            return ""
        return block("key_targets", target_lines(self.targets))

    # ---- read -------------------------------------------------------------------

    def policy_problems(
        self,
        document: SourceDocument,
        kind: str,
        fields: Mapping[str, object],
        corpus: Collection[str] = (),
    ) -> list[str]:
        """The spec's evidence policy for one record."""
        return evidence_policy(
            self.spec.evidence, document, kind, fields, corpus=corpus
        )

    async def read_all(
        self, starts: Iterable[ReadStart], *, corpus: Collection[str] = ()
    ) -> list[tuple[ReadJob | None, list[Evidence]]]:
        """Every job's evidence, in order, at most ``concurrency`` readers at
        once; evidence is cached per :func:`reader_key` (the job's key and
        the reader's whole prompt), and a failure cancels the
        rest. ``corpus``: the authorities of the run's sources (the policy
        reads them)."""
        gate = asyncio.Semaphore(self.run.concurrency)

        def rule(
            document: SourceDocument, kind: str, fields: Mapping[str, str]
        ) -> list[str]:
            return self.policy_problems(document, kind, fields, corpus)

        async def one(start: ReadStart) -> tuple[ReadJob | None, list[Evidence]]:
            async with gate:
                job = await start()
                if job is None:
                    return None, []
                prompt = "\n\n".join(p for p in (job.prompt, self.targets_block()) if p)
                key = reader_key(job, prompt, self.roles.get("read"))
                if self.store is not None and (cached := self.store.get(key)):
                    document = job.document
                    kept = (
                        cached_evidence(
                            cached,
                            document,
                            lambda e: rule(document, e.kind, e.model_dump(mode="json")),
                            key=key,
                        )
                        if self.spec.evidence.recheck_cached
                        else cached_evidence(cached, document, key=key)
                    )
                    if kept is not None:
                        # What the read noted holds for its evidence too.
                        self.notes.extend(cached_notes(cached))
                        return job, kept
                read_notes: list[str] = []
                items = await self.read_document(
                    job.document,
                    prompt=prompt,
                    limits=job.limits,
                    required=job.required,
                    rule=rule,
                    minimum=job.minimum,
                    notes=read_notes,
                )
                if self.store is not None and items:
                    self.store.put(key, dump_read(items, read_notes))
                return job, items

        with usage_stage("read"):
            return await all_or_none(one(s) for s in starts)

    @staticmethod
    def ledger_of(results: Iterable[Sequence[Evidence]]) -> EvidenceLedger:
        """One ledger of the jobs' evidence, numbered in job order."""
        ledger = EvidenceLedger()
        for items in results:
            ledger.extend(items)
        return ledger

    # ---- outline ----------------------------------------------------------------

    async def plan(self, ledger: EvidenceLedger) -> OutlineT:
        """The outline: asked, its duplicate facts merged, repaired in code,
        and asked again with its problems up to ``outline_repairs`` times. A
        key target no evidence of a section is mapped to is a problem too,
        unless no evidence of the ledger can state it (its sources gave
        none); on the last attempt it stays a note rather than a failed
        run. A section title written as a sentence is asked again the same
        way, and noted. When no attempt is free of both, the structurally
        sound outline with the fewest unmapped targets, then the fewest
        sentence titles, is kept (the later one on a tie), whether the last
        attempt was sound or not.

        Raises:
            ValueError: The outline still breaks the spec's rules.
        """
        name = self.spec.outline_name
        duplicates = self.offered_duplicates(ledger)
        if duplicates:
            logger.info("%d evidence items repeat another.", len(duplicates))
        base = "\n\n".join(
            p
            for p in (self.outline_prompt(ledger, duplicates), self.targets_block())
            if p
        )
        prompt = base
        problems: list[str] = []
        best: tuple[OutlineT, list[str], tuple[int, int, int], list[str]] | None = None
        known = {e.id for e in ledger}
        with usage_stage("outline"):
            for attempt in range(self.run.outline_repairs + 1):
                # The schema is a pydantic model the protocol cannot name.
                outline = cast(
                    OutlineT,
                    await self.model.structured(
                        role="outline",
                        system=self.post_prompts.outline,
                        prompt=prompt,
                        schema=cast(type[BaseModel], self.outline_schema),
                    ),
                )
                same = duplicates | same_fact_duplicates(
                    self.same_fact_groups(outline.same_facts, ledger),
                    ledger,
                    self.source_rank(),
                )
                outline = self.canonical_elsewhere(
                    with_canonical_evidence(outline, same), same
                )
                outline, repaired = self.repair(outline, ledger, merged=same)
                if repaired:
                    logger.info(
                        "%s repaired in code: %s",
                        name.capitalize(),
                        "; ".join(repaired),
                    )
                outline = outline.model_copy(
                    update={
                        "coverage": repair_coverage(
                            outline.coverage, self.targets, known, same
                        )
                    }
                )
                problems = self.outline_problems(outline, ledger)
                used = self.used_evidence(outline)
                missing = coverage_problems(
                    self.targets, outline.coverage, used, ledger
                )
                # Asked again, but never what makes an outline unusable, and
                # weighed after coverage: an outline that keeps them is kept
                # with a note.
                titles = sentence_title_problems(s.title for s in outline.sections)
                # The same way: source sections the readers recorded
                # evidence from that the outline leaves out entirely.
                gaps = unused_source_sections(
                    self.covered_documents(), ledger, used, same
                )
                if not problems:
                    # The structurally sound outline that maps the most
                    # targets, then leaves out the fewest source sections,
                    # then has the fewest sentence titles (the later one on a
                    # tie): what the last attempt gives back.
                    rank = (len(missing), len(gaps), len(titles))
                    if best is None or rank <= best[2]:
                        best = (outline, missing, rank, gaps)
                    if not missing and not gaps and not titles:
                        return self._noted_coverage(outline, ledger)
                    if attempt == self.run.outline_repairs:
                        break
                problems += missing + gaps + titles
                logger.warning(
                    "%s attempt %d unusable: %s",
                    name.capitalize(),
                    attempt + 1,
                    problems,
                )
                prompt = (
                    base
                    + "\n\n"
                    + block(
                        "previous_problems",
                        "Fix these in your next outline:\n"
                        + "\n".join(f"- {p}" for p in problems),
                    )
                )
        if best is not None:
            logger.warning(
                "%s: no attempt was free of coverage gaps and sentence titles; "
                "keeping the sound outline that mapped the most, %d key targets "
                "unmapped, %d source sections left out, %d sentence titles.",
                name.capitalize(),
                *best[2],
            )
            if best[3]:
                self.notes.append(
                    "Source sections the outline leaves out (evidence was "
                    "recorded from them): " + " | ".join(g[:160] for g in best[3][:6])
                )
            return self._noted_coverage(best[0], ledger)
        raise ValueError(
            f"The {name} breaks the rules after repairs: " + "; ".join(problems)
        )

    def _noted_coverage(self, outline: OutlineT, ledger: EvidenceLedger) -> OutlineT:
        """``outline``, with a PR note of the key targets no section has
        evidence for (those no evidence of the ledger can state named
        apart) and of its section titles written as a sentence."""
        if sentences := sentence_title_problems(s.title for s in outline.sections):
            self.notes.append(
                "Section titles to reword as noun phrases: " + "; ".join(sentences)
            )
        used = self.used_evidence(outline)
        lacking = unassigned(self.targets, outline.coverage, used)
        hopeless = {t.id for t in unsatisfiable(self.targets, ledger)}

        def listed(targets: list[CoverageTarget]) -> str:
            return " | ".join(
                f"{t.id} [{t.source}] {t.statement[:140]}" for t in targets[:8]
            )

        if mappable := [t for t in lacking if t.id not in hopeless]:
            self.notes.append(
                "Key targets no section of the outline has evidence for: "
                + listed(mappable)
            )
        if unread := [t for t in lacking if t.id in hopeless]:
            self.notes.append(
                "Key targets whose sources gave no evidence (the post cannot "
                "state them): " + listed(unread)
            )
        return outline

    def used_evidence(self, outline: OutlineT) -> set[str]:
        """The evidence the outline uses: its sections' and what the kind
        cites elsewhere."""
        return {
            eid for s in outline.sections for eid in s.evidence
        } | self.cited_elsewhere(outline)

    @staticmethod
    def _without(outline: OutlineT, ids: Collection[str]) -> OutlineT:
        return outline.model_copy(
            update={
                "sections": [
                    s.model_copy(
                        update={"evidence": [e for e in s.evidence if e not in ids]}
                    )
                    for s in outline.sections
                ],
                "coverage": without_evidence(outline.coverage, ids),
            }
        )

    async def drop_wrong(self, outline: OutlineT, ledger: EvidenceLedger) -> OutlineT:
        """The outline without the evidence it uses (sections and what the
        kind cites elsewhere) whose claim misstates its quote."""
        used = self.used_evidence(outline)
        wrong = await self.wrong_evidence([e for e in ledger if e.id in used])
        if not wrong:
            return outline
        return self.drop_elsewhere(self._without(outline, wrong), wrong)

    def keep_code_language(self, outline: OutlineT, ledger: EvidenceLedger) -> OutlineT:
        """Evidence the reader recorded as one other SDK language's (a
        TypeScript-only behaviour in a Python post) leaves the outline:
        stated there it reads as true for the post's language."""
        language = self.code_language(outline)
        if not (self.spec.evidence.same_code_language and language):
            return outline
        other = [
            eid
            for sec in outline.sections
            for eid in sec.evidence
            if (e := ledger.get(eid)) and other_sdk(e.sdk, language)
        ]
        if not other:
            return outline
        logger.info(
            "Evidence for another SDK than %s left out: %s", language, ", ".join(other)
        )
        return self._without(outline, set(other))

    def drop_empty(self, outline: OutlineT) -> OutlineT:
        """Sections the checks emptied, dropped with a note when the spec
        says so (the TL;DR has no evidence of its own)."""
        if self.spec.empty_sections != "drop":
            return outline
        if empty := [
            s.title
            for s in outline.sections
            if not s.evidence and not self.spec.is_tldr(s.kind)
        ]:
            self.notes.append(
                "Sections left without evidence were dropped: " + "; ".join(empty)
            )
            outline = outline.model_copy(
                update={
                    "sections": [
                        s
                        for s in outline.sections
                        if s.evidence or self.spec.is_tldr(s.kind)
                    ]
                }
            )
        return outline

    # ---- the flow ---------------------------------------------------------------

    async def compose(self, ledger: EvidenceLedger) -> ResultT:
        """From the read evidence to the assembled post. The outline, each
        section, the whole-post result and the TL;DR are checkpointed
        stages (:meth:`staged`); the chain starts at the outline, whose key
        holds the ledger and the key targets."""
        self._upstream = ""
        # A version the sources write after a name ("Llama 3.1") may name a
        # model in any sentence.
        if versions := named_versions(
            (f"{e.quote} {e.claim}" for e in ledger if e.kind != "number"),
            (f"{e.quote} {e.claim} {e.cell}" for e in ledger if e.kind == "number"),
        ):
            self.free = self.free.with_names(versions)
        outline = await self.staged(
            "outline",
            [
                input_hash(self.post_prompts.outline, self.prompts.evidence_check),
                self.role_inputs("outline", "evaluate"),
                self.outline_prompt(ledger, self.offered_duplicates(ledger)),
                [t.model_dump(mode="json") for t in self.targets],
                [e.model_dump(mode="json") for e in ledger],
                self.run.outline_repairs,
                self.spec.empty_sections,
                self.spec.evidence.same_code_language,
                [d.paper_id for d in self.covered_documents()],
            ],
            lambda: self.planned(ledger),
            dump=lambda o: cast(BaseModel, o).model_dump(mode="json"),
            load=lambda data: cast(
                OutlineT,
                cast(type[BaseModel], self.outline_schema).model_validate(data),
            ),
        )
        self._names = self.source_names(outline)
        glossary = {t.english: t.rendering for t in outline.glossary}
        targets = self.section_targets(outline)
        written, tldr = await self.write_sections(outline, ledger, glossary, targets)
        written = await self.reviewed(outline, ledger, written)
        opening = (
            await self.staged(
                "tldr",
                [self.spec.tldr, tldr[1], *self.writing_inputs()],
                lambda: self.write_tldr(outline, ledger, *tldr, written),
                dump=lambda md: md,
                load=lambda data: str(data),
            )
            if tldr is not None
            else None
        )
        self.note_total_length([md for _, md in written], opening)
        return await self.assemble(outline, ledger, written, opening)

    def note_total_length(self, sections: Sequence[str], tldr: str | None) -> None:
        """A PR note when the post's prose (its sections and the TL;DR) is
        below the configured minimum total (``run.min_total_chars``, in
        the post's language's characters), naming the shortfall."""
        minimum = round(self.run.min_total_chars * char_factor(self.run.language))
        total = sum(prose_length(md) for md in [*sections, *([tldr] if tldr else [])])
        if total < minimum:
            self.notes.append(
                f"The post has {total:,} characters of prose, {minimum - total:,} "
                f"short of the configured minimum of {minimum:,}: develop its "
                "thinnest sections from their evidence before publishing."
            )

    async def planned(self, ledger: EvidenceLedger) -> OutlineT:
        """The outline as the sections are written from it: planned, without
        wrong evidence, other-SDK evidence and emptied sections."""
        outline = await self.plan(ledger)
        outline = await self.drop_wrong(outline, ledger)
        outline = self.keep_code_language(outline, ledger)
        return self.drop_empty(outline)

    async def reviewed(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        written: list[tuple[SectionT, str]],
    ) -> list[tuple[SectionT, str]]:
        """:meth:`review_post` as a checkpointed stage: the sections' text
        after the whole-post steps."""

        def load(data: Any) -> list[tuple[SectionT, str]]:
            texts = [str(md) for md in data]
            if len(texts) != len(written):
                raise ValueError("the saved post has another number of sections")
            return [(s, md) for (s, _), md in zip(written, texts, strict=True)]

        return await self.staged(
            "review",
            [
                input_hash(*(self.pass_systems[n] for n in sorted(self.pass_systems))),
                self.spec.post_passes,
                self._run_inputs(),
                *self.writing_inputs(),
            ],
            lambda: self.review_post(outline, ledger, written),
            dump=lambda fixed: [md for _, md in fixed],
            load=load,
        )

    @staticmethod
    def own(section: SectionT, ledger: EvidenceLedger) -> list[Evidence]:
        return [e for eid in section.evidence if (e := ledger.get(eid))]

    @staticmethod
    def cited(written: Sequence[tuple[Any, str]]) -> list[str]:
        """The evidence ids the written sections cite, in order, once."""
        return list(
            dict.fromkeys(
                eid
                for _, md in written
                for s in prose_sentences(md)
                for eid in cited_ids(s)
            )
        )

    # ---- sections ---------------------------------------------------------------

    async def write_sections(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        glossary: Mapping[str, str],
        targets: Sequence[int],
    ) -> tuple[list[tuple[SectionT, str]], tuple[SectionT, int] | None]:
        """Every section in order (the TL;DR set aside, written last), and
        the numbers each states tracked by the section that states them
        first."""
        written: list[tuple[SectionT, str]] = []
        numbers: dict[str, str] = {}
        tldr: tuple[SectionT, int] | None = None
        for section, target in zip(outline.sections, targets, strict=True):
            if self.spec.is_tldr(section.kind):
                tldr = (section, target)
                continue
            markdown = await self.write_section(
                outline, ledger, section, written, numbers, glossary, target
            )
            heading = section_heading(markdown) or section.title
            for n in section_numbers(markdown, self.free):
                numbers.setdefault(n, heading)
            written.append((section, markdown))
        return written, tldr

    async def write_section(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        section: SectionT,
        written: list[tuple[SectionT, str]],
        numbers: dict[str, str],
        glossary: Mapping[str, str],
        target: int,
    ) -> str:
        """One section, drafted and checked, or loaded from its checkpoint
        (keyed by its prompt, which holds its evidence, what earlier
        sections taught and the last ones in full; the earlier sections'
        text is in the chain). A loaded section registers the checks the
        writer would have registered, not run again."""
        own = self.own(section, ledger)
        # What earlier sections actually state (their final text), so a
        # back-reference never points at a removed sentence.
        earlier_ids = self.cited(written)
        earlier = [e for eid in earlier_ids if (e := ledger.get(eid))]
        # Evidence a later section already referred back to once; a section
        # may always state its own.
        exhausted = {
            eid
            for s, md in written
            for eid in {i for snt in prose_sentences(md) for i in cited_ids(snt)}
            if eid not in s.evidence
        } - set(section.evidence)
        ctx = SectionContext(
            outline=outline,
            ledger=ledger,
            section=section,
            own=own,
            earlier_ids=earlier_ids,
            earlier=earlier,
            written=written,
            glossary=glossary,
            target=target,
            reference=self.reference_for(own),
        )
        prompt = self.section_prompt(ctx)
        verifiers = self.section_verifiers(ctx, numbers, exhausted)
        judgement = section.kind in self.spec.judgement_kinds

        async def compute() -> SavedSection:
            draft = await self.draft(
                prompt, own, target, retry_hint=self.spec.retry_hint
            )
            self.reference = self.checking_reference(ctx.reference, draft)
            markdown = await self.verify_and_fix(
                draft,
                title=section.title,
                review_context=self.review_context(ctx),
                verifiers=verifiers,
                ledger=ledger,
                own=own,
                earlier=earlier,
                free=self.free,
                section=section,
                judgement=judgement,
                rest="\n\n".join(md for _, md in written),
            )
            self.reference = ""
            return SavedSection(
                markdown=self.finish_section(section, markdown), draft=draft
            )

        def load(data: Any) -> SavedSection:
            saved = SavedSection.model_validate(data)
            # What verify_and_fix registers: every later change of the
            # section (a whole-post fix) runs these.
            self.checked[id(section)] = SectionChecks(
                verifiers,
                tuple(own),
                tuple(earlier),
                self.checking_reference(ctx.reference, saved.draft),
                judgement,
                self.free,
            )
            return saved

        saved = await self.staged(
            "section",
            [prompt, self.review_context(ctx), *self.writing_inputs()],
            compute,
            dump=lambda saved: saved.model_dump(mode="json"),
            load=load,
        )
        return saved.markdown

    def written_sections(
        self, written: Sequence[tuple[SectionT, str]]
    ) -> list[WrittenSection]:
        """The written sections as the earlier-section context reads them."""
        return [
            WrittenSection(s.title, md, s.kind in self.spec.runnable_kinds)
            for s, md in written
        ]

    def section_prompt(self, ctx: SectionContext[OutlineT, SectionT]) -> str:
        """The section writer's prompt: the shared skeleton with the kind's
        blocks. What earlier sections said comes as the list of what each
        taught and the last ``limits.previous_sections`` in full
        (docs/engine.md, 4.8)."""
        section, outline = ctx.section, ctx.outline
        done = self.written_sections(ctx.written)
        sections_map = "\n".join(
            f"- {'(this) ' if s is section else ''}{s.kind}: {s.title} — {s.thesis}"
            + (f" [teaches: {', '.join(t)}]" if (t := self.taught(s)) else "")
            for s in outline.sections
        )
        return "\n\n".join(
            [
                self.prompt_head(),
                block("outline", sections_map),
                block(
                    "section",
                    f"kind: {section.kind}\ntitle: {section.title}\nthesis: {section.thesis}\n"
                    f"question: {section.question}\ndepth: {section.depth}\n"
                    f"length: about {ctx.target:,} characters of prose\n"
                    f"running example: {outline.running_example or 'none'}"
                    + self.section_lines(outline),
                ),
                block("evidence", evidence_lines(ctx.own)),
                block(
                    "already_taught",
                    taught_list(
                        done, ctx.ledger, claim_chars=self.limits.taught_claim_chars
                    ),
                ),
                *self.naming_blocks(ctx),
                block(
                    "glossary",
                    glossary_view(dict(ctx.glossary), [m for _, m in ctx.written]),
                ),
                block(
                    "previous_sections",
                    previous_sections(done, self.limits.previous_sections),
                ),
                *self.closing_blocks(ctx),
            ]
        )

    def section_verifiers(
        self,
        ctx: SectionContext[OutlineT, SectionT],
        numbers: Mapping[str, str],
        exhausted: Collection[str],
    ) -> VerifierSet:
        """The verifier set of a section, built once: the shared code checks,
        then the kind's (``extra_problems``). What it reads of the post is
        fixed here (the numbers earlier sections state), so a later run of
        the set, after the whole post is written, judges the section as its
        own fixes were judged."""
        section = ctx.section
        figures = [e.figure_url for e in ctx.own if e.kind == "figure" and e.figure_url]
        depth = section.depth if section.kind in self.spec.depth_kinds else ""
        earlier_numbers = dict(numbers)
        spent = frozenset(exhausted)
        papers = tuple(self.named_papers(section))

        def shared(markdown: str) -> list[Problem]:
            return section_problems(
                markdown,
                title=section.title,
                own=section.evidence,
                earlier=ctx.earlier_ids,
                ledger=ctx.ledger,
                earlier_numbers=earlier_numbers,
                figure_urls=figures,
                free_numbers=self.free,
                passages=self.passages,
                exhausted=spent,
                papers=papers,
                target_chars=ctx.target,
                depth=depth,
                limits=self.limits,
            )

        def kind(markdown: str) -> list[Problem]:
            return as_problems(section.kind, self.extra_problems(ctx, markdown))

        return VerifierSet((shared, kind), frozenset(figures))

    # ---- whole post -------------------------------------------------------------

    async def review_post(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        written: list[tuple[SectionT, str]],
    ) -> list[tuple[SectionT, str]]:
        """The whole-post steps the spec lists, in order (docs/engine.md,
        4.7): the consistency and repetition reads of the post (run
        together; with the spec's code trace, code's program check of the
        runnable sections joins them), each section they name fixed as an
        edit transaction, and the coverage check of the fixed post. A read
        that gives no valid answer blocks the post."""
        steps = self.spec.post_passes
        if not written:
            return written
        reads = list(self.pass_systems)
        with usage_stage("review"):
            answers = await asyncio.gather(
                *(
                    self._evaluated(
                        f"the whole-post {name} review",
                        blocking=True,
                        unchecked="the post is unreviewed",
                        role="evaluate",
                        system=self.pass_systems[name],
                        prompt=self.pass_prompt(name, outline, ledger, written),
                        schema=PostReview,
                    )
                    for name in reads
                )
            )
        reviews: list[tuple[str, PostReview]] = [
            (n, r) for n, r in zip(reads, answers, strict=True) if r is not None
        ]
        for name, review in reviews:
            if review.trace:
                logger.info("Post review %s trace: %s", name, review.trace[:600])
        program = steps.code_trace
        found = self.program_findings(written) if program else []
        # The code is run only when it reads clean: a block the static check
        # flags fails the run the same way.
        ran = await self.run_program(written) if program and not found else None
        found += [
            PostFinding(
                section=f.section,
                problem=f.problem,
                fix="Make the block run as pasted after the earlier blocks.",
            )
            for f in (ran.findings if ran else [])
        ]
        issues = [i for _, r in reviews for i in r.issues] + found
        fixed = written
        if issues:
            logger.info(
                "Post review: %d issue(s) (%s), %d from the code check.",
                len(issues),
                ", ".join(f"{n}: {len(r.issues)}" for n, r in reviews),
                len(found),
            )
            fixed = await self.apply_findings(
                written,
                issues,
                ledger,
                title_of=lambda s: s.title,
                post_check=self.program_check if self.spec.runnable_kinds else None,
            )
        # The fixes are checked again, not trusted: what is left goes on the
        # PR checklist.
        if program and (left := self.program_problems(fixed)):
            self.notes.append(
                "Code that may not run as pasted: "
                + " | ".join(f"{t}: {p}"[:200] for t, p in left[:5])
            )
        elif program and found and ran is None:
            # The static problems are fixed: the fixed code runs now.
            if (ran := await self.run_program(fixed)) is not None:
                self.program_run_notes(ran)
        elif ran is not None:
            # The fixes may have changed the code (any finding's, not only
            # the run's): what is noted is the last code's run.
            if self._program_code(fixed) != self._program_code(written):
                ran = await self.run_program(fixed)
            if ran is not None:
                self.program_run_notes(ran)
        self.coverage_notes(outline, fixed)
        return fixed

    def pass_prompt(
        self,
        name: str,
        outline: OutlineT,
        ledger: EvidenceLedger,
        written: Sequence[tuple[SectionT, str]],
    ) -> str:
        """The prompt of the whole-post read ``name``: the post in order;
        the consistency read also gets the running example and, with the
        spec's code trace, the code language and the kind's code sources."""
        post = "\n\n".join(md for _, md in written)
        if name != "consistency":
            return "\n\n".join([self.prompt_head(), block("post", post)])
        head = (
            f"{self.prompt_head()}\nRunning example: "
            f"{outline.running_example or 'none'}."
        )
        if not self.spec.post_passes.code_trace:
            return "\n\n".join([head, block("post", post)])
        head += f" Code language: {self.code_language(outline) or 'any'}."
        return "\n\n".join(
            [head, block("post", post), *self.code_reference(ledger, post)]
        )

    def program_problems(
        self, written: Sequence[tuple[SectionT, str]]
    ) -> list[tuple[str, str]]:
        """(section heading, problem) of the code a reader pastes in order:
        a block that does not parse, a name nothing defines (the runnable
        sections only; every section's definitions count)."""
        return python_code_problems(self._program_sections(written))

    def _program_sections(
        self, written: Sequence[tuple[SectionT, str]]
    ) -> list[tuple[str, str, bool]]:
        """(heading, markdown, runnable) of the written sections, in order."""
        return [
            (section_heading(md) or s.title, md, s.kind in self.spec.runnable_kinds)
            for s, md in written
        ]

    def program_check(self, written: Sequence[tuple[SectionT, str]]) -> list[Problem]:
        """The program check of :meth:`program_problems` as problems (a
        whole-post fix's transaction refuses a fix that adds one)."""
        return program_problems(self._program_sections(written))

    def _run_inputs(self) -> object:
        """What of the sandbox run changes the reviewed post (a checkpoint
        input)."""
        runs = self.program_runs
        if runs is None:
            return None
        return [
            runs.runner.name,
            sorted(runs.pins),
            sorted(runs.modules),
            runs.block_timeout_s,
            runs.total_timeout_s,
        ]

    async def run_program(
        self, written: Sequence[tuple[SectionT, str]]
    ) -> ProgramRun | None:
        """The post's code run in the sandbox in reading order (None without
        a sandbox, without a runnable section's block, or when the sandbox
        could not run it: a note says why)."""
        runs = self.program_runs
        if runs is None:
            return None
        blocks = program_blocks(self._program_sections(written))
        if not any(checked for _, _, checked in blocks):
            return None
        mark = run_mark(uuid.uuid4().hex)
        script = program_script(
            [code for _, code, _ in blocks],
            block_timeout_s=runs.block_timeout_s,
            mark=mark,
        )
        requirements = run_requirements(runs.pins, "\n\n".join(md for _, md in written))
        try:
            result = await runs.runner.run(
                script, requirements=requirements, timeout_s=runs.total_timeout_s
            )
        except SandboxUnavailable as e:
            logger.warning("The code sandbox is unavailable: %s", e)
            self._unsaved = "the code sandbox was unavailable"
            self._note_once(f"The code examples were not run: {e}")
            return None
        if result.failure:
            logger.warning("The code sandbox did not run the code: %s", result.failure)
            self._unsaved = "the code sandbox did not run the code"
            self._note_once(f"The code examples were not run: {result.failure}")
            return None
        if result.warning:
            logger.warning("Code sandbox: %s", result.warning)
            self._note_once(f"Code sandbox: {result.warning}")
        ran = judge_run(
            blocks,
            read_outcomes(result.stdout, mark),
            runs.modules,
            exact=result.exact,
        )
        logger.info(
            "Code run in the sandbox: %d of %d block(s), %d failure(s)%s.",
            ran.blocks_run,
            ran.blocks_total,
            len(ran.findings),
            f"; limited: {ran.limited}" if ran.limited else "",
        )
        return ran

    def _program_code(self, written: Sequence[tuple[SectionT, str]]) -> list[str]:
        return [code for _, code, _ in program_blocks(self._program_sections(written))]

    def _note_once(self, note: str) -> None:
        if (note := note[:300]) not in self.notes:
            self.notes.append(note)

    def program_run_notes(self, ran: ProgramRun) -> None:
        """PR notes of the last sandbox run: a block that still fails, or
        what limited the check."""
        if ran.findings:
            self.notes.append(
                "Code that fails when run in the sandbox: "
                + " | ".join(f"{f.section}: {f.problem}"[:240] for f in ran.findings)
            )
        elif ran.limited:
            self.notes.append(
                f"The sandbox check of the code examples was partial: {ran.limited}"[
                    :400
                ]
            )

    def program_findings(
        self, written: Sequence[tuple[SectionT, str]]
    ) -> list[PostFinding]:
        """:meth:`program_problems` as whole-post findings."""
        return [
            PostFinding(
                section=title,
                problem=problem,
                fix="Make the code run as pasted after the earlier blocks.",
            )
            for title, problem in self.program_problems(written)
        ]

    def coverage_notes(
        self, outline: OutlineT, written: Sequence[tuple[SectionT, str]]
    ) -> None:
        """A PR note for each key target none of whose evidence the body
        cites (a reader decides; it does not block the post)."""
        if missing := uncited(self.targets, outline.coverage, self.cited(written)):
            self.notes.append(
                "Key targets the body never states (no evidence mapped to them "
                "is cited): "
                + " | ".join(
                    f"{t.id} [{t.source}] {t.statement[:140]}" for t in missing[:8]
                )
            )

    async def write_tldr(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        section: SectionT,
        target: int,
        written: Sequence[tuple[SectionT, str]],
    ) -> str:
        """The TL;DR from its evidence, the evidence the sections cite and
        their headings and opening sentences: it may state what the body
        states (the one place a fact is said twice)."""
        tldr = self.spec.tldr
        assert tldr is not None
        # The headline targets' evidence comes first: the TL;DR must state
        # them (docs/engine.md, 4.6), and the prompt shows a bounded list.
        flagged = [
            eid
            for t in self.targets
            if t.headline
            for eid in outline.coverage.get(t.id, ())
        ]
        pool = [
            e
            for eid in dict.fromkeys(
                [*section.evidence, *flagged, *self.cited(written)]
            )
            if (e := ledger.get(eid)) is not None
        ]
        # Figures stay in their sections.
        context = [e for e in pool if e.kind != "figure"]
        shown = (pool if tldr.figures_in_prompt else context)[
            : self.limits.tldr_evidence
        ]
        headline = headline_evidence(
            self.targets,
            outline.coverage,
            {e.id for e in shown if e.kind != "figure"},
        )
        openings = "\n".join(
            f"- {section_heading(md) or s.title}: "
            + " ".join(prose_sentences(md)[:2])[:400]
            for s, md in written
        )
        prompt = self.tldr_prompt(
            outline, section, target, openings, evidence_lines(shown)
        )
        if headline:
            prompt += "\n\n" + block(
                "headline_targets",
                "State each of these, citing its evidence:\n"
                + "\n".join(
                    f"- {t.id}: {t.statement} (evidence: {', '.join(ids)})"
                    for t, ids in headline
                ),
            )
        draft = await self.draft(
            prompt,
            shown,
            target,
            retry_hint=tldr.retry_hint,
            evidence=shown,
        )
        body = [s for _, md in written for s in prose_sentences(md)]

        def shared(md: str) -> list[Problem]:
            return section_problems(
                md,
                title=tldr.title,
                own=[e.id for e in context],
                earlier=(),
                ledger=ledger,
                earlier_numbers={},
                figure_urls=(),
                free_numbers=self.free,
                passages=self.passages,
                target_chars=target if tldr.checks_length else 0,
                limits=self.limits,
                minimum_length=False,
                subsections=False,
            )

        def kind(md: str) -> list[Problem]:
            problems = as_problems(tldr.kind, self.tldr_problems(md))
            if tldr.no_body_copies:
                problems += copied_sentences(md, body)
            cited_here = {i for snt in prose_sentences(md) for i in cited_ids(snt)}
            problems += as_problems(
                "coverage_headline", headline_problems(headline, cited_here)
            )
            return problems

        markdown = await self.verify_and_fix(
            draft,
            title=tldr.title,
            review_context=tldr.review_context,
            verifiers=VerifierSet((shared, kind), frozenset()),
            ledger=ledger,
            own=context,
            earlier=(),
            free=self.free,
            judgement=tldr.kind in self.spec.judgement_kinds,
            rest="\n\n".join(md for _, md in written),
            subsections=False,
        )
        markdown = _with_title(markdown, tldr.title)
        if not markdown.startswith("## "):
            markdown = f"## {tldr.title}\n\n{markdown}"
        return markdown

    # ---- hooks: outline ---------------------------------------------------------

    def outline_prompt(
        self, ledger: EvidenceLedger, duplicates: Mapping[str, str]
    ) -> str:
        """The outline writer's prompt (``duplicates``: evidence not offered)."""
        raise NotImplementedError

    def offered_duplicates(self, ledger: EvidenceLedger) -> dict[str, str]:
        """Evidence that repeats an earlier item's fact, not offered to the
        outline: duplicate id -> the id kept."""
        return {}

    def source_rank(self) -> Mapping[str, int] | None:
        """Source id -> rank (lower keeps a fact the outline marks as one)."""
        return None

    def same_fact_groups(
        self, groups: Sequence[Sequence[str]], ledger: EvidenceLedger
    ) -> list[list[str]]:
        """The outline's same-fact groups as this kind merges them (as given
        by default: a guide merges what several pages state)."""
        return [list(g) for g in groups]

    def repair(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        merged: Mapping[str, str] | None = None,
    ) -> tuple[OutlineT, list[str]]:
        """What code fixes in an outline, and what it changed. ``merged``:
        evidence merged as the same fact -> the item kept."""
        raise NotImplementedError

    def outline_problems(self, outline: OutlineT, ledger: EvidenceLedger) -> list[str]:
        """What makes an outline unusable (the outline writer is told)."""
        raise NotImplementedError

    def covered_documents(self) -> Sequence[SourceDocument]:
        """The sources whose every section with recorded evidence the
        outline should draw on (none by default: a guide or survey picks
        from many sources; a paper post covers its paper)."""
        return ()

    def cited_elsewhere(self, outline: OutlineT) -> set[str]:
        """Evidence the outline cites outside its sections (a table)."""
        return set()

    def canonical_elsewhere(
        self, outline: OutlineT, duplicates: Mapping[str, str]
    ) -> OutlineT:
        """``cited_elsewhere`` with each duplicate replaced by its kept id."""
        return outline

    def drop_elsewhere(self, outline: OutlineT, wrong: Collection[str]) -> OutlineT:
        """``cited_elsewhere`` without the ``wrong`` evidence."""
        return outline

    def code_language(self, outline: OutlineT) -> str:
        """The SDK language the post's code is in ("" when none)."""
        return ""

    def source_names(self, outline: OutlineT) -> dict[str, str]:
        """Source id -> the name the claim check shows for it."""
        return dict(self._names)

    def section_targets(self, outline: OutlineT) -> list[int]:
        """Prose characters per section."""
        return depth_targets(
            outline.sections,
            min_total=self.run.min_total_chars,
            max_total=self.run.max_total_chars,
            language=self.run.language,
        )

    # ---- hooks: sections --------------------------------------------------------

    def reference_for(self, own: Sequence[Evidence]) -> str:
        """The source passages a section's writer, checks and fixes read
        besides its evidence ("" when none)."""
        return ""

    def checking_reference(self, reference: str, draft: str) -> str:
        """The passages the checks and fixes of ``draft`` read."""
        return reference

    def prompt_head(self) -> str:
        """The first line of a section prompt (language and subject)."""
        raise NotImplementedError

    def taught(self, section: SectionT) -> Sequence[str]:
        """What a section teaches (shown in the outline map)."""
        return ()

    def section_lines(self, outline: OutlineT) -> str:
        """Lines appended to the prompt's section block."""
        return ""

    def naming_blocks(self, ctx: SectionContext[OutlineT, SectionT]) -> list[str]:
        """Prompt blocks that name the sources, after the evidence."""
        return []

    def closing_blocks(self, ctx: SectionContext[OutlineT, SectionT]) -> list[str]:
        """The prompt's last blocks."""
        return [block("source_sections", ctx.reference)]

    def review_context(self, ctx: SectionContext[OutlineT, SectionT]) -> str:
        """What the section reviewer is told about the section."""
        s = ctx.section
        return f"Section kind: {s.kind}; thesis: {s.thesis}; depth: {s.depth}."

    def named_papers(self, section: SectionT) -> Sequence[str]:
        """Papers a section must cite."""
        return ()

    def extra_problems(
        self, ctx: SectionContext[OutlineT, SectionT], markdown: str
    ) -> Sequence[str | Problem]:
        """The kind's own problems of a section draft (a message is a problem
        of the section's kind that quotes no passage)."""
        return []

    def finish_section(self, section: SectionT, markdown: str) -> str:
        """A written section with its heading: the outline title when the
        draft has none, and code's title for a kind the spec fixes."""
        if not markdown.lstrip().startswith("## "):
            return f"## {section.title}\n\n{markdown.lstrip()}"
        if section.kind in self.spec.fixed_titles:
            return _with_title(markdown, section.title)
        return markdown

    # ---- hooks: whole post and assembly -----------------------------------------

    def code_reference(self, ledger: EvidenceLedger, post: str) -> list[str]:
        """Prompt blocks the consistency read checks the post's code against
        (the sources' code, the product's definitions), with the spec's code
        trace."""
        return []

    def tldr_prompt(
        self,
        outline: OutlineT,
        section: SectionT,
        target: int,
        openings: str,
        evidence: str,
    ) -> str:
        """The TL;DR writer's prompt."""
        raise NotImplementedError

    def tldr_problems(self, markdown: str) -> Sequence[str | Problem]:
        """The kind's own problems of a TL;DR draft (a message is a problem
        of the TL;DR's kind that quotes no passage)."""
        return []

    async def assemble(
        self,
        outline: OutlineT,
        ledger: EvidenceLedger,
        written: list[tuple[SectionT, str]],
        tldr: str | None,
    ) -> ResultT:
        """The kind's result from the written sections and the TL;DR."""
        raise NotImplementedError
