# Evaluation

How post quality is measured. The protocol and the adoption rule are in [ADR 0002](../docs/adr/0002-evaluation-protocol.md); the judging steps in [JUDGE.md](./JUDGE.md); the rubric in [ARTIFACT_RUBRIC.md](./ARTIFACT_RUBRIC.md); every measurement, in time order, in [RESULTS.md](./RESULTS.md) (Korean).

## Layout

| Path | What it holds |
| --- | --- |
| `sets/baseline.yaml` | The frozen inputs. `split: dev` items are the ones prompts and checks were tuned on; `split: confirm` items decide a default change (today three reviews only, so a default change for summaries, guides or surveys needs confirm items first) |
| `sets/human_fixes.yaml` | Defects humans fixed or flagged in published posts: a check each regenerated post must pass |
| `sets/claim_truth.yaml` | Hand-fixed sentences with verbatim anchors, the ground truth of the claim-check replay |
| `sets/plans/` | Fixed survey plans, so surveys are compared on the same papers |
| `baselines/<date>/` | Judge results and metrics of a baseline run (generated posts and run manifests are not tracked) |
| `runs/<name>/` | Judge results, blind mappings and replays of one comparison; `RESULTS.md` names the folder of each section |

## Commands

```bash
# Deterministic metrics of finished posts (free)
poetry run python -m eval.metrics KIND FILE [FILE ...]
# Unblind a pairwise judging run and count it (free)
poetry run python -m eval.compare eval/runs/<name> [--set eval/sets/baseline.yaml]
# Claim-check replay against the ground truth: `run` calls Bedrock (about $1.5-3 for the set)
poetry run python -m eval.claim_replay run WORK_DIR [--only NAME ...] [--out FILE]
poetry run python -m eval.claim_replay score WORK_DIR/replay.json
poetry run python -m eval.claim_replay anchors WORK_DIR
```

Generating posts for a comparison uses the normal CLI with `SCHOLAR_LENS_CONFIG` pointing at the run's settings, and the judge follows `JUDGE.md`. Paid runs need the owner's approval first.
