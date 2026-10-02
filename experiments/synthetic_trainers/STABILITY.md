# Optimizer stability and architecture follow-up

This loop starts on 2026-10-02 from the completed nine-run modular campaign.
The local manuscript is *Convexifying Transformers*, arXiv:2211.11052v1,
Section 4. The experiment retains its mod-97 division task and explicitly
different GPTMini architecture. The original calibrations used raw AMSGradW.
On 2026-10-02 the user explicitly selected AdamW as the primary optimizer for
the continuing benchmark; raw AMSGradW remains a paired control. The existing
phase and persistence criteria, six independent confirmations, architecture
comparison and complementary mechanics remain required. Historical AdamW is
less often below target, but still fails the stronger persistence criterion;
its stability remains to be tested. The double-descent
control remains the completed *Deep Double Descent* Appendix C random-feature
archive; rerunning it is not needed to inspect the modular instability.

## Prospective criteria

The existing two-observation onset, legacy memorization diagnostic, epoch-error
shape, and final accuracy remain separate measurements. Fresh follow-up plans
also freeze [PersistenceConfig](persistence.py):

- Memorization requires train accuracy at least 99% and exhaustive held-out
  accuracy at most 10% for at least five consecutive canonical observations
  spanning at least 1,000 updates, before the first held-out 99% crossing.
- Long confirmation requires both train and held-out accuracy at least 99%
  for 20 consecutive canonical observations (4,750 updates at cadence 250).
- Persistent final performance requires both scores at least 99% at every
  canonical observation over the last 50,000 updates, including the final one.
  A 150,000-update run has 201 such observations at cadence 250.
- Stable grokking requires the complete budget/history, the memorization
  plateau, later long confirmation, and persistent final performance.

This criterion concerns scheduled finite observations. It makes no guarantee
between evaluations or beyond the budget. Rapid, persistent generalization
without the low-held-out plateau is reported as such. Tail failures and final
rebounds remain failures of persistence.

Applied retrospectively, no old confirmation run passes this stronger tail
criterion. Reference/AdamW seeds 1/2/3 each have three failing tail observations;
GPTMini/AdamW has 6/3/4; GPTMini/raw AMSGradW has 16/148/13. These new criteria
do not alter the historical frozen targets or their recorded successes.

## Current primary optimizer pair

The user-selected primary optimizer is AdamW. The prospective
[optimizer-pair protocol](protocols/adamw_stability_20261002/optimizer-pair-protocol.md)
and [frozen plan](protocols/adamw_stability_20261002/optimizer-pair-plan.json)
retain learning rate 0.001, model/data seeds 0/0, the 50% split, decay 0.1,
short-final batching, the original model, 150,000 updates each and all criteria
above. AdamW runs first and raw AMSGradW second; optimizer selection is their
only configuration difference. Native AdamW uses betas (0.9, 0.98), bias
correction and no maximum buffer. The pair does not isolate those components.

Both controls now record the already-computed gradient norm on every update;
complete archives must retain all 150,000 trace records, their CSV and the
original canonical/neighbor/full-tensor histories. Diagnostic logging cost is
recorded separately. Fresh paired timing avoids mixing instrumentation versions.
Only a passing **primary AdamW** recipe can enter the unchanged independent
confirmation gate; all six new cases must pass before architecture comparison.
The [complete primary result](protocols/adamw_stability_20261002/adamw-result.md)
has now finished all 150,000 updates and verified its full archive: final
train/held-out is 100%, but seven of 201 final-window observations fail,
with worst held-out 43.9003% at 129,000. The required memorization phase
is absent. Its observed long confirmation at 6,000 remains visible, while
eligible persistent timing has support zero. Finish the running raw control,
retain the full comparison, then execute the separately frozen mod-193 adaptation.
The first complete dense archive also exposed quadratic CSV verification.
Its [runtime execution repair](protocols/adamw_stability_20261002/csv-verification-repair.md)
passes all 235 CPU tests and preserves frozen sources, criteria and complete
comparison semantics. Dense archives currently use its offline verification
command; the additional runtime SHA256 and exact worker sources are retained.
Training continues unchanged, and the queued mod-193 launcher still checks
the original manifests and every frozen fingerprint.

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_optimizer_pair
```

The custom launcher checks the committed manifest, training and analysis sources,
criterion, environment, local manuscripts and its own SHA256. Its `--check-only`
mode verifies without training. The default CLI below describes the completed
historical raw grid and must not reconstruct the new optimizer pair.

## Original frozen raw AMSGradW calibration

Four exploratory runs share data seed 0, initialization seed 0, the exhaustive
4,656/4,656 split, width 128, two layers, four heads, 423,816 parameters,
float32, batch 512, decay 0.1, ten-update warmup, and 150,000 updates.
All use raw AMSGradW betas (0.9, 0.999), epsilon 1e-8, no bias correction,
all-parameter decoupled decay, and no clipping or schedule change.

| Recipe | Learning rate | Batch policy | Changed mechanism |
| --- | ---: | --- | --- |
| `short-lr001` | 0.001 | Original short epoch tail | Unchanged-model control |
| `wrap-lr001` | 0.001 | Full batch across shuffled epochs | Sampling policy only |
| `short-lr0003` | 0.0003 | Original short epoch tail | Learning rate only |
| `short-lr0001` | 0.0001 | Original short epoch tail | Learning rate only |

All four record the same instrumentation: per-tensor gradients, weights, actual
updates, moment norms, learned inverse temperatures, batch/cursor state, and
numeric-answer/EOS losses every 250 updates and at its immediate neighbors.
Canonical exhaustive evaluations remain every 250 updates, plus step zero and
the final step. Neighbor evaluations are separate probes and never certify
targets. The sampling control consumes more examples at fixed updates; actual
exposure is retained. Measurements do not establish why any collapse occurs.

The driver writes an immutable `plan.json` before training and a separate
`state.json` for progress. It freezes recipes, criteria, instrumentation,
portable source hashes, local manuscript hashes, split fingerprints, and the
actual environment. Resume rejects changed sources, settings, or environment.
Completed runs are verified against complete histories and exhaustive scoring;
unsuccessful targets are retained. Only one process can execute a given stage.

```bash
.venv/bin/python -m experiments.synthetic_trainers.stability \
  --output experiments/runs/amsgradw_stability_20261002/calibration
```

Use the same command with `--resume` after a checkpointed interruption.
`--plan-only` freezes without starting training; executing that plan requires
`--resume`. Every scheduled budget completes regardless of target crossings.

After each complete run, archive it separately before the next campaign result
is available. The archive contains its unchanged measurements, all diagnostic
and neighbor logs, canonical/neighbor/diagnostic CSV, an assessment and collapse
neighborhood report, standalone PNG/PDF curves, and artifact checksums. Example:

```bash
.venv/bin/python -m experiments.synthetic_trainers.stability_report \
  experiments/runs/amsgradw_stability_20261002/calibration \
  --run short-lr001 \
  --archive experiments/synthetic_trainers/baselines/amsgradw_stability_short_lr001_seed0_data0_20261002
python3 -m experiments.synthetic_trainers.stability_report \
  experiments/synthetic_trainers/baselines/amsgradw_stability_short_lr001_seed0_data0_20261002 --verify
```

The destination must be fresh. Verification checks the budget, exact scheduled
supports, exhaustive split scoring, declared batch exposure and warmup, metric
and norm consistency, checksums, derived assessment, and CSV rows. It runs
without PyTorch or the original data/checkpoints. A default complete run has
601 canonical observations, 1,200 neighbor probes, and 1,800 diagnostic records.
Checkpoint hashes are recorded, with weights/optimizer kept as local resources.
Single completed calibrations do not complete the four-run stage or certify
independent confirmation. Neighbor categories can overlap and describe observed
associations; they do not establish the cause of a collapse.

Once all four individual archives are complete, assemble their comparison:

```bash
.venv/bin/python -m experiments.synthetic_trainers.stability_comparison \
  experiments/synthetic_trainers/baselines/amsgradw_stability_short_lr001_seed0_data0_20261002 \
  experiments/synthetic_trainers/baselines/amsgradw_stability_wrap_lr001_seed0_data0_20261002 \
  experiments/synthetic_trainers/baselines/amsgradw_stability_short_lr0003_seed0_data0_20261002 \
  experiments/synthetic_trainers/baselines/amsgradw_stability_short_lr0001_seed0_data0_20261002 \
  --archive experiments/synthetic_trainers/baselines/amsgradw_stability_calibration_20261002
```

`stability_comparison <combined-archive> --verify` works without PyTorch or any
original run paths. It requires every frozen recipe exactly once and identical
frozen plans, and retains the individual artifacts without regenerating them.
The combined archive adds complete per-recipe curves, comparison CSV, measured
prose, a separate confirmation-readiness flag, and checksums including the nested
archives' manifests. No mean is taken across distinct mechanisms. Missing target
events stay empty with support zero; early long confirmations remain visible
alongside persistence and phase failures. Calibration does not finish independent
confirmation even if a recipe passes its gate.

## Independent confirmation driver

The [confirmation driver](stability_confirmation.py) requires a complete,
verified calibration comparison and a recipe that passes stable grokking.
Replace `PASSING_RECIPE` below only after the complete calibration comparison.
A negative or incomplete grid cannot launch this stage. The driver retains the
selected training protocol, criterion, instrumentation, sources, environment,
and manuscripts. It freezes model seeds 4/5/6 crossed with data seeds 2/3 before
the first update. Calibration seed 0 and the historical model seeds 1/2/3/data
seed 1 are excluded; new split fingerprints must differ from each other and
from calibration.

```bash
.venv/bin/python -m experiments.synthetic_trainers.stability_confirmation \
  --calibration experiments/synthetic_trainers/baselines/amsgradw_stability_calibration_20261002 \
  --recipe PASSING_RECIPE \
  --output experiments/runs/amsgradw_stability_20261002/confirmation --plan-only
.venv/bin/python -m experiments.synthetic_trainers.stability_confirmation \
  --output experiments/runs/amsgradw_stability_20261002/confirmation --resume
```

The immutable root plan contains all six recipes and both corpus fingerprints;
each case has its own matching immutable plan. The complete calibration evidence
is copied unchanged. Resume checks every source, environment, manuscript, root
and case plan. A serial lock prevents duplicate execution. After a checkpointed
interruption, use the same `--resume` command. Completed cases are verified and
retained without further training. Target failures complete their budgets and do
not skip the remaining repetitions.

Each completed case produces a portable, verified archive with all histories,
neighbor/moment diagnostics, CSV and PNG/PDF curves under `confirmation/archives/`.
Curate and commit each completed result regularly. Once all six are complete:

```bash
.venv/bin/python -m experiments.synthetic_trainers.confirmation_report \
  experiments/runs/amsgradw_stability_20261002/confirmation \
  --archive experiments/synthetic_trainers/baselines/amsgradw_independent_confirmation_20261002
python3 -m experiments.synthetic_trainers.confirmation_report \
  experiments/synthetic_trainers/baselines/amsgradw_independent_confirmation_20261002 --verify
```

Verification needs neither PyTorch nor original runs/checkpoints/manuscripts.
The report includes the unchanged calibration evidence, every repetition,
per-run timing support, descriptive timing means/sample SD, complete curves,
and an all-cases benchmark gate. Crossed seeds/splits are not independent IID
replicates; sample SD is not a confidence interval. All six must pass before
the architecture comparison is ready. If confirmation fails, retain the failures,
continue justified calibration, and use fresh confirmation seeds/splits after
that calibration; do not tune against and reuse the failed confirmation cases.

## Architecture comparison gates

The unchanged-model benchmark must first pass stable grokking in every frozen
independent-confirmation case. Preparing the later protocol does not open that
gate, select an architecture, or authorize a claim that the benchmark is stable.

For a modular architecture candidate, retain complete-budget accuracy, the
first long joint 99% confirmation, the unchanged final-tail assessment, and
the memorization-phase label as separate outcomes. A candidate can reduce the
delay by generalizing rapidly and persistently without a memorization plateau.
Such a result is eligible for the quality/time comparison and is labeled rapid
persistent generalization; it is not labeled stable grokking. Candidate timing
eligibility requires a complete budget/history, a long joint confirmation,
and a passing final tail. First-target timing is not time to permanent convergence.

The comparison plan must freeze its primary quality metric, target, cadence,
full budget, selection rule, and improvement rule before the first comparison
update. Retain observed first-long-confirmation times even when a run later
fails. Eligible comparison timing is null with support zero when the complete
budget, long confirmation, or final-tail gate fails; absent target events also
remain null. Do not form speed ratios for ineligible pairs or silently restrict
a claim to successful pairs. Report every pair's final quality and observed training
and wall times, together with persistence, exposure, parameter count, and peak
memory. Do not average different interventions into a seed statistic.

After exploratory architecture selection, use model seeds and data splits held
out from candidate selection. Treat failed confirmation cases used for further
tuning as calibration and choose fresh confirmation IDs. Freeze all paired controls/candidates and
corpus fingerprints before training, retain the same optimizer and scoring
protocol, and specify how common parameter initialization and execution order
are paired. Any optimizer change is an explicitly separate intervention.
The [portable architecture metrics](architecture_metrics.py) now enforce the
all-case AdamW benchmark gate, unchanged task/data/seeds/optimizer/training and
scoring, and explicit architecture configuration differences. They retain
observed long-confirmation timing while leaving eligible timing and speed ratios
empty for failed pairs. Full quality, exposure, parameter counts, measured costs
and peak memory remain available for every complete pair. Candidate persistence
without a plateau is labeled separately from grokking. This preparation passed
the 230-test full CPU suite and seven final affected checks, including real
complete negative paired archives and a real six-case negative confirmation.
No scientific architecture has been selected or compared. See the
[verification metadata](protocols/adamw_stability_20261002/architecture-metrics-validation.json).

The later complementary-task plan must likewise freeze its metrics and support;
novel-ID and length-transfer outcomes remain separate, and empty novel support
cannot certify transfer. No scientific architecture comparison has started.

## Environment and next decisions

The restored environment initially had Python 3.12.13 and PyTorch 2.7.1+cu118.
Before calibration it was upgraded to PyTorch 2.14.0+cu126, NumPy 2.5.3, and
Matplotlib 3.11.2. CUDA smoke training passed on a GTX 1050 with 2 GiB VRAM.
The official [PyTorch packaging notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
identifies 2.14/CUDA 12.6 as the last published build for Pascal. Install with:

```bash
uv pip install --python .venv/bin/python 'torch==2.14.0' \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python 'numpy==2.5.3' 'matplotlib==3.11.2'
```

The old manuscripts and self-contained archives are available. Historical
model/optimizer checkpoints and Fashion-MNIST IDX files were not transferred.
Fresh modular training rebuilds its complete oracle corpus and uses the tracked,
licensed reference model; these missing historical weights/data do not block
the current calibration. Timing comparisons use paired controls on this GPU.

The four original raw calibrations and the midpoint learning-rate control are
complete and negative; their full archives, curves and diagnoses are preserved.
The current user-selected AdamW/raw AMSGradW pair is frozen. Primary AdamW
completed its full negative phase/persistence result; the raw control is running.
Finish its complete budget, preserve every dense gradient record, verify the
complete pair, and execute the user-directed frozen mod-193 AdamW adaptation.
Any subsequent
fraction, decay, schedule, or model intervention needs a fresh frozen plan; a
negative result does not justify relaxing the success criterion.

Independent confirmation will exclude calibration model/data seed 0: use model
seeds 4/5/6 crossed with data seeds 2/3 in a new frozen plan. Require stable
grokking in all six to declare the benchmark repeatable; report every failure.
The architecture comparison follows after reproducibility. Freeze a targeted
change justified by diagnostics, retain an unchanged-model paired control,
and compare complete-budget quality and measured time to the predefined
target, including support, persistence, parameters, and peak memory.
Extend that comparison to MQAR, copy/parity, lookup/composition, and prefix
counting with independently frozen seeds, pools, targets, and budgets. Report
novel-ID and length transfer separately. Preserve all curves, negative results,
source/data hashes, and full reports. Commit verified logical changes during
the loop and update the handoff with measured outcomes and the next action.
