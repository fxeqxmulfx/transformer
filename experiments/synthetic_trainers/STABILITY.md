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
confirmation gate; all six new cases must pass before claiming a repeatable
architecture improvement. The later user-directed sparsemax exploration is
recorded below and does not open that gate.
The [complete primary result](protocols/adamw_stability_20261002/adamw-result.md)
has now finished all 150,000 updates and verified its full archive: final
train/held-out is 100%, but seven of 201 final-window observations fail,
with worst held-out 43.9003% at 129,000. The required memorization phase
is absent. Its observed long confirmation at 6,000 remains visible, while
eligible persistent timing has support zero. The
[complete raw control](protocols/adamw_stability_20261002/raw-amsgradw-result.md)
also fails the required phase and persistence, with ten failing tail observations
and worst held-out 58.1400%. The
[verified complete pair](protocols/adamw_stability_20261002/optimizer-pair-result.md)
retains all 300,000 updates; neither recipe opens independent confirmation.
The user-directed frozen mod-193 AdamW adaptation is now complete.
Its [first target at 40,000](protocols/adamw_stability_20261002/first-target-mod193.md)
has no required pre-target memorization plateau, ruling out stable-grokking
eligibility. Its [first final-window failure at 102,750](protocols/adamw_stability_20261002/first-tail-failure-mod193.md)
also rules out persistence: train/held-out 95.1749%/94.6082%, with failing
immediate neighbors and EOS 100%. The
[complete result](protocols/adamw_stability_20261002/larger-modulus-result.md)
retains all 150,000 updates and seven final-window failures, with worst held-out
79.8953% and final 100%. Neither gate passes; eligible timing support is zero.
Slower generalization alone does not satisfy either gate.
The first complete dense archive also exposed quadratic CSV verification.
Its [runtime execution repair](protocols/adamw_stability_20261002/csv-verification-repair.md)
passes all 235 CPU tests and preserves frozen sources, criteria and complete
comparison semantics. Dense archives currently use its offline verification
command; the additional runtime SHA256 and exact worker sources are retained.
Both frozen campaigns and their archive workers have now completed, with
the original manifests and every frozen fingerprint checked. The runtime
adapter's identical optimization is now applied to the
[ordinary verifier](protocols/adamw_stability_20261002/production-csv-repair.md).
All 235 CPU tests pass without skips, and three full dense archives / both
comparisons return identical results through both paths. New plans must record
the new analysis hash; historical strict launchers pin their old source version.

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
The user has subsequently selected sparsemax for the next exploratory pair;
no scientific architecture comparison has started. See the
[verification metadata](protocols/adamw_stability_20261002/architecture-metrics-validation.json).

The later complementary-task plan must likewise freeze its metrics and support;
novel-ID and length-transfer outcomes remain separate, and empty novel support
cannot certify transfer. No scientific architecture comparison has started.
The [existing-trainer audit](protocols/adamw_stability_20261002/complementary-preparation.md)
now verifies seven real CPU AdamW implementation paths and separate final/
selected metrics. General AdamW currently decays matrices only and has no
warmup; these differ from the modular primary. Freeze and document any task-
specific protocol adaptation or verify matching options before the later
scientific comparison. Its two-update probes provide implementation evidence
only, and do not open the modular independent-confirmation or architecture gate.

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
The user-selected AdamW/raw AMSGradW pair completed both full negative
phase/persistence results. All dense gradient records and the complete pair
are verified and preserved. The user-directed frozen mod-193 AdamW adaptation
also completed its full negative phase/persistence result. The later 25%
fraction control and lower-rate follow-up are recorded below. The ordinary
CSV-verifier repair is complete.
The [25% scientific plan](protocols/adamw_stability_20261002/fraction25-protocol.md)
has now been frozen after the complete parent result, changing only the training
fraction and recording the two analysis-only repairs. Its gates remain unchanged.
Its full budget and archives have completed. The
[verified positive phase prefix](protocols/adamw_stability_20261002/first-long-confirmation-fraction25.md)
preserves all 100 canonical and 24,750 gradient observations: a qualifying
plateau at 4,500–11,500, first held-out target at 20,000 and 20 consecutive
joint target observations through 24,750. This observed delayed generalization
does not yet establish final-tail persistence or independent repeatability.
The [first quarter-split final-window failure](protocols/adamw_stability_20261002/first-tail-failure-fraction25.md)
is now verified at 115,000: train / held-out 92.7461%/91.6271%, with EOS 100%
and failures at both immediate neighbors. This refutes the frozen persistence
gate despite the observed delayed generalization. The
[complete quarter-split archive](protocols/adamw_stability_20261002/fraction25-result.md)
now verifies all 150,000 updates and 601 canonical observations, with final
100%/100% and three failed final-window points. The minimum final-window canonical held-out
accuracy is 48.9206% at 139,000. Both processes finish with code 0. Six-case
confirmation and architecture selection remain closed; the prepared single-
field lower-rate control is the next calibration, with unchanged criteria.
The [lower-rate plan](protocols/adamw_stability_20261002/lower-rate-protocol.md)
is now frozen before scientific training: only learning rate 0.0003 differs;
the real launcher and four negative manifest checks pass. The same full
budget, source/environment/corpus fingerprints and every gate remain fixed.
After manifest commit `ed83c85`, trainer PID 126400/session 93469 and NoTorch
archive worker PID 126478/session 19327 launched; their actual launch is
[verified](protocols/adamw_stability_20261002/lower-rate-launch-validation.json).
At update 13,250, the incomplete history has train/held-out 99.9028%/0.8096%
and a qualifying plateau from 4,500–13,250, but no held-out target or long
confirmation. Finish the full budget and keep all pinned sources immutable;
this partial observation does not open independent confirmation.
The later [lower-rate phase prefix](protocols/adamw_stability_20261002/first-long-confirmation-lower-rate.md)
now verifies a qualifying plateau at 4,500–27,250, first held-out 99% at
95,750 and twenty consecutive joint targets through 100,500. It retains all
403 canonical / 100,500 gradient observations, with PNG/PDF figures reviewed.
The three prefix tail points pass, but later held-out 97.4813% at 105,250
already refutes persistence, with train/EOS 100%. The whole budget is now
complete; this valid phase and negative persistence do not open confirmation.
The [complete lower-rate result](protocols/adamw_stability_20261002/lower-rate-result.md)
preserves all 150,000 updates, eight failed tail observations and final 100%/100%.
Both scientific processes are terminal; all full archives verify without PyTorch,
and their PNG and actual PDF figures were reviewed. The additional
[frequency/recovery metrics](protocols/adamw_stability_20261002/recovery-metrics.md)
count three episodes and a final 38,000-update sampled joint-target span.
Fixed 10,000-update tail windows show onset counts 2/1/0/0/0 and failed fractions
7.5%/12.5%/0%/0%/0%, retaining exact supports and the original strict gate.
The user requested a maximum of 300,000 training updates per run. A new frozen
budget extension must preserve this original negative result, restore the full
native optimizer/sampling checkpoint, change only steps and score the fixed
250,000–300,000 final window. Pin the new metric hash and width before training.
The [prospective continuation](protocols/adamw_stability_20261002/budget300k-protocol.md)
is now frozen with those settings, only total steps changed, the complete
native checkpoint preserved, 39 source fingerprints pinned and actual launcher
validation passed. Eight isolated negative fixtures are rejected. Additional
whole-history recovery windows retain the interval before the new final tail.
After manifest commit `3cf14e2`, the actual trainer PID 135520/session 64768
and NoTorch archive worker PID 135597/session 37682 were launched. Their
[launch receipt](protocols/adamw_stability_20261002/budget300k-launch-validation.json)
checks all 39 sources, exact original prefixes, native extension metadata and
real first update 150,001 with unchanged rate. The
[complete result](protocols/adamw_stability_20261002/budget300k-result.md)
now verifies all 300,000 updates, every original prefix and all 1,201 canonical
observations. Final train/held-out is 100%/100%, but the frozen final window
has failures at 274,000 and 275,500, minimum held-out 49.1796%. Six observed
episodes all recover; the final sampled target span is 24,250 updates.
Tail failures are 2/201 compared with the original 8/201, but the minimum
accuracy is worse and whole-history onset frequency is nonmonotonic. Both
processes are terminal with code zero, archives verify without Torch, all
five PNG/actual PDF figures were reviewed, and the frozen persistence and
stable-grokking gates remain false. No independent-confirmation plan is opened.

The user requested sparsemax after the current run and pointed to its existing
Lean formalization. The [CPU preparation](protocols/adamw_stability_20261002/sparsemax-preparation.md)
implements that causal Euclidean-simplex projection with original QK/RoPE/
temperature/XSA operations and unchanged parameters. Compare a separately
frozen softmax/sparsemax pair with identical native AdamW, corpus, seeds,
training settings and at most 300,000 total updates per run. Retain complete
quality, losses, frequency/recovery, timing and memory outcomes regardless of
the benchmark gate. If the all-six benchmark requirement has not passed, label
the user-directed experiment exploratory and make no repeatable architecture
improvement claim. Lean's convex row inference does not establish convex joint
training or numerical correctness of the Python implementation.
The [paired pipeline](protocols/adamw_stability_20261002/attention-pair-protocol.md)
is prepared and tested with a real negative full-width CPU pair, portable
verification and reviewed PNG/actual PDF figures. Sparsemax runs first and
a fresh softmax control follows; both full budgets are frozen together before
training. All 43 source fingerprints and the reference/proof provenance are
checked, recovery metrics are derived from full histories, and failed timing
support remains zero. No scientific pair manifest or normalizer run has
started; commit the complete reviewed 300,000-update result before freezing it.
The [harder-task summary](protocols/adamw_stability_20261002/harder-task-results.md)
keeps the three complete single-case AdamW outcomes and their task/exposure
differences together; it does not average interventions or form ineligible ratios.
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
