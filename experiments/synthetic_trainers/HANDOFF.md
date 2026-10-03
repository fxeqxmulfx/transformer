# Synthetic trainer and paper reproduction handoff

State recorded on 2026-10-02. Run commands from the repository root.
Read [AGENTS.md](../../AGENTS.md), [PLAN.md](PLAN.md), and
[STUDIES.md](STUDIES.md) before continuing.

## Current state

All scheduled training in the previous reproduction campaign has finished. Its
serial driver completed with exit code 0; that campaign has no job to resume.
The previous reproduction campaign is finished. A separate follow-up goal
is now active; its calibration bootstrap is described below.
The nine modular runs each completed 150,000 updates, with 601 observations:
1,350,000 updates and 5,409 observations in total. Source hashes, recipes,
budgets, all histories, final observations, and paired split fingerprints were
checked. Calibration seed 0/data seed 0 is excluded from confirmation.

The suite has 17 algorithmic tasks and 37 variants inspired by MQAR, RASP,
RASP-L, and C-RASP, plus a separate random-sequence memorization control.
Memorization/double-descent profiles, fixed label noise, paired size/data
sweeps, free answer generation, and novel-ID/length-transfer diagnostics exist.
The full CPU suite passed 191 tests. Checks cover arithmetic oracles, causal
masking for both models, exact optimizer continuation, independent least-squares
solutions, campaign integrity, failed targets, phase diagnostics, and archives.

| Completed study | Main result | Tracked archive under `baselines/` |
| --- | --- | --- |
| Raw AMSGradW + softmax baseline, 61 runs | No strong delayed length transfer | `amsgradw_softmax_20261002` |
| Data/width/depth scaling, 27 runs | Larger copy models solve ID, still fail longer inputs; no repeated mean parity size-double-descent | `amsgradw_softmax_scaling_20261002` |
| Fashion-MNIST random Fourier features, 123 fits | Model- and sample-wise classification-error double descent in 3/3 seeds; peak n=d=1,000 | `fashion_rff_20261002` |
| Mod-97 confirmation, nine full budgets | Reference grokking repeats; GPTMini optimizer outcomes differ | `mod97_fraction50_wd01_confirmation_20261002` |

The largest trained copy model has 18,893,872 parameters, width 512, six
layers, and eight heads. The 16,384-example pool covers all 6,960 specified
four-bit motif events. The conservative 95% coverage calculation needs 12,118
rows; this is a finite coverage result, not a GPT learning or double-descent
guarantee. Model size cannot be substituted for population EMC.

## Modular results and limits

Confirmation uses prime 97, data seed 1, initialization seeds 1/2/3,
4,656 train / 4,656 exhaustive held-out equations, width 128, two layers,
four heads, batch 512, learning rate 0.001, decay 0.1, warmup 10, and
evaluation every 250 updates. The metric is the author's complete RHS accuracy
(numeric answer and EOS), with a 99% target and two-observation confirmation.

| Model / optimizer | Train≥99%, held-out≤10% plateau then generalization | Restricted epoch-error double descent | Final held-out by seed | Confirmation training seconds, mean ± sample SD |
| --- | ---: | ---: | --- | ---: |
| Author reference / AdamW | 3/3 | 0/3 | 100% / 100% / 100% | 498.55±166.87 |
| GPTMini / AdamW | 1/3 | 1/3 | 100% / 100% / 99.57% | 177.40±283.23 |
| GPTMini / raw AMSGradW | 0/3 | 3/3 | 100% / 97.53% / 100% | 374.56±125.71 |

All nine reached the sustained held-out target during training; AMSGradW seed
2 fails the frozen final target, with 115 incorrect held-out equations.
Raw AMSGradW lags are 18,000 / 35,500 / 34,250 updates. After confirmation,
90.12% / 37.44% / 93.30% of its observations retain 99%; worst held-out
accuracies are 0% / 4.32% / 1.72%. The numeric answers also fail during these
collapses; EOS-only scoring does not explain them. Their cause is unresolved.

This confirms qualitative reference effects, with explicit deviations.
Reference generalization occurs earlier than the manuscript's >100,000
updates; train fraction/regularization were calibrated because the local
manuscript omits them. GPTMini changes architecture and initialization.
The epoch-error diagnostic is post hoc and excludes peaks after first
generalization; it is an adaptation of the paper's noisy CIFAR experiment.
Random-feature normalization, complex classification rule, seeds, and
infinite-time solver are explicit choices where the paper is silent. Its
peak is about 86%, rather than the figure's approximately 75%.
These effects do not identify an internal algorithm or establish that
grokking causes double descent. Three initializations on one modular split
are not three independent data splits, and sample SD is not a confidence
interval. First-target timing does not imply persistent convergence.
Modular peak CUDA allocation was not measured; model size/device were recorded.

The original campaign used the repository's raw AMSGradW, distinct from
bias-corrected PyTorch AdamW with `amsgrad=True`: betas (0.9, 0.999), a maximum second moment,
no bias correction, epsilon 1e-8, and decoupled decay on all trainable parameters.
The reference uses AdamW betas (0.9, 0.98). Its update was independently
matched against the pinned author's optimizer over 20 warmup updates.
See [MODULAR.md](paper_reproduction/MODULAR.md) and
[RANDOM_FEATURES.md](paper_reproduction/RANDOM_FEATURES.md) for source fidelity.

## Files and migration

Git contains trainer/model code, tests, licensed author model, source metadata,
all curated histories/configurations/CSV, and standalone PNG/PDF figures.
The combined modular archive can be analyzed without the original run paths
or model checkpoints. Individual seed archives and all three unsuccessful or
successful exploratory calibrations also remain under `baselines/`.
Recent per-seed commits are `fc1f7ef`, `eba9204`, and `314d05a`.
The commit containing this handoff also contains the combined report.
Transfer the local Git history as well as the checkout: copy the repository
including `.git`, or use a Git bundle. A remote checkout contains these local
commits only after they have been transferred to that remote.

Copy these ignored/local resources separately when needed:

- `papers/` (currently about 228 MB): the authoritative local manuscripts;
  repository instructions require reading these instead of browsing their contents.
- `experiments/runs/paper_data/fashion-mnist/` (about 30 MB): official compressed
  IDX data and checksums, needed to rerun random-feature fits.
- `experiments/runs/paper_reproduction/mod97_fraction50_wd01_confirmation_20261002/`
  (about 59 MB): final/resumable model and optimizer checkpoints plus raw logs,
  needed for weight or moment inspection. Archived analysis does not need these.
- Other `experiments/runs/` subdirectories if old baseline/scaling weights are
  needed; curated histories are already tracked.
- `/tmp/openai-grok-reference/` (about 84 KB): cached author source at
  `3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3`, useful for source rechecks.

Recreate the Python environment in the new checkout. Recorded versions are
Python 3.14.7, PyTorch 2.14.0+cu130, and NumPy 2.5.3. Plotting used system
Matplotlib 3.10.7+dfsg1. Training used an RTX 3050 Laptop GPU with 4 GB VRAM.
Record any environment/GPU changes. Compare paired timing controls on the new
machine; historical seconds on the old GPU are descriptive reference values.

Frozen training hash dictionaries currently contain absolute paths. A checkout
at a different root can therefore fail strict campaign `--resume` even when
file contents match. Use portable archives for completed results and fresh
directories/manifests for new studies; preserve historical manifests.

Useful checks and plot commands after restoring dependencies:

```bash
git status --short
python3 -m experiments.synthetic_trainers check --examples 128
.venv/bin/python -m unittest discover -s experiments/synthetic_trainers/tests
MPLCONFIGDIR=/tmp/synthetic-trainer-mpl python3 \
  -m experiments.synthetic_trainers.paper_plots modular \
  experiments/synthetic_trainers/baselines/mod97_fraction50_wd01_confirmation_20261002
MPLCONFIGDIR=/tmp/synthetic-trainer-mpl python3 \
  -m experiments.synthetic_trainers.paper_plots rff \
  experiments/synthetic_trainers/baselines/fashion_rff_20261002
```

Regenerating a PDF may change its metadata. Preserve archived measurements and
source fingerprints. Fresh archive destinations are required by `paper_report`.
Main entry points are `paper_reproduction/grokking.py`, `reproduction.py`,
`paper_reproduction/rff_experiment.py`, `paper_phases.py`, `paper_report.py`, and
`paper_plots.py`; the model is [gpt_mini.py](../gpt_mini.py).

No Lean source changed in this experimental campaign. The current generated
[INDEX.md](../../INDEX.md) records 1,766 modules, 7,099 theorems, and 157
`sorry` globally; RASP, RASP-L, and C-RASP rows each record zero `sorry`.
This index is not a new full-tree axiom audit. Follow the Lean build, paper
fidelity, external-dependency audit, and index requirements for future proofs.
[AGENTS.md](../../AGENTS.md) now requires regular commits of verified logical
changes and forbids subagents. No new approval is needed for already authorized
reversible work or regular commits.

## Active follow-up bootstrap (2026-10-02)

The user explicitly selected AdamW as the primary optimizer for the continuing
benchmark on 2026-10-02. Raw AMSGradW remains a paired control, with every
completed negative result retained. Phase, persistence, independent-confirmation,
architecture and complementary-task requirements are unchanged. Historical
AdamW has fewer sampled failures but is not yet a verified stable benchmark.

The completed scientific stage is the prospectively frozen
[AdamW/raw AMSGradW optimizer pair](protocols/adamw_stability_20261002/optimizer-pair-protocol.md).
Both recipes retain learning rate 0.001, the same GPTMini model/data/seeds,
short-final batches, decay 0.1, and 150,000-update budgets. Only optimizer
selection differs. AdamW (betas 0.9/0.98, bias correction, no maximum buffer)
is primary and runs first; raw AMSGradW is the paired control. This compares
whole optimizers, not individual update components. Both record every gradient
norm; no clipping is introduced. The new native moment diagnostics and unchanged
six-case gate are verified with 224 passing CPU tests and exact real GPU
instrumentation-equivalence checks. Old raw archives still verify offline.

The pair was frozen at 2026-10-02T13:21:44.775126+00:00 from source commit
`08ed18c` before scientific training. The immutable
[plan](protocols/adamw_stability_20261002/optimizer-pair-plan.json) includes both
recipes, criterion, sources, analysis/confirmation sources, environment, local
papers, split fingerprints and launcher SHA256. Launch/resume from the root:

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_optimizer_pair
```

Add `--check-only` for verification. Live data will be under
`experiments/runs/adamw_stability_20261002/calibration_optimizer_pair_lr001/`.
Do not use the default raw four-recipe CLI for this custom pair. Archive and
commit each full result independently, then assemble the complete comparison.
Only a passing primary AdamW calibration permits new independent confirmation;
architecture and scientific complementary mechanics remain pending.

The optimizer-pair trainer (initial PID 112572, session 92762) and replacement
archive worker (PID 117200, session 10373) finished with exit code 0. Both
terminal handles were consumed; do not restart or poll them. The complete
pair preserves 300,000 updates and 1,202 canonical observations. Driver output is
`experiments/runs/adamw_stability_20261002/optimizer-pair-driver.log`;
worker metadata is under `experiments/runs/adamw_stability_20261002/bootstrap/optimizer-pair-archive-worker.json`.
The first full dense archive exposed a quadratic CSV column-union loop.
The [verified runtime repair](protocols/adamw_stability_20261002/csv-verification-repair.md)
preserves every frozen file and scientific check, while computing the identical
columns once. All 235 CPU tests pass; original staged and final AdamW summaries
and measurement files agree. Original non-training workers 112672/115191 were
deliberately stopped, with terminal sessions 78057/18664 (exit 143). Replacement
archive worker PID 117200/session 10373 and queued task worker
PID 117203/session 46645 have completed with exit code 0. Their terminal
handles are consumed. The original GPU trainer was not restarted.
The adapter SHA256 and exact worker source snapshots are preserved separately;
the mod-193 launcher's frozen checks passed through the adapter. Both frozen
campaigns have finished. The
[ordinary-verifier repair](protocols/adamw_stability_20261002/production-csv-repair.md)
is now verified after both frozen campaigns: all 235 CPU tests pass without
skips, and both verification paths return identical complete archive results.
Future plans use the new analysis SHA256; historical strict launchers require
their recorded source version. Completed archives verify with current ordinary tools.
See the [launch validation](protocols/adamw_stability_20261002/launch-validation.json).
The [early AdamW prefix](protocols/adamw_stability_20261002/first-long-confirmation.md)
now preserves all 25 canonical and 6,000 gradient observations through update
6,000. First held-out target is 1,250; long confirmation spans 1,250–6,000 and
costs 181.81 training / 194.91 wall seconds. There is no required low-held-out
memorization plateau, so this 50% recipe is phase-ineligible regardless of later
persistence. It is rapid generalization, with persistence tracked separately.
The [first final-tail failure](protocols/adamw_stability_20261002/first-tail-failure.md)
is now preserved at update 104,000: exhaustive train/held-out accuracy
90.0129%/87.9940%. Both immediate neighbors also fail; EOS stays 100%.
This observed final-window failure rules out the frozen persistence criterion,
irrespective of later recovery. Its portable prefix verifies without PyTorch.
The [full primary result](protocols/adamw_stability_20261002/adamw-result.md)
is now complete and archived: 150,000 updates, all 601 canonical/1,200 neighbor/
1,800 tensor/150,000 gradient observations, with final train/held-out 100%.
Seven of 201 final-window observations fail; worst held-out is 43.9003%
at 129,000. Phase and persistence both fail. Full costs are 4,022.04 training /
4,284.18 wall seconds, with 44.04 diagnostic seconds. Both PNG/PDF curve pairs
were reviewed and the archive verified without PyTorch; original bytes and
frozen fingerprints agree. The
[complete raw control](protocols/adamw_stability_20261002/raw-amsgradw-result.md)
also finished: final train/held-out 100%/99.9785%, absent required plateau,
ten final-window failures and worst held-out 58.1400%. The
[complete optimizer comparison](protocols/adamw_stability_20261002/optimizer-pair-result.md)
verifies both original archives and byte-identical nested copies without
PyTorch. Both have empty eligible persistent-target timing support. Original
curves and reviewed standalone PNG/PDF figures with readable update ticks
retain all observations. Neither recipe permits independent confirmation.
The separately frozen mod-193 adaptation is now complete; the later
train-fraction control and lower-rate calibration are recorded below.
After each full result, review figures, commit the verified archive and measured
outcome, and apply the unchanged gate. No scientific independent confirmation
or architecture run has started.

The user next requested a harder task. The separately frozen
[mod-193 adaptation](protocols/adamw_stability_20261002/larger-modulus-protocol.md)
changes only `prime` in the primary AdamW configuration, retaining the model
block dimensions, 50% split, seeds, rate/decay, full 150,000-update budget,
instrumentation and unchanged phase/persistence criterion. Its 37,056 legal
pairs give exhaustive 18,528/18,528 splits. Vocabulary growth changes total
parameters to 436,104 (blocks remain 393,224); short tails become 96 and
full-budget exposure becomes 75,113,536 examples. These derived changes are
reported explicitly, not treated as isolated arithmetic difficulty.
The complete oracle corpus and both split class coverages are checked; a real
38-update full-width CPU smoke and portable archive verification passed.

Its [manifest](protocols/adamw_stability_20261002/larger-modulus-plan.json) was
frozen at 2026-10-02T13:42:36.938813+00:00 from `6e172bb`.
Scientific mod-193 training started at 2026-10-02T15:50:59.797959+00:00,
after both mod-97 budgets and their complete verified comparison:

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.csv_verification \
  module experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_larger_modulus
```

Use `--check-only` to verify the new manifest before parent completion.
Live output will be under `experiments/runs/adamw_stability_20261002/calibration_mod193_lr001/`.
The replacement serial queue worker (PID 117203, session 46645) and child
trainer PID 119293 have completed. They executed the unchanged launcher
through the verified CSV adapter. Original command, scientific state/history
and frozen fingerprints are checked in the
[launch validation](protocols/adamw_stability_20261002/larger-modulus-launch-validation.json).
The worker created and verified both full-budget portable archives. Both
processes are terminal; do not restart or poll their consumed handles. See the
[original queue validation](protocols/adamw_stability_20261002/larger-modulus-queue-validation.json),
[runtime-repair validation](protocols/adamw_stability_20261002/csv-verification-repair-validation.json)
and `experiments/runs/adamw_stability_20261002/bootstrap/queued-mod193-worker.json`.
Check actual process/state before restart; the driver log will be
`experiments/runs/adamw_stability_20261002/mod193-driver.log`.
The [first-target prefix](protocols/adamw_stability_20261002/first-target-mod193.md)
now preserves all 161 canonical / 40,000 gradient observations through update
40,000, where train/held-out accuracy is 99.5952%/99.3253%. First train 99%
is at 39,500 with held-out already 98.5320%; zero pre-target observations meet
the required memorization condition. The recipe is phase-ineligible regardless
of future persistence. Finish its full budget and retain the final-tail outcome;
this prefix does not open independent confirmation.
The [first final-window failure](protocols/adamw_stability_20261002/first-tail-failure-mod193.md)
is now preserved at 102,750: train/held-out 95.1749%/94.6082%, with both
immediate neighbors below target and EOS 100%. The canonical failure follows
a full 512-example batch. Its complete 412-point prefix and 252-gradient /
five-tensor neighborhood verify offline with semantic corruption checks.
The recipe now fails both phase and persistence regardless of later recovery.
The [complete mod-193 result](protocols/adamw_stability_20261002/larger-modulus-result.md)
now preserves the whole 150,000-update budget and all dense/sampled logs.
Final train/held-out is 100%/100%, but seven of 201 tail observations fail,
with worst held-out 79.8953% at 148,000. First long confirmation is
52,250–57,000, with 1,436.21 training / 1,779.09 wall seconds; eligible timing
support is zero. Complete costs are 3,750.27 training / 4,648.78 wall seconds.
Both archives verify without PyTorch; source/checkpoint bytes, all 31 frozen
fingerprints and earlier prefixes agree. All three PNG/PDF figure pairs were
reviewed; the original comparison has overlapping ticks and is preserved
alongside a reviewed [readable PNG/PDF sidecar](protocols/adamw_stability_20261002/larger-modulus-curves/mod193-calibration.png)
with all 601 points and verified provenance. The standard renderer now uses
thousands for update labels and a wider single-case layout; its three comparison
tests pass. Future plans record its new analysis hash. No scientific
trainer remains in this completed stage; the later interventions are below.
At selection time, this user-directed harder-task adaptation superseded the
earlier tentative train-fraction idea. Its historical prefix inspections
remain partial evidence with their original labels.

After the absent pre-target phase was established, a separate
[25% fraction preparation](protocols/adamw_stability_20261002/fraction25-preparation.md)
checked exhaustive nested splits, identical CPU initial model states and
unchanged 436,104 parameters. A real 20-update full-width CPU smoke checks
the 48-example tail / next 512-example batch and verifies its portable archive
without PyTorch. Those smoke measurements are preparation only. The parent
full budget, archive and ordinary CSV-verifier repair are complete. The
[25% scientific plan](protocols/adamw_stability_20261002/fraction25-protocol.md)
is now prospectively frozen at 2026-10-02T17:26:11.071473+00:00 from `5bdc43d`.
Its [manifest](protocols/adamw_stability_20261002/fraction25-plan.json) changes
only the training fraction, with unchanged task/model/optimizer sources,
environment, papers, instrumentation and all phase/persistence gates. It records
the two verified analysis-only repairs. The real launcher check and isolated
negative manifest tests are recorded in the
[freeze validation](protocols/adamw_stability_20261002/fraction25-freeze-validation.json).
Scientific training has completed: trainer PID 123201/session 61726 and
archive worker PID 123310/session 95343 both exit with code 0; terminal
handles are consumed and must not be restarted or polled. The worker has no
PyTorch imported, checks all immutable fingerprints, creates both complete
archives with PNG/PDF figures, and verifies them offline with the ordinary
tools. Actual commands, state and fingerprints
are recorded in the
[launch validation](protocols/adamw_stability_20261002/fraction25-launch-validation.json).
Driver output is `experiments/runs/adamw_stability_20261002/fraction25-driver.log`;
worker metadata is `experiments/runs/adamw_stability_20261002/bootstrap/fraction25-archive-worker.json`.
Check these actual processes before restart. The
[verified phase prefix](protocols/adamw_stability_20261002/first-long-confirmation-fraction25.md)
now preserves all 100 canonical and 24,750 gradient observations through the
first long confirmation. The longest qualifying memorization plateau spans
4,500–11,500 (29 observations), followed by first held-out 99% at 20,000 and
20 consecutive joint target observations through 24,750. Standalone PNG/PDF
figures show every canonical point. Six corrupted copies are rejected offline,
including absent actual memorization after recomputing its assessment; all
31 frozen files remain unchanged. This is observed delayed generalization in
one incomplete calibration. Finish all 150,000 updates and score the unchanged
100,000–150,000 persistence window before applying the six-case confirmation gate.

The [quarter-split final-tail failure](protocols/adamw_stability_20261002/first-tail-failure-fraction25.md)
now preserves all 461 canonical observations through 115,000, five exhaustive/
full-tensor samples and 252 gradient records around the first failed final-window
evaluation. Train / held-out accuracy is 92.7461%/91.6271%; EOS remains 100%
and both immediate neighbors fail on full batches. The observed memorization
and generalization phases remain valid, but this single final-window failure
refutes the unchanged persistence gate and prevents independent confirmation.
Existing offline verification and four repaired-hash semantic corruptions pass;
all 31 frozen files remain unchanged. The trainer and archive worker continued
the whole budget in that earlier inspection. The
[complete quarter-split result](protocols/adamw_stability_20261002/fraction25-result.md)
is now verified: all 150,000 updates and 601 canonical observations, final
100%/100%, but three final-window failures at 115,000/119,250/139,000. The
worst final-window canonical held-out accuracy is 48.9206% at 139,000. The observed phase
passes, persistent performance fails and the six-case gate stays closed.
All 1,200 neighbors, 1,800 tensor samples, 150,000 gradient records, source
bytes and nested archive copies verify without PyTorch; all three PNG/PDF
pairs are retained, with readable PNGs reviewed. No criterion is relaxed.
The [harder-task summary](protocols/adamw_stability_20261002/harder-task-results.md)
compares all three complete primary AdamW calibrations, retaining the different
phase outcomes, every persistence failure and derived task/exposure changes.
Its offline audit verifies 450,000 updates and 1,803 canonical observations;
the three cases have zero eligible stable-target timing support.

The [lower-rate CPU preparation](protocols/adamw_stability_20261002/lower-rate-preparation.md)
checks learning rate 0.0003 as the only field change from the quarter-split
parent. Complete corpus, initial CPU state, parameters, exposure and all
31 frozen files match; 20 full-width updates verify warmup, the 48-example tail,
the next full batch and portable analysis without PyTorch. CUDA is hidden from
the CPU process and no CUDA context is created. This preparation overlapped
the parent's final training portion, which matters for descriptive timing.
It is not a scientific plan, launch or demonstrated stability result. The
complete parent negative archive is verified and its figures reviewed;
the subsequent scientific freeze and launch are recorded below.

The [lower-rate scientific plan](protocols/adamw_stability_20261002/lower-rate-protocol.md)
is now prospectively frozen at 2026-10-02T19:11:49.393424+00:00 from `bee6e44`.
It changes only learning rate to 0.0003; all 31 training/analysis/manuscript
fingerprints, corpus, environment, instrumentation and criteria match the
complete quarter-split parent. Actual `--check-only` passes; four isolated
changed manifests are rejected, and the pinned archive worker imports without
PyTorch. Scientific GPU training had not started at that freeze record.
After manifest commit `ed83c85`, the actual trainer PID 126400/session 93469
and archive worker PID 126478/session 19327 launched. Their commands,
state, optimizer and all immutable fingerprints are verified in the
[launch receipt](protocols/adamw_stability_20261002/lower-rate-launch-validation.json).
At 2026-10-02T19:26:27+00:00, the incomplete history reaches update 13,250:
train/held-out accuracy 99.9028%/0.8096%, with a qualifying memorization plateau
from 4,500 through 13,250 (36 canonical observations). Neither held-out 99%
nor long confirmation has occurred. Keep the full 150,000-update budget;
this partial phase is not a stability or repeatability result.

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_lower_rate
```

Driver output is `experiments/runs/adamw_stability_20261002/lower-rate-driver.log`;
worker metadata is `experiments/runs/adamw_stability_20261002/bootstrap/lower-rate-archive-worker.json`.
Both processes have now completed and their sessions were consumed. The worker
archived and verified the full budget without importing PyTorch. The complete
negative result below preserves every observation and the original frozen files.
Do not resume or rerun this historical 150,000-update stage.
The [phase-prefix exporter](protocols/adamw_stability_20261002/record_phase_prefix.py)
is verified outside that frozen source list: real parent replay matches all
five previously archived prefix files byte for byte, an existing destination
is rejected, and the actual unconfirmed lower-rate stage creates no positive
archive. It imports no PyTorch and reuses the unchanged semantic verifier.
Use it after the first ordered-phase long confirmation, preserve exact source
lines, and plot/review the explicitly partial result in a fresh destination.

The [lower-rate confirmed phase](protocols/adamw_stability_20261002/first-long-confirmation-lower-rate.md)
now preserves all 403 canonical / 100,500 gradient observations through the
first twenty joint targets at 95,750–100,500. First train 99% is at 1,250;
the longest qualifying plateau is 4,500–27,250 (92 points), and first held-out
99% is 95,750. Observed confirmation costs are 2,765.91 training / 3,375.93
wall seconds. The exact prefix verifies without PyTorch; all 31 core hashes,
both launchers and original lines match. Both standalone PNG and the actual
PDF were visually reviewed, with every point, rate/seeds and explicit partial
scope shown. Historical quarter-split figure bytes remain unchanged. The prefix
has only three passing tail points. Later canonical held-out 97.4813% at
105,250 already refutes persistence, with train and EOS 100%; the validated
later point is retained separately beside the prefix audit. Finish the full
budget and retain all failures; independent confirmation remains closed.
The full result has since completed, as recorded below.

The [complete lower-rate result](protocols/adamw_stability_20261002/lower-rate-result.md)
now verifies 150,000 updates, all 601 canonical/1,200 neighbor/1,800 diagnostic
observations and all 150,000 gradient norms. Final train/held-out is 100%/100%,
but eight of 201 final-window observations fail; both frozen gates are negative.
Every failed point has numeric-answer failures at both immediate neighbors and
EOS 100%. All canonical failure batches are full; neighbor 110,751 is the
48-example epoch tail. All original files and nested archive copies match,
the local checkpoint and 31 core fingerprints match, and both actual processes
are terminal. All original PNG and actual PDF figures were visually reviewed.

The user requested additional [frequency and recovery metrics](protocols/adamw_stability_20261002/recovery-metrics.md).
Thirteen affected/persistence tests pass; four historical complete archives keep
their original gates and tail supports. The new full lower-rate report counts
three episodes, recovering after 500/250/1,250 sampled updates. Fixed 10,000-
update tail windows have onset counts 2/1/0/0/0 and failed fractions
7.5%/12.5%/0%/0%/0%, with supports 40/40/40/40/41. Its final 153 joint targets
span 112,000–150,000 (38,000 updates). Separate recovery PNG/PDF and CSV were
reviewed and hashed. These descriptive metrics do not change the strict gate.

The user then specified a maximum total budget of 300,000 training updates per
run. Preserve the negative original 150,000-update stage and prepare a new
frozen extension before any additional updates. Change only total steps;
restore weights, native AdamW buffers/steps and shuffle/cursor state. Freeze
the new 250,000–300,000 final window and the 10,000-update recovery metric width
and source hash. This remains exploratory calibration. Fresh six-case full-
budget confirmation, architecture controls and scientific complementary tasks
remain outstanding. The user has since requested sparsemax after this run;
its preparation and Lean specification are recorded below.

The [300,000-update continuation](protocols/adamw_stability_20261002/budget300k-protocol.md)
is now frozen at 2026-10-02T21:15:19.158465+00:00 from implementation commit
`5c744d8`. The actual launcher check passes; eight isolated negative fixtures
are rejected. All original checkpoint/log copies match, native steps are
150,000, and the 39 core/manuscript/extension/recovery files are pinned.
The full-width CPU continuation matches an uninterrupted run in weights,
moments, native steps, sampling state, canonical metrics and dense diagnostics.
Three extra series tests pass; five complete historical cases retain every
post-onset observation and failure. The new total budget is 300,000 updates,
with final tail 250,000–300,000 and fixed 10,000-update recovery windows across
the whole post-generalization history. Original completed archives stay intact.
The manifest was committed as `3cf14e2` before launch. Trainer PID 135520 /
session 64768 and NoTorch archive worker PID 135597 / session 37682 were
launched as recorded in the [launch receipt](protocols/adamw_stability_20261002/budget300k-launch-validation.json).
The actual first added update is 150,001 at rate 0.0003; the first added
canonical evaluation is 150,250. Both commands, every original prefix, the
new child budget-extension metadata and all 39 immutable hashes match.
The [complete 300,000-update result](protocols/adamw_stability_20261002/budget300k-result.md)
now retains 1,201 canonical, 2,400 neighboring, 3,600 diagnostic and 300,000
gradient observations. Final train/held-out is 100%/100%, but two of the 201
frozen final-window points fail, at 274,000 and 275,500. Worst canonical
held-out is 49.1796% at 274,000; the latter failure is 81.7681%. Both recover
at the next canonical observation. Six post-long-onset episodes are observed;
the final target streak is 275,750–300,000, spanning 24,250 updates. Tail
failure support declines from the original 8/201 to 2/201, but the minimum
accuracy is worse and whole-history onset rates are nonmonotonic. Persistence
and stable-grokking eligibility remain false; independent confirmation is closed.
Both scientific processes finish with code zero and their sessions are consumed.
All original prefixes/nested archives and 39 source files match; verification
without Torch and all five original PNG/actual PDF reviews are retained in
the [receipt](protocols/adamw_stability_20261002/budget300k-result-validation.json).
Driver output is `experiments/runs/adamw_stability_20261002/budget300k-driver.log`;
worker metadata is `experiments/runs/adamw_stability_20261002/bootstrap/budget300k-archive-worker.json`.
The trainer and worker are terminal; do not restart this completed run.
Freeze the user-directed normalizer pair after committing the complete result.

The next user-directed normalizer candidate is **sparsemax**, with a paired
softmax control preserving AdamW, corpus, seeds, the total 300,000-update cap,
QK normalization, learned temperatures, RoPE and XSA. The
[preparation](protocols/adamw_stability_20261002/sparsemax-preparation.md)
uses the exact causal-simplex projection already formalized in
[Lean](../../src/Transformer/GPTMini/Convex/Attention.lean).
Six independent CPU checks pass, including exhaustive simplex-face KKT
solutions, finite-difference derivatives, causal prefixes, parameter/RNG
preservation and exact restoration of original softmax outputs/gradients.
The [tagged training adapter](attention_training.py) passes five further CPU
checks: original softmax optimization is exact, sparsemax resume preserves
native buffers/sampling/warmup, wrong-normalizer/ordinary-core resumes are
rejected before loading, and factory/source patches restore after exceptions.
The actual 436,104-parameter mod-193 initial tensors and RNG match for both
normalizers, with finite CPU forward/backward and all 39 live hashes unchanged;
see the [receipt](protocols/adamw_stability_20261002/sparsemax-training-validation.json).
The [complete paired pipeline](protocols/adamw_stability_20261002/attention-pair-protocol.md)
now passes four additional tests and twelve invalid manifest variants. A real
20+20-update full-width CPU pair retains its negative outcomes, verifies
without Torch and has all six original PNG and actual PDF figures reviewed;
see the [receipt](protocols/adamw_stability_20261002/attention-pair-preparation-validation.json).
Its scientific order is sparsemax first, then a fresh softmax control, with
both full 300,000-update configurations frozen before either case. The serial
driver, per-case archive worker, recovery comparison and all 43 source
fingerprints are ready. The [scientific manifest](protocols/adamw_stability_20261002/attention-pair-plan.json)
is now frozen at 2026-10-02T22:47:04.605873 UTC after complete reference-result
commit `8785c6b`. Both full budgets, sources, nine Lean fingerprints and the
byte-exact complete reference copy are checked. The actual launcher passes;
eighteen isolated negative scientific fixtures are rejected before training,
as retained in the [manifest receipt](protocols/adamw_stability_20261002/attention-pair-manifest-validation.json).
The manifest was committed as `a76e0b6` before scientific launch. Trainer PID
139507/session 47567 remains live. The initial NoTorch archive worker PID
139645/session 69196 was retired because its system Python lacks matplotlib;
its terminal exit 143 was consumed. The unchanged worker now runs in the
existing venv as PID 145323/session 25387, with matplotlib 3.11.2 and no Torch
import. The original log is preserved by append. The GPU trainer was not
restarted; all 43 Python/nine Lean fingerprints still match. See the
[worker repair receipt](protocols/adamw_stability_20261002/attention-pair-archive-worker-repair.json).
The [launch receipt](protocols/adamw_stability_20261002/attention-pair-launch-validation.json)
checks the real case's 436,104 parameters, native AdamW, all seventeen training
fingerprints, original corpus/instrumentation, fresh step-one warmup, empty
budget-extension history, all 43 Python/nine Lean sources and both commands.
At 3,750, train/held-out is 99.9028%/0.7376%, with finite gradients. This is
an incomplete scientific prefix, not a phase/persistence or improvement result.
The fresh softmax case has not started and will run automatically after the
whole sparsemax budget. Keep every frozen source immutable while either process
is active. Inspect state/processes before any restart and review/commit each
complete archived case while the serial driver proceeds.
A user-directed exploratory pair may be measured before the all-six
benchmark gate opens; that does not certify repeatability or an architecture
improvement. Keep accuracy, loss, episode frequency/recovery, actual time and
memory, and retain every failure. The six-case requirement for an improvement
claim and scientific complementary tasks remain outstanding.

While that immutable pair runs, a separate
[conditional fixed-schedule preparation](protocols/adamw_stability_20261002/schedule-preparation.md)
is verified with five CPU checks. It retains the original constant native-AdamW
trajectory exactly and supports a prospective cosine reduction after 150,000,
ending at 250,000 with a fixed 0.1 rate factor over the final window. Tagged
native continuation preserves buffers, sampling and actual recorded rates;
portable scheduled archives retain all original non-rate checks and reject
forged rates after rehashing. The actual scientific initial tensors/RNG match
on CPU, with 436,104 parameters and all current 43 Python/nine Lean sources
unchanged. This is preparation only: no scientific schedule is selected,
frozen or launched. Finish/review/commit the current pair first; use the
unchanged independent gate if its fresh softmax control passes.
The conditional paired launcher/archive/recovery pipeline is now implemented
and validated: four additional protocol tests plus the two affected archive
tests pass (six tests, 5.403 seconds). A real full-width CPU pair completes
40 updates per case, archives both negatives, verifies without Torch and
retains actual rates. All six original PNG figures and six actual PDF renders
are inspected. See the [paired receipt](protocols/adamw_stability_20261002/schedule-pair-preparation-validation.json).
The actual scientific freezer rejects the incomplete live normalizer pair
before schedule selection or GPU initialization. A future scientific pair
requires both complete reviewed/committed normalizer results, a failed primary
softmax criterion, then a separately committed full-budget manifest. A passing
primary control instead enters six independent confirmations. Run any future
plotting archive worker with `.venv/bin/python`; the system Python remains
useful for verification without Torch. The current pair's frozen 43 Python
and nine Lean sources remain unchanged.

The separate [tagged confirmation adapter](protocols/adamw_stability_20261002/tagged-confirmation-preparation.md)
now prepares the first fresh model-seed 4/5/6 × data-seed 2/3 cohort for a
passing native softmax or scheduled-softmax recipe. It retains the exact
seventeen-source trainers; the old ordinary-config driver cannot directly
consume these tags/source sets. Four tests pass in 12.484 seconds. Two actual
full-width six-case CPU fixtures complete twelve cases and 360 updates, keep
every negative, verify without Torch and have no scientific benchmark claim.
The four new combined PNGs and their four actual PDFs are inspected; reference
calibration artifacts remain byte-identical. See the
[receipt](protocols/adamw_stability_20261002/tagged-confirmation-preparation-validation.json).
No scientific confirmation is selected, frozen or started. The actual freezer
rejects the incomplete current pair before creating a manifest or GPU context.
A full passing primary result, committed reviewed evidence and a new committed
manifest remain required. Every scientific case is capped at 300,000 updates;
all six must pass the unchanged criterion before architecture selection.
Use `tagged_confirmation_report.verified_benchmark` for this tagged pipeline;
the prepared schedule's 55 and active normalizer's 43 Python/nine Lean sources
are unchanged. A failed scientific cohort consumes its seeds and requires a
new prospective cohort for any later confirmation.

```bash
cat experiments/runs/adamw_stability_20261002/attention_mod193_fraction25_lr0003_budget300k/state.json
cat experiments/runs/adamw_stability_20261002/bootstrap/attention-pair-archive-worker.json
tail -n 3 experiments/runs/adamw_stability_20261002/attention-pair-driver.log
```

The [complementary-trainer preparation](protocols/adamw_stability_20261002/complementary-preparation.md)
also verifies seven existing AdamW CPU paths for MQAR, one/two-hop lookup,
copy, direct/running parity and C-RASP counting. Final and selected checkpoint
reloads reproduce all ID/longer-input metrics, with separate novel-input support.
The general trainer's matrices-only decay and absent warmup differ from the
modular primary protocol; a later scientific plan must explicitly resolve or
label those differences. No production source changed, no architecture was
selected and no scientific complementary campaign began. The current frozen
GPU/queued source fingerprints remain unchanged.

The separate [native complementary adapter](protocols/adamw_stability_20261002/complementary-preparation.md#explicit-native-adamw-adapter)
now resolves the optimizer differences through an opt-in path: native AdamW,
all-parameter decay, betas (0.9, 0.98), epsilon 1e-8, no clipping, explicit
matrix std 0.02, ten-update warmup and constant/fixed-cosine rate schedules.
Every run is capped at 300,000 updates and retains clean disjoint study pools,
both model checkpoints, CPU optimizer state and every applied group rate.
Four targeted tests pass in 3.125 seconds; actual parameters/moments exactly
match an independent native loop for both schedules. Eight real CPU fixtures
finish 128 updates with separate four-example ID/length novel support and
independently reproduced final/selected checkpoint predictions. All final and
selected test sequence accuracies are zero; these are implementation checks.
The full archives verify without Torch and retain source snapshots/raw data.
See the [receipt](protocols/adamw_stability_20261002/complementary-native-preparation-validation.json).
The active 43 Python/nine Lean, prepared 55 and both 62-source manifests remain
unchanged. No scientific complementary campaign or architecture is selected;
the current sparsemax-first pair and all-six primary gate still come first.

The [prefix-counting oracle audit](protocols/adamw_stability_20261002/complementary-preparation.md)
now independently checks all 2,187 length-seven words and 15,309 inclusive
prefix labels for the prepared C-RASP program, plus 128 length-fifteen words.
Equal-frequency words with distinct final labels explicitly witness order
sensitivity. All 15 existing prefix-program tests pass without PyTorch;
the audit replays exactly in JSON normal form and rejects a constant oracle.
This finite preparation audit records the additional C-RASP manuscript hash;
it is not scientific training, transfer evidence or a change to the modular gate.

The follow-up goal from the startup message is active, without subagents.
Read [STABILITY.md](STABILITY.md) for its prospectively specified criteria,
four paired calibration recipes, and independent-confirmation gate.
The restored checkout has a GTX 1050 (Pascal, 2 GiB), Python 3.12.13,
PyTorch 2.14.0+cu126, NumPy 2.5.3, and Matplotlib 3.11.2. The preexisting
PyTorch 2.7.1+cu118 was upgraded before calibration. A real 20-update GPU
smoke run passed with peak allocation 119,166,464 bytes.

Local manuscripts and tracked complete archives are present. Historical model
checkpoints, Fashion-MNIST IDX data, and the temporary author-source cache
were not transferred. Fresh modular training uses the complete oracle corpus
and tracked licensed reference code; these missing historical resources do
not prevent it. Historical timing is descriptive, with paired controls on
this GPU required for new comparisons.

Read-only gradient/moment/update/temperature diagnostics, neighbor probes,
portable fingerprints, and an explicitly labeled full-batch epoch-wrap control
are verified and committed as `a0dcee2`. CPU checks preserve exact model,
optimizer buffers, canonical scores, and checkpoint continuation. The stronger
criterion requires a memorization plateau, 20 consecutive joint 99% scores,
and joint 99% throughout the last 50,000 updates. None of the nine historical
runs satisfies that prospective persistence criterion when assessed
retrospectively; the old frozen successes remain unchanged. See the
[baseline assessment](protocols/amsgradw_stability_20261002/historical-persistence.json)
and [restored environment](protocols/amsgradw_stability_20261002/environment.json).

The four-run calibration plan was frozen at 2026-10-02 06:04:54 UTC after
208 CPU tests passed on the upgraded environment. Sources and split fingerprints
were rechecked before launch. All four budgets have now completed. The serial
driver (initial PID 87620) and archive worker (PID 96589) both exited with code
0; their absence was checked. All four individual portable archives and the
complete comparison are verified without PyTorch, and the new figures have
been visually inspected. No job remains in this original four-recipe stage.

All four fresh calibrations finish 150,000 updates each (600,000 total), with
one mechanism changed at a time: unchanged learning rate 0.001, full-batch
sampling at 0.001, and short-tail learning rates 0.0003/0.0001. All use
initialization/data seed 0, the 50% split, and decay 0.1. The immutable
[frozen plan](protocols/amsgradw_stability_20261002/calibration-plan.json) records
source commit `ff1d122`, all recipes, prospective targets, environment, and
source/data/manuscript hashes. Live state, complete histories, diagnostics,
neighbor probes, and checkpoints are under
`experiments/runs/amsgradw_stability_20261002/calibration/`; the driver log is
`experiments/runs/amsgradw_stability_20261002/calibration-driver.log`.

The archive worker completed the fourth individual archive and the complete
[calibration comparison](baselines/amsgradw_stability_calibration_20261002/REPORT.md).
The comparison retains all 600,000 updates, 2,404 canonical observations,
4,800 neighbor probes, and 7,200 diagnostic records. All four recipes fail the
unchanged criterion; the eligible calibration list is empty. Long-confirmation
timing is retained where observed, with null timing and support zero for the
lowest rate. See the [comparison verification](protocols/amsgradw_stability_20261002/calibration-result-validation.json).
The worker metadata is under
`experiments/runs/amsgradw_stability_20261002/bootstrap/final-archive-worker.json`.

For an interrupted original four-recipe stage, the identical manifest can be
continued with the command below; completed budgets are checked and skipped.
It is not the resume command for a custom follow-up recipe:

```bash
.venv/bin/python -m experiments.synthetic_trainers.stability \
  --output experiments/runs/amsgradw_stability_20261002/calibration --resume
```

The archive/diagnostic and complete-comparison tools are now verified with 218
passing CPU tests, actual rendered PNG/PDF archives, deletion of original run
paths, and offline verification without PyTorch. The comparison builder requires
all four frozen complete runs, retains failures and missing timing support, and
rejects mixed plans. It does not imply independent confirmation. Use the per-run archive commands in
[STABILITY.md](STABILITY.md) immediately after each full-budget result.

The completed `short-lr001` unchanged-rate control is a negative result:
there is no required low-held-out memorization plateau, and 10 of its 201
final-tail observations fail the joint 99% criterion. The lowest tail held-out
accuracy is 58.14%. Final train/held-out accuracy is 100%/99.9785% (one incorrect
held-out equation); the rebound does not pass persistence. First long joint
confirmation is 52,750–57,500, costing 1,720.56 training / 1,808.46 wall seconds.
The complete budget costs 4,481.67 training / 4,708.09 wall seconds and peaks
at 119,166,464 allocated CUDA bytes. All 601 canonical observations, 1,200
neighbor probes, and 1,800 diagnostic records are archived with PNG/PDF curves.
See the [measured result](protocols/amsgradw_stability_20261002/short-lr001-result.md)
and [portable archive](baselines/amsgradw_stability_short_lr001_seed0_data0_20261002).

After legacy held-out confirmation at update 6,000, 64 canonical observations
fail the target. All fail numeric answers, none fails EOS, and 63 remain below
99% at both immediate neighbors. The remaining failure crosses down from 99%
before the canonical observation and stays below it afterward. No failure is
isolated to its canonical update or recovers at the next update. This weakens
a one-update observation artifact explanation without identifying a cause or
excluding a longer-lasting batch-size effect. Both lower-rate controls are now
complete and included in the comparison.

The completed `wrap-lr001` control also fails the prospective criterion: no
required memorization plateau and 26/201 final-tail target failures. Its tail
minimum is 1.9759% held-out accuracy at update 110,500; final train/held-out
accuracy rebounds to 100%/100%. First long confirmation is 24,250–29,000,
costing 854.82 training / 898.16 wall seconds. The full budget costs 4,530.51
training / 4,754.69 wall seconds, with the same 119,166,464-byte peak CUDA
allocation. All 601 canonical, 1,200 neighbor, and 1,800 diagnostic observations
are verified and archived with standalone PNG/PDF curves. It consumes 76,800,000
examples versus 69,840,000 for the original short-tail control at equal updates.
See the [measured result](protocols/amsgradw_stability_20261002/wrap-lr001-result.md)
and [portable archive](baselines/amsgradw_stability_wrap_lr001_seed0_data0_20261002).

After legacy confirmation at 24,500, the wrap control has 40 canonical held-out
target failures. All fail numeric answers, none fails EOS; 37 stay below target
at both immediate neighbors, two are isolated at the canonical observation,
and three recover by the next update. Categories can overlap. All batches
contain 512 examples, so post-confirmation instability also occurs with full
batches; its cause remains unresolved. An earlier long target streak in this
single paired calibration does not certify persistence or a repeatable speedup.
The full-batch control also fails stable grokking; no recipe was selected from
its earlier target streak.

The completed `short-lr0003` control fails the prospective criterion: no required
memorization plateau and 2/201 final-tail target failures, at 103,000 and 134,500.
Its tail minimum is 40.5498% held-out accuracy; final train/held-out accuracy
rebounds to 100%/100%. First long joint confirmation is 3,000–7,750, costing
240.46 training / 252.06 wall seconds. The full budget costs 4,560.76 training /
4,785.06 wall seconds, with the same 119,166,464-byte peak CUDA allocation.
All 601 canonical, 1,200 neighbor, and 1,800 diagnostic observations are
verified and archived with standalone PNG/PDF curves. It consumes 69,840,000
examples, matching the original short-tail control at equal updates.
See the [measured result](protocols/amsgradw_stability_20261002/short-lr0003-result.md)
and [portable archive](baselines/amsgradw_stability_short_lr0003_seed0_data0_20261002).

After legacy confirmation at 3,250, nine canonical held-out target failures
occur. All fail numeric answers, none fails EOS, and both immediate neighbors
also fail in every case. At update 103,000, train accuracy is 46.7998%; held-out
accuracy before, at, and after the update is 40.8935% / 40.5498% / 38.8316%.
The earlier long target streak and fewer sampled failures in this one paired
calibration do not certify persistence or a repeatable speedup.

The completed `short-lr0001` control has the required memorization plateau
at updates 1,000–61,250 (242 observations), but never reaches held-out 99% or
long joint confirmation. Its maximum canonical held-out accuracy is 67.1607%
at 149,750. All 201 final-tail observations fail; final train/held-out accuracy
regresses to 31.2715%/17.5258%. The immediate pre-final probe at 149,999 already
fails after a full 512-example batch, so the regression is not confined to the
last 48-example batch or final observation. No after-budget probe was measured;
the cause remains unresolved. Numeric answers fail; EOS stays above 99% in
the final tail. The full budget costs 4,523.19 training / 4,747.52 wall seconds,
with the same 119,166,464-byte peak CUDA allocation and complete 601/1,200/1,800
canonical/neighbor/diagnostic histories. Its empty post-confirmation plot means
no prior target confirmation, not an absence of failures. See the
[measured result](protocols/amsgradw_stability_20261002/short-lr0001-result.md) and
[portable archive](baselines/amsgradw_stability_short_lr0001_seed0_data0_20261002).
The [15,000-update prefix](protocols/amsgradw_stability_20261002/short-lr0001-prefix.json)
is retained as a historical partial observation, superseded by the full result.

The complete original grid is negative. The next controlled calibration changes
only the learning rate to 0.0002, between 0.0001 (memorization plateau without
99% generalization) and 0.0003 (early generalization without the plateau and
with late failures). This is a testable interpolation, not a monotonicity or
stability guarantee. Preserve the 50% split, seeds 0/0, decay 0.1, short-final
batching, 150,000-update budget, diagnostics, and unchanged phase/persistence
criterion. Its fresh manifest was frozen before training. Independent confirmation
and architecture/complementary comparisons remain pending.

A reproducible post hoc [tensor inspection](protocols/amsgradw_stability_20261002/collapse-tensor-inspection.md)
selects each completed recipe's worst final-tail held-out observation and the
latest earlier observation with train accuracy at least 99%. Tied embeddings
have the largest gradient at both selected fit and regression points, so their
dominance alone does not distinguish failure. The largest tensor update/weight
ratios at regressions are 4.94% / 18.24% / 1.61% / 0.86%, in attention
projection or QKV weights. Temperatures are similar within each selected pair;
the 250-update gaps leave unmeasured dynamics. These associations do not prove
cause or select an intervention. All eight rows reproduce exactly from the
verified complete comparison without importing PyTorch. The completed midpoint
plan and success criterion are unchanged.

The [midpoint protocol](protocols/amsgradw_stability_20261002/midpoint-protocol.md)
and [single-recipe plan](protocols/amsgradw_stability_20261002/midpoint-plan.json)
are now frozen at 2026-10-02 11:41:05 UTC. Only `learning_rate` differs from the
original short-final control. All 19 training/protocol source hashes, both
manuscripts, corpus fingerprints, environment, instrumentation and criterion
match the completed parent grid. The custom launcher's check-only validation
passed before launch. Its launch/resume command is:

```bash
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.amsgradw_stability_20261002.run_midpoint
```

The midpoint trainer (initial PID 98387, session 52847) and archive worker
(PID 98517, session 18237) both completed with exit code 0. No job remains in
this stage. The complete individual and single-recipe comparison archives
were verified without PyTorch, all three distinct PNG figures were visually
inspected, and the frozen 19 sources and both manuscripts still match.
See the [complete measured result](protocols/amsgradw_stability_20261002/midpoint-result.md)
and [verification metadata](protocols/amsgradw_stability_20261002/midpoint-result-validation.json).
The full run has no required memorization plateau and 150/201 tail failures;
all canonical observations from 112,750 through 150,000 fail. Final train /
held-out accuracy is 100% / 88.6598%. The full budget costs 4,318.45 training /
4,543.32 wall seconds, with complete 601/1,200/1,800 histories. It is ineligible
for independent confirmation. The historical launch validation and prefix
inspections below remain preserved partial evidence, superseded by this result.

Raw output is under
`experiments/runs/amsgradw_stability_20261002/calibration_midpoint_lr0002/`.
After instrumentation changes, preserve this completed manifest and use a
fresh plan for new work; strict resume requires the original source version.
No scientific independent confirmation or architecture run has started.

The midpoint's first canonical held-out 99% crossing occurs at 93,250:
train/held-out accuracy is 100%/99.0120% (46 incorrect held-out equations).
Its [recorded prefix](protocols/amsgradw_stability_20261002/midpoint-first-target.md)
retains all 374 canonical observations and unchanged plan/source/manuscript
fingerprints. No required low-held-out memorization plateau occurs before
that first crossing, so the recipe is ineligible under the frozen phase gate
regardless of later persistence. Long joint confirmation and final-tail
persistence were unmeasured at that historical prefix. The complete archive
now records long confirmation followed by a failed final tail.

The later [first tail collapse](protocols/amsgradw_stability_20261002/midpoint-first-tail-collapse.md)
is recorded at 112,750 after long joint confirmation over 95,750–100,500.
The first 51 final-tail observations pass; the next falls to train/held-out
1.2672%/0.7947%. Both immediate neighbors have the same low accuracies, with
EOS 100%. Scores are still high at the previous after-probe 112,501, but are
low at 112,749; the raw second-moment norm rises from 0.00257 to 23.78 between
them. This indicates large unprobed gradients under the frozen raw recurrence,
without identifying a trigger or causal direction. Complete source samples
and the calculation's assumptions are retained. The now-complete budget
confirms that final persistence fails; the complete outcome is preserved.

A read-only [fraction-control corpus inspection](protocols/amsgradw_stability_20261002/candidate-corpus-coverage.md)
finds all 97 answer classes and legal operand classes in the 50%, 20%, 10%,
and 5% pools on calibration data seed 0. Counts, oracle answers, and nested
split fingerprints are verified. This is finite data coverage, not a learning
result; no fraction recipe has been selected or launched. The next paired optimizer control
retains the 50% split. Retain the historical 20% reference control's full-budget
negative result.

The independent-confirmation driver and portable report are prepared and tested;
no scientific confirmation plan or run has started. See the commands in
[STABILITY.md](STABILITY.md) and
[verification metadata](protocols/amsgradw_stability_20261002/confirmation-driver-validation.json).
The driver rejects incomplete/negative calibration, excludes current and
historical calibration seeds, freezes all six crossed cases before training,
and retains failed targets. Tests include continuation after interruption,
immutable-plan/source checks, complete real CPU negative repeats, offline
archives and rendered curves. The small pipeline smoke uses an explicitly
hypothetical passing calibration fixture; it is not evidence of a learning
effect. All actual eight-update CPU repeats in that smoke fail, as retained.

Confirmation and architecture comparisons require separately frozen plans after
these complete histories. The [architecture comparison gates](STABILITY.md#architecture-comparison-gates)
retain the all-six stable-grokking requirement for the unchanged benchmark,
while allowing a candidate's rapid persistent generalization to be compared
without falsely labeling it grokking. The user has now selected sparsemax for
the next exploratory pair; no scientific architecture comparison has started.
The portable [architecture metric rules](architecture_metrics.py)
are now prepared: complete persistent candidates without a memorization plateau
retain their own label; failed pairs retain observed timings but cannot certify
speed ratios; task/data/seeds/optimizer/training changes are rejected. The full
CPU suite passed 230 tests before the final real-pair integration check; all
seven affected checks then passed. The current frozen training and analysis
source hashes are unchanged. This is pipeline preparation, not benchmark or
architecture success. See the [validation](protocols/adamw_stability_20261002/architecture-metrics-validation.json).
Keep the active goal running, retain failures, archive
each verified completed result, and commit logical changes during the loop.

## Proposed sequence, now being activated

1. Use AdamW as the primary optimizer following the user's explicit switch.
   Investigate instability in fresh paired AdamW/raw-AMSGradW calibration runs. Log gradient norms, parameter/moment norms, learned temperatures,
   numeric-answer/EOS losses, and batch sizes. Check sensitivity to the short
   last epoch batch (48 versus 512 examples) and observation cadence; these
   are hypotheses to test, not established causes.
2. The raw AMSGradW rate/sampling calibrations are complete and negative.
   Freeze a paired AdamW/raw-AMSGradW comparison at learning rate 0.001,
   holding decay 0.1, model, data, and full budgets fixed. Change one mechanism at a time. Any fraction/decay/schedule/model
   change is a separately labeled adaptation, with a fresh frozen plan.
3. Before confirmation, define both phase and persistence targets. Require a
   low held-out memorization plateau followed by sustained generalization;
   retain final failures and later collapses. Freeze confirmation seeds/splits
   independently of calibration. Repeat on at least three initializations
   and additional data splits, reporting per-seed outcomes and timing support.
4. Once the trainer is reproducible, evaluate a targeted GPTMini architecture
   improvement using paired data/seeds/budgets on the new GPU. Compare task
   quality at fixed budget and measured time to the predefined target, with
   parameter counts and memory. Keep an unchanged-model control.
5. Extend the comparison to complementary existing mechanics, including MQAR,
   copy/parity, lookup/composition, and prefix counting. Keep novel-ID and
   longer-input success separate; better ID fitting alone did not solve length
   transfer in the completed scaling study.

The follow-up goal is active. Calibration launch and measured progress are
recorded in the active bootstrap section; later stages await its results.

## Ready-to-send message for a new loop

Open a new session in the restored repository on the new machine, then send:

```text
Start a new loop. Read AGENTS.md and experiments/synthetic_trainers/HANDOFF.md,
then PLAN.md, STUDIES.md, and the recorded modular/random-feature protocols.

Create an active goal: establish a repeatable, stable grokking benchmark for
AdamW + softmax GPTMini (the user-selected primary optimizer), then evaluate a targeted
architecture improvement for better quality at the same budget or less
measured time to the predefined target.

First inspect the actual checkout, environment, archives, and process state.
The previous nine-run campaign is complete; use its full results, including
AMSGradW's failed final seed and later collapses, as the baseline. Restore
local paper/data/checkpoint resources needed for the next action and run
appropriate checks. Work directly in one session without subagents.

Investigate instability with controlled diagnostics, then calibrate one
mechanism at a time. Preserve the historical raw AMSGradW negatives and use
raw AMSGradW as a paired optimizer control. Freeze new data, seeds,
targets, observation cadence, full budgets, and source hashes before each
confirmation. Separate calibration from independent repetitions; retain all
failures and complete histories. Compare paired controls on this GPU.

Operationally separate train memorization, low-held-out plateau followed by
generalization, error double descent, and persistent final performance. Define
the stronger persistence criterion prospectively. Do not treat first target
crossing or finite curves as a proof of an internal algorithm or causality.
After reproducibility is established, test the architecture improvement on
complementary existing mechanics and separate novel-ID from length transfer.

Commit completed, verified logical changes regularly. Keep the loop active
until the declared experiments, independent confirmations, tests, complete
archives/curves, and honest comparison report are finished. If the intended
effect fails, preserve the negative result and continue a justified controlled
experiment rather than changing the success criterion retroactively.
Update HANDOFF.md with measured outcomes and the next concrete action.
```
