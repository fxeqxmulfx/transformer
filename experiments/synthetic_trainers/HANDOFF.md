# Synthetic trainer and paper reproduction handoff

State recorded on 2026-10-02. Run commands from the repository root.
Read [AGENTS.md](../../AGENTS.md), [PLAN.md](PLAN.md), and
[STUDIES.md](STUDIES.md) before continuing.

## Current state

All scheduled training in the reproduction campaign has finished. The serial
driver completed with exit code 0; there is no training job to resume.
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

The requested optimizer is the repository's raw AMSGradW, not bias-corrected
PyTorch AdamW with `amsgrad=True`: betas (0.9, 0.999), a maximum second moment,
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
[INDEX.md](../../INDEX.md) records 1,762 modules, 7,090 theorems, and 158
`sorry` globally; RASP, RASP-L, and C-RASP rows each record zero `sorry`.
This index is not a new full-tree axiom audit. Follow the Lean build, paper
fidelity, external-dependency audit, and index requirements for future proofs.
[AGENTS.md](../../AGENTS.md) now requires regular commits of verified logical
changes and forbids subagents. No new approval is needed for already authorized
reversible work or regular commits.

## Active follow-up bootstrap (2026-10-02)

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
were rechecked before launch. The serial driver is running under process 87620
(initial launch PID); inspect the actual process state before any restart.
The first `short-lr001` run has completed and its full portable archive is
verified. The driver has continued to `wrap-lr001`; one of four calibration
budgets is complete. Inspect the live history for its current update.

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

After a checkpointed interruption, continue the identical manifest with:

```bash
.venv/bin/python -m experiments.synthetic_trainers.stability \
  --output experiments/runs/amsgradw_stability_20261002/calibration --resume
```

The archive/diagnostic and complete-comparison tools are now verified with 215
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
excluding a longer-lasting batch-size effect. The full-batch control is now
running; both lower-rate controls remain queued. Finish all three complete
budgets before selecting or rejecting a calibration recipe.

Confirmation and architecture comparisons require separately frozen plans after
these complete histories. Keep the active goal running, retain failures, archive
each verified completed result, and commit logical changes during the loop.

## Proposed sequence, now being activated

1. Investigate late raw-AMSGradW collapses in fresh instrumented calibration
   runs. Log gradient norms, parameter/moment norms, learned temperatures,
   numeric-answer/EOS losses, and batch sizes. Check sensitivity to the short
   last epoch batch (48 versus 512 examples) and observation cadence; these
   are hypotheses to test, not established causes.
2. Calibrate raw AMSGradW stability with learning rates 0.0003 and 0.0001
   against 0.001, initially holding decay 0.1, model, data, and full budgets
   fixed. Change one mechanism at a time. Any fraction/decay/schedule/model
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
the repository's raw AMSGradW + softmax GPTMini, then evaluate a targeted
architecture improvement for better quality at the same budget or less
measured time to the predefined target.

First inspect the actual checkout, environment, archives, and process state.
The previous nine-run campaign is complete; use its full results, including
AMSGradW's failed final seed and later collapses, as the baseline. Restore
local paper/data/checkpoint resources needed for the next action and run
appropriate checks. Work directly in one session without subagents.

Investigate instability with controlled diagnostics, then calibrate one
mechanism at a time. Preserve raw AMSGradW semantics unless an optimizer
change is explicitly labeled as a separate control. Freeze new data, seeds,
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
