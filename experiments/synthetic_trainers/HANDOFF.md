# Synthetic trainer and paper reproduction handoff

State recorded on 2026-10-02. Run commands from the repository root.
Read [AGENTS.md](../../AGENTS.md), [PLAN.md](PLAN.md), and
[STUDIES.md](STUDIES.md) before continuing.

## Current state

All scheduled training in the reproduction campaign has finished. The serial
driver completed with exit code 0; there is no training job to resume.
The current reproduction campaign is finished. The startup message below
describes a separate proposed follow-up loop, which has not been started.
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

## What I would do next

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

These are proposed follow-up experiments; they have not been launched.

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
