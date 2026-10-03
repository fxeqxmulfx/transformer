# Full paired architecture cohort preparation

The native normalizer/schedule adapter now has a complete six-pair serial
driver and portable campaign report. Scientific selection still requires the
actual all-six primary AdamW/softmax confirmation archive to pass. An
exploratory normalizer pair or successful CPU checks cannot substitute for it.
The current scientific softmax control is still running; no scientific
architecture cohort is selected, frozen or launched by this preparation.

The future first cohort uses model seeds 7/8/9 crossed with data seeds 4/5,
excluding the scientific calibration and first primary-confirmation IDs.
Both normalizers share each corpus, model seed, parameter shapes, optimizer,
selected native rate path and full budget. The adapter consumes no extra
initialization RNG. Softmax/sparsemax execution order alternates across pairs,
with every case frozen before training. Scientific budgets remain exactly
300,000 updates; all parameters, canonical and neighboring observations,
dense gradients, actual rates, costs, exposure and peak memory are retained.

The prospective default improvement rule requires all six pairs to be eligible
and either every final complete-RHS gain to reach one percentage point, or
every training **and** wall target-time ratio to reach 1.05. Controls require
stable grokking. Persistent candidates without a plateau retain their separate
phase label. Failed pairs have null eligible timings/ratios and remain in the
six-pair denominator. Reported mean/sample SD is descriptive, not an IID
confidence interval. This implemented rule is not a current scientific claim.

The driver checks the complete benchmark copy, all case plans, actual corpora,
environment, manuscripts, 70 production/driver and nineteen native training
hashes, nine Lean files and prior audit. It serializes execution with a
nonblocking lock, retains partial/completed failures and resumes native state.
Existing archives must match all four complete original measurement/log files
byte-for-byte. Complete reruns inspect existing cases without retraining.
Portable verification recomputes every outcome, CSV, prose and recovery series
from all twelve archives, with no Torch, raw run tree or checkpoint dependency.

The [actual CPU preparation](architecture-cohort-preparation/validation.json)
contains two separately frozen six-pair cohorts on mod-7, 50% data, width 128,
two layers/four heads, 412,296 parameters, batch four and one-example tails.
The constant path completes 12 × 20 updates; cosine-tail completes 12 × 40,
with its original CPU bounds 20–30. Total is 24 runs, 720 updates and 168
canonical observations. In the constant fixture only, inactive schedule tags
use bounds 10–20 so the native config fits its short budget; the applied
constant rate remains unchanged. Scientific bounds and the live pair are intact.
Both cohorts have zero eligible pair support and no architecture improvement.
All 24 native model/moment checkpoints independently reload and are finite,
with eleven states at the actual complete budgets, betas (0.9, 0.98), epsilon
1e-8, decay 0.1 and no maximum buffer. Audit steps are zero; CUDA remains
uninitialized. Every complete negative result, checkpoint hash and both full
benchmark copies are retained. CPU fixtures do not consume scientific cases or
select a candidate from empirical scientific outcomes.

Four targeted tests pass, including common initialization/RNG across all three
seeds, changed-plan/source rejection before state writes, native interruption
after two complete cases, exact archive preservation, all-case portable
verification after deleting the raw stage, rehashed missing/recovery failures
and archive/original gradient comparison. The entire current CPU suite passes
284 tests in 90.639 seconds, with no failures, errors or skips. Actual scientific
CLI guards reject both negative CPU benchmark references before manifest writes
and reject a CPU plan from the scientific launcher. See the
[complete source/test/review receipt](architecture-cohort-preparation-validation.json).
All four new PNGs and their actual PDFs are visually reviewed. Their titles
explicitly mark CPU fixtures; retained older reference figures are byte exact.
The active 43-source pair, prepared 55/both 62/66-source sets and Lean audit
remain unchanged. No Lean source or new full-tree audit is claimed.

Reproduce the full CPU preparation into fresh destinations:

```bash
env CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.prepare_architecture_cohort \
  --stage /tmp/architecture-cohort-raw --archive /tmp/architecture-cohort-archive
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_architecture_cohort \
  /tmp/architecture-cohort-archive
```

After the genuine all-six scientific benchmark has passed, been reviewed and
committed, the scientific commands are:

```bash
.venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.freeze_architecture_cohort \
  --benchmark BASELINE_CONFIRMATION_ARCHIVE --benchmark-validation REVIEW_RECEIPT \
  --output FRESH_RAW_STAGE --manifest FRESH_TRACKED_MANIFEST
# Commit and review the prospective manifest before the first scientific update.
.venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_architecture_cohort \
  --manifest FRESH_TRACKED_MANIFEST --check-only
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_architecture_cohort \
  --manifest FRESH_TRACKED_MANIFEST
```

The reviewed benchmark receipt must record six scientific runs, 300,000 updates
each, successful offline verification, reviewed PNG/PDF figures and the consumed
trainer terminal session. Both evidence and all code must be committed. Complete
case archives are available during execution for regular verified commits; after
all twelve cases, assemble/verify the combined report and review every new figure.
Complementary scientific tasks and separate novel-ID/length transfer still
follow the primary benchmark and architecture gates.
