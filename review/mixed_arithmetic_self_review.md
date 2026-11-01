# Mixed-arithmetic and residual research self-review

The user explicitly requested that the author perform the review. The same
assistant, registered as `mixed_arithmetic`, reviewed tasks #138, #140, #141,
#142, #143, #144, #146 and #147 on 2026-11-01. This is **self-review**, not
independent review. No work was delegated. The submitted repository heads
were transformer `e95311ee` and diff_gpt `4e58930`.

Seven tasks passed their acceptance checks without a code change. Task #142
required the profile-exporter correction committed at `57935e11`, described
below. All eight tasks are now approved as self-reviewed. No new false Lean statement or
incorrect guarded-XSA equation was found. Numerical replay of the old model
remains imperfect; the review preserves those differences.

## Checks and scope

| Task | Review evidence | Result |
| --- | --- | --- |
| #138 source and early experiments | Independent pinned-source initialization, logits and every parameter gradient; all 48 saved endpoints; spectral and observation-graph checks | Source fixture and exact-check JSON reproduce identically; all 99% classifications agree, with additional compiled count drift retained |
| #140 local attention bounds | Full checks at `730c0155`, seven printed theorem axioms, statements beside LR1--LR4 | Pass; a local necessary mass/width condition, not a task or multilayer impossibility theorem |
| #141 long continuation | All10 complete source-to-initial states, final 100000-step checkpoints, curves and compiled follow-up | Pass; initial states bitwise equal, recorded cap 6283.38s < 7200s, all final success classifications agree |
| #142 profiles | Original raw timings/traces, six regression checks, fresh wide/deep sampling and CUDA traces | Pass after `57935e11`; historical missing rates explicit, fresh rates read from retained logs |
| #143 falsification ledger | Before-long ledger, frozen protocol, all 48 early and10 long outcomes, exact graph premises | Pass; learning-time interpretations corrected without weakening valid completion theorems |
| #144 residual audit | Local AttnRes/mHC manuscripts, fixed first 512 train/heldout rows for all 8 early/long pairs | All three diagnostic JSON reports reproduce identically; no residual-superiority conclusion |
| #146 normalized depth attention | Full checks at `1a9b522e`, eight printed axioms, source equations/gradients and 21 saved probes | Pass; probe report reproduces identically; a score bound is not a gradient lower bound |
| #147 guarded XSA | Full checks at `49e0e24f`, 17 printed axioms; actual-GPT six-variant audits of8 checkpoints; compiled numerics and six division checkpoints | Pass; allCPU reports reproduce identically; fresh compiled numerical errors match the original report |

Each historical Lean checkout used the unchanged root `lake-manifest.json`
and toolchain. Full builds, audit, regenerated-index comparison and forbidden
checks passed at all three theorem commits. All 32 named theorem axiom lists
contain only the standard axioms. The final audit retains 36 existing `sorry`,
zero results resting on them, zero extra axioms and zero vacuous/placeholder
statements. No unfinished statement was removed to obtain a pass.

The submitted Python and optimizer source/test trees are byte-identical
between `2bd11dbd` and `e95311ee`. The full `./run.py verify` at the latter
ran 446 lab tests with 13 skips and 104 optimizer tests with 14 skips. These
were CPU-visible executions; historical CUDA Muon allocation failures are
not erased. The affected GPT model, division and residual-probe suites at
`4e58930` passed 139 tests with one slow eager learning test skipped. The six
separate saved compiled division runs supply the reported learning regression;
this is not an assertion that the skipped eager test ran unchanged.

## Profile finding and correction

`report_profiles.py` returned 12.69 for the wide sampler error rate and 8.28
for the deep rate regardless of its inputs. Python 3.15's JSONL collector
retains 24454/49517 valid samples but does not serialize the error statistic.
The original console logs were not retained, so those percentages cannot be
verified from the archived raw files.

The exporter now parses the percentage from each retained sampling console
log, records its source and units, and emits `null` with an explicit missing
status when that evidence is absent. It rejects malformed or concatenated
statistics, incomplete JSONL and mismatched case/checkpoint/recipe records.
The update count comes from the actual profiler annotation. Six regression
checks cover changed rates, measured zero versus missing evidence, malformed
logs, incompatible profiles and incomplete streams. The documented sampling
command captures stdout and stderr for subsequent reports. After the fix,
`./run.py test` ran 452 lab tests (13 skipped) and 104 optimizer tests (14 skipped),
with no failures. The six new profile tests also pass separately.

Historical throughput, overhead, valid samples, kernel totals and raw hashes
are unchanged. The original profile commit `bc0c2433` predates the long
phase start at 13:07:26UTC. Any fresh review profiles are separate diagnostic
copies and cannot retroactively replace that chronology or quantify missing
historical sampler coverage. The original binary collector failure remains
retained. No optimizer, model or learning recipe changed in this correction.

The six fresh review profiles (wide/deep, plain/sampling/CUDA) completed after
the full suite finished. Plain throughput is 290.12/130.98 updates per second;
sampling overhead is 0.545%/0.572%. Attention contributes 51.13%/71.07% of GPU
kernel time. Retained logs report 12.90%/9.01% sampler errors, parsed by the
corrected exporter, with 24432/51202 valid JSONL samples. The new artifact
hashes and summary are included in the portable review data. These results
support the original bottleneck interpretation, without a speedup claim.

## Old-model replay differences

The early eager CUDA report reproduces the original measurements. Fresh
compiled follow-up has five discrepant splits:

| Case | Split | Fresh correct-count minus recorded count | Loss within unchanged tolerance |
| --- | --- | ---: | --- |
| Division P97 W128D4 | heldout | -3 | yes |
| Division P193 W128D4 | heldout | -3 | yes |
| Confirmation mixed P97 W128D4 | heldout | +1 | yes |
| Confirmation mixed P193 W128D4 | train | +2 | yes |
| Chain P17 length1 W128D4 | heldout | -1 | no |

The last loss difference remains 0.0073447227. Every 99% classification agrees.
The old report's two one-answer discrepancies describe that execution; they
are not a deterministic bound on future compiled replays. The new review
preserves both reports without replacing recorded training metrics or widening
tolerances. Long CPU replay reproduces every original metric, and the flagged
compiled long follow-up again matches answer counts and loss tolerances.

These observations are consistent with the already documented cancellation
sensitivity of the frozen old arithmetic. They do not establish one exclusive
cause of every kernel-level difference. The long trajectories were never
retrained or changed during review.

## Guarded-XSA boundaries

The source correspondence retains the clipped own-value remainder, arbitrary
normalized causal weights and RoPE, parameter-free RMS normalization and tied
readout. The real prefix equality is conditional on every relevant value head
being above clipping. Runtime sharing uses a strict `norm > 1e-12` test and
permanently disables sharing after any failed layer. Cached execution has
centering without prefix-state sharing; its backward stability is not claimed.

Three fresh CUDA max-autotune repetitions reproduce maximum logit error
5.42913e-7, relative parameter-gradient RMS error 8.32084e-7 and zero changed
predictions. All six division checkpoint reports, including complete optimizer
state and exact sampler draws, reproduce identically. The historical mean
400/430/360 is 396.667; the original protocol's rounded 400 remains explicitly
corrected in the existing report.

An additional adversarial two-layer check makes the first block's value head
clipped and the second block's head unclipped. A cyclic output projection
makes later prefix logits differ by 0.06923, exposing an incorrect reactivation
of sharing. The actual implementation agrees with the legacy FP64 source to
5.21e-17, and enabling gradient checkpointing preserves the checked outputs
and all parameter gradients bitwise. The first attempted identity projection
did not produce the required contrast after normalization; that failed test
input and its replacement are both retained.

The first historical-source replay wrapper also needed a correction: omitting
the old parent package initializers let Python import the live GPT package.
Exporting those exact historical files restored the intended source oracle;
the final fixture, initialization, logits and gradients are bitwise identical.
This was a review-harness import error, not a source-model discrepancy.

## Reproduction and remaining work

The portable [review data](mixed_arithmetic_self_review.json) contains every
early/long replay discrepancy, exact theorem commits, numerical results and
SHA-256 hashes of the root's ignored evidence directory:
`.kanban/artifacts/self-review-20261101/`. It retains raw logs, historical
source export, full replay JSON, failed harness attempts and scripts. Original
training artifacts remain in the experiment roots and the GPT root's `runs/`.

The review reran `./run.py verify`, the per-commit Lean checks and explicit
axiom scripts, `source_check.py` against exported `b98c665b`, `compile_check.py`,
`spectral.py`, `recovery_patterns.py`, both `analyze.py`/`replay_compiled.py`
pairs, both residual diagnostics and their comparison, `parameter_precision.py`,
the 21-case residual replay, and GPT's centered-XSA `validate.py numerics` and
`report.py`. Only disposable profiling copies perform new diagnostic updates.
The authorized two-hour long cohort is finished and was not restarted.

The next research tasks remain #139 (structured source construction),
#133/#134 (learning bridge and depth/width tradeoffs), #145 (full residual
ablation), #148 (paired deep numerical retraining with a separate budget),
and #149 (conditional repeated-token length collisions). Self-review does
not close these theory or learning gaps and is not independent replication.
