# Matched modular-division MLP controls

Which part of the scalar MLP's difficulty comes from its input code, its
answer code, its loss or AdamW decay? These controls use the browser's exact
division equations, half-table split and continuing sampler, while changing
one of those choices at a time. They complement the
[Rotate90 controls](../rotate90_analytic/README.md); they do not isolate every
architectural difference between a plain MLP and Rotate90.

## Runs

Every model is a plain MLP with two biased SiLU hidden layers of width 195.
The output is an untied affine map. Scalar operands are normalized to
`2v/96 - 1`; one-hot operands are concatenated into 194 coordinates.

| Label prefix | Input coordinates | Output coordinates | Objective per equation | Parameters |
| --- | ---: | ---: | --- | ---: |
| `scalar-number-mse` | 2 | 1 | half squared error to `2q/96 - 1` | 39,001 |
| `onehot-number-mse` | 194 | 1 | half squared error to `2q/96 - 1` | 76,441 |
| `scalar-vector-mse` | 2 | 97 | half squared Euclidean error to `e_q` | 57,817 |
| `onehot-vector-mse` | 194 | 97 | half squared Euclidean error to `e_q` | 95,257 |
| `scalar-vector-ce` | 2 | 97 | full 97-class cross-entropy | 57,817 |
| `onehot-vector-ce` | 194 | 97 | full 97-class cross-entropy | 95,257 |

Each row has decay `wd0` or `wd1` and seeds `s11`, `s17`, `s23`: 36 runs.
The vector squared loss sums coordinates; it does not divide by 97. Both
squared objectives include a factor of one half, unlike the older
[whole-table regression](../mod97_scalar_mlp/README.md). Cross-entropy uses
raw logits and has no EOS term. Changing vector MSE to CE preserves the
entire model and its initial weights.

All runs use AdamW with learning rate 0.003, betas (0.9, 0.98), epsilon
1e-8, inclusive ten-update linear warmup, batch size 512 and 30,000
updates. Decay applies to every parameter, including biases. There are
4,656 training and 4,656 held-out equations; one continuing source epoch
has nine full batches and a final batch of 48. All training budgets are
3,000 such epochs. Evaluation and gradient/update observations occur every
100 updates, or ten epochs, and checkpoints every 1,000 updates. The
coarser evaluation cadence must be retained when reporting first crossings.
The Eager CUDA backend is the locked lab backend.

`TorchDefault` initializes each network. For a fixed input code and seed,
the scalar and vector heads share every hidden parameter draw. Both vector
objectives also share their output parameter draws. Input widths change
PyTorch's fan-in initialization and the parameter count; these are fixed
hidden-width controls, not matched-capacity or matched-initial-kernel
controls. Observations include first-layer preactivation/activation RMS on
the fixed first 512 training equations to expose that conditioning change.
The Rotate90 comparisons additionally differ in tied weights, residual
connections and normalization, so an MLP/classifier comparison alone cannot
attribute the outcome to any one of those mechanisms.

## Decoding and expressivity

Scalar answers are decoded by rounding `(f + 1) * 48`, without clipping to
the vocabulary. Vector answers use argmax, with PyTorch's smallest-index
tie policy. Every run reports train and held-out accuracy, objective loss
and decoded numeric MSE in residue-value units. Scalar runs additionally
retain continuous numeric MSE before rounding. The losses have different
scales and must not be compared as if they were the same accuracy measure.

An independent multiplication oracle checks every data row. Tests check
the original rows, splits and 21 consecutive batches against `LabDivision`
for all three seeds, loss and decoding formulas on a small exact example,
every parameter's float64 finite-difference gradient, shared initialization,
and the actual engine/checkpoint path for all three objectives. A finite
two-layer SiLU scalar lookup construction at width 195 answers all 9,312
equations, with raw numeric MSE below 1e-8. That construction only certifies
expressivity in the test; training always starts from random parameters
and never receives held-out answers or constructed teacher weights.

## Resumed cohort

Active again at the user's request on 2026-10-19, board task #74. This
workspace contains the portable partial snapshot below but none of its raw
ignored run directories. The currently available GPU is an NVIDIA GeForce
GTX 1050 with 2 GiB VRAM, driver 580.178.04. The locked environment reports
Python 3.14.8 and PyTorch 2.14.1+cu126 with CUDA 12.6 and working CUDA access.
The earlier RTX 3050 measurements describe a different environment.

All 36 cases are therefore being run afresh, from their original random
initializations, on the current device and runtime. No saved result from the
earlier cohort is used to fill a missing case. The grid, fingerprints, data,
sampler and budgets are unchanged. `progress.json` remains the historical
partial cohort; complete new results require every original run and the
analyzer's source, split, initialization and final-checkpoint replay checks.
Raw new runs are retained at the repository root and linked into the isolated
worktree so that removing that worktree cannot delete the trajectories.

## Jacobian access diagnostic

Board task #75 adds [access.py](access.py) for all twelve scalar-output
controls. At the original random initialization and final checkpoint it
computes every prediction derivative with respect to every parameter on
the complete table. The calculation uses four CPU threads and float64,
starting from the exact float32 parameter values used in training. It never
updates a model, optimizer or sampler.

Let `P` subtract row and column means and restore the overall mean on the
97-by-96 equation table. This is the orthogonal projection away from
additive operand effects. Every equation remains in the calculation;
zero numerators are included. The diagnostic reports the raw and projected
Jacobian energies, and separate contractions for the projected numeric
target and actual prediction residual. With `N = 9312` and
`K = (P J)(P J)^T / N`, the target Rayleigh quotient is
`||J^T P y||^2 / (N ||P y||^2)`. Its ratio to `trace(K)` removes a common
Jacobian scale. Per-parameter-tensor contractions identify which layers
supply that local response. Independent per-example autograd and parameter
finite differences check the batched Jacobian and the projected response.

The same derivative also defines a frozen linear prediction model on the
first 512 training and first 512 held-out equations of each original source
permutation. It follows full-batch Euclidean GD exactly by kernel powers,
using step `1 / lambda_max` of the 512-row training kernel, at 0 and powers
of ten through `10^12` updates. Eigenvalues at most `10^-12 * lambda_max`
are treated as zero; the full measured spectrum and the retained numerical
rank are exported. That cutoff is a numerical convention, not a proved rank
or an impossibility statement about the discarded directions. An independent
linear example with a genuine kernel nullspace checks these predictions
against explicit parameter updates. The subset and step rule differ from
the actual 4,656-row minibatch AdamW training protocol.

These are static tangent measurements, including an offline use of held-out
labels. They are not training interventions, bounds on future feature
learning, or an identification of AdamW with Euclidean gradient descent.
The full-table projection is not the training objective. A weak initial
alignment alone would not explain the optimizer's eventual outcome; compare
the initial and final measurements and the actual fitting/generalization
curves before choosing the next Landscape statement.
Long frozen-model trajectories do not certify that large parameter
displacements stay within a valid Taylor approximation of the real network.
This is exploratory mechanism analysis, not a preregistered confirmation
test. No complete access-results artifact is claimed before all twelve
paired checkpoints are available and the diagnostic finishes.

## Access findings on the fresh scalar-output cohort

All twelve scalar-output controls completed their original budgets. The
[portable measurements](access_results.json) contain every initial and final
Jacobian contraction, per-tensor contribution, frozen spectrum and trajectory,
with source and checkpoint hashes. The vector-output comparison is still
incomplete at this milestone.

The full-table target Rayleigh quotient below measures absolute local
sensitivity. The numerical ranks refer to the separate 512-row training
kernel, at the stated relative cutoff; they are not full-table rank claims.
Each cell shows initialization / final checkpoint.

| Inputs | Decay | Seed | Retained subset rank | Full-table target Rayleigh quotient |
| --- | ---: | ---: | ---: | ---: |
| scalar | 0 | 11 | 30 / 171 | 0.000188 / 0.0154 |
| scalar | 0 | 17 | 32 / 152 | 0.000243 / 0.00908 |
| scalar | 0 | 23 | 30 / 206 | 0.000198 / 0.0182 |
| scalar | 1 | 11 | 30 / 15 | 0.000188 / 0.000635 |
| scalar | 1 | 17 | 32 / 16 | 0.000243 / 0.000482 |
| scalar | 1 | 23 | 30 / 18 | 0.000198 / 0.00057 |
| onehot | 0 | 11 | 512 / 512 | 5.43e-09 / 0.713 |
| onehot | 0 | 17 | 512 / 512 | 6.22e-09 / 0.632 |
| onehot | 0 | 23 | 512 / 512 | 5.33e-09 / 0.717 |
| onehot | 1 | 11 | 512 / 512 | 5.43e-09 / 0.00944 |
| onehot | 1 | 17 | 512 / 512 | 6.22e-09 / 0.00821 |
| onehot | 1 | 23 | 512 / 512 | 5.33e-09 / 0.00793 |

At initialization the scalar-input target Rayleigh quotient is much larger
than the one-hot value. Thus a claim that one-hot inputs start with better
access according to this scalar statistic is contradicted by these three
paired starts. During training without decay, the one-hot quotient grows
from about `5e-9` to `0.63–0.72`; its trace-normalized alignment rises from
about `0.00011` to `0.083`. Scalar inputs increase their absolute sensitivity,
but the normalized alignment falls. Both the absolute scale and the direction
alignment matter, and an initialization-only statistic misses feature changes.

Frozen initial scalar kernels retain only 30–32 numerical directions out
of the 512 sampled training equations. Even the `10^12`-update frozen
control leaves continuous training numeric MSE 668–723. The initial one-hot
kernels retain all 512 directions and report zero training error at that
long horizon, but held-out numeric MSE is 1,453–1,510. The real scalar-output
runs also remain near chance on held-out classification (0.86–1.70%).
Interpolation of these sampled equations therefore does not supply the
division rule in this frozen model.

These measurements distinguish local numerical conditioning, access through
feature changes, and behavior outside the fitted sample. They do not prove
an exact rank restriction, a whole-grid frozen interpolation result, or the
optimizer's attraction to a generalizing Rotate90 solution. Decay also changes
the final geometry, so its effect cannot be inferred from the initial kernel.
The [Lean control and source review](../../review/mod97_tangent.md) formalizes
the quadratic GD recurrence and a finite-budget eigenmode floor, with an
attainable zero-loss witness. It supplies no AdamW or changing-feature bridge.

The complete diagnostic and independent file audit verify all twelve
checkpoint hashes and the analysis-source hash. All four new `MLPAccess`
tests pass. The existing full Python gate checks 339 tests (three skipped,
all others passing). Full Lean build, global audit, generated index and
forbidden checks pass; the thirteen tangent theorems introduce no `sorry`
or extra axioms. The global 39 existing unfinished statements are retained.

## Earlier partial cohort

Stopped at the user's explicit request on 2026-10-19. Eleven of 36 runs
completed their 30,000-update budget: all six scalar-input number-output
controls, all three one-hot number-output controls without decay, and the
one-hot number-output controls with decay at seeds 11 and 17. The remaining
one-hot number-output run, `onehot-number-mse-wd1-s23`, has a valid checkpoint
at update 25,000 and observations through 25,300. All 24 vector-output
controls have not started. This is an incomplete comparison.

[progress.json](progress.json) preserves every run's status, completed
results, the partial observation and raw-file hashes. The recorded lab
sources are identified by [engine_code.json](engine_code.json). At that stop,
raw ignored run directories and their model/optimizer/sampler checkpoints
were retained at the repository root. They are absent from this resumed
workspace; the new trajectories at `runs/` belong to the fresh cohort above.
When a checkpoint is available, the normal run command resumes from it and
rewinds later observations, rather than resuming at an unsaved observation.

The new [analyzer](analyze.py) is committed and syntax-checked; its full-grid
audit remains unexecuted because the grid is incomplete. Once all 36 runs
are complete with the recorded lab code and runtime, run it from `python/`
with `uv run --locked python ../experiments/mod97_representation_controls/analyze.py`.
It requires every run and replays checkpoints before writing final curves
and results; no final-results artifact is claimed here.

The full Python suite previously passed 339 tests in 804.625 seconds, with
three archived GPU comparisons skipped. At the stop, the repeated
`./run.py verify` had completed its Lean stages; its Python phase was
interrupted. A failure within this finite training budget would distinguish
observed fitting from generalization and would not prove that gradient
training can never reach a good solution. Quantitative gradient-access
and attraction claims remain the next Landscape obligation.

```sh
./run.py check experiments/mod97_representation_controls
./run.py show experiments/mod97_representation_controls scalar-number-mse-wd0-s11
./run.py run experiments/mod97_representation_controls
./run.py report experiments/mod97_representation_controls
```

Protocol source: `/home/misha/rust_projects/grokking_abmod` at
`f49969393a7482490d45483d8a4fd3046b1adede`, ported into `LabDivision` at
`c59d4426`. The scalar MLP and finite lookup construction extend the lab's
regression model at `897e80be`; the readout and loss interventions are
specified by M9.3 of [the research task](../../landscape_grokking_plan.md).
