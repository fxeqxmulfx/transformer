# What do new gradients add to Schedule-Free averaging?

All six wide P97/P193 continuations exceed 99% heldout accuracy with weight
decay removed after update 400. Removing gradients while retaining decay and
averaging leaves them at 2.95--5.15% after the same 1100 additional updates.
The new Lean result separates these effects exactly in the optimizer's two
buffers. It does not yet explain why the gradients learn the modular rule.

The memory study stopped both gradients and query decay when it set the rate
to zero. This study separates them while preserving the same complete source
state and next minibatch. It uses all fifteen transfer-study checkpoints at
update 400: P97 K2 and P193 K3, widths 16/64; P17 K1 width 64; seeds 31/37/41.
Data and batch seeds remain 11. All source hashes and predictions are fixed
in `hypotheses.json` before the continuations run.

| Arm | New gradients | Query decay | State and duration |
| --- | --- | --- | --- |
| `full-*` | Actual sampled loss | 1 | Retain source state; updates 401--1500 |
| `no-decay-*` | Actual sampled loss | 0 | Same source and samples; updates 401--1500 |
| Zero-gradient diagnostic | Zero tensors | 1 | Exact source scalar recurrence; fixed horizons through 1100 extra updates |
| Zero-rate diagnostic | Irrelevant to parameters | No base movement | Prior proved closed form at 1100 extra updates |

Both training arms use only Schedule-Free AdamW, lr .003, beta (.9, .98),
the original warmup and batch size 512. `ScheduleFreeContinuation` retains
query, base, raw second moments, averaging clock and sampler, and overrides
only rate/decay. Update budgets are absolute. Initial compilation and the
additional source observation preserve the pinned query exactly. Subsequent
observations use the unchanged engine. This is a continuation, not a restart.

`./run.py check experiments/rotate90_forcing` checks the 30 recipes;
`./run.py run experiments/rotate90_forcing` trains them with Python 3.15,
Torch 2.15 and CUDA max-autotune/fullgraph/static shapes. Source checkpoint
files remain immutable. `analyze.py` checks independent source identities,
evaluates the diagnostic paths and reports every declared result, including
narrow models, P17 and failed hypotheses. Setup commit `9f769888` predates
every continuation and diagnostic outcome below.

The proved Lean extension splits the actual two-buffer update into a
homogeneous scalar linear map plus the normalized-gradient forcing. Without
gradients, query and base remain in their initial two-vector span even though
decay and averaging keep moving them. New gradients supply the forcing term.
Neither leaving that span nor the finite-network decomposition is assumed
to imply heldout generalization or typical random-start convergence.

## Results

All accuracies below are percentages on the unchanged heldout split. The
first three result columns use absolute update 1500. The last column is the
best zero-gradient accuracy among the ten declared diagnostic horizons;
parentheses give additional updates after the source, not a new stopping rule.

| Source | Full | No decay after 400 | Zero gradient, decay retained | Best zero-gradient diagnostic |
| --- | ---: | ---: | ---: | ---: |
| p97k2-w64-s31 | 99.87 | 99.85 | 4.17 | 92.23 (100) |
| p97k2-w64-s37 | 99.96 | 99.87 | 3.61 | 90.68 (100) |
| p97k2-w64-s41 | 99.87 | 99.87 | 3.99 | 94.63 (100) |
| p193k3-w64-s31 | 99.97 | 99.97 | 2.95 | 48.50 (100) |
| p193k3-w64-s37 | 99.93 | 99.97 | 5.15 | 61.09 (100) |
| p193k3-w64-s41 | 99.88 | 99.95 | 4.11 | 75.33 (100) |

The six width-16 full continuations finish at 1.96--17.93%; their no-decay
counterparts finish at 2.00--21.54%. P17 full controls finish at 8.09--9.56%,
and no-decay controls at 7.35--9.56%. Every seed and all 690 recorded training
observations, 150 zero-gradient points and 15 zero-rate endpoints are in
[`results.json`](results.json). All thirty final states and their 50-update
checkpoint histories remain in `runs/`.

| Frozen hypothesis | Outcome |
| --- | --- |
| H1: reproduce all fifteen original full endpoints | **Fails:** 12/15 pass. All three P17 controls exceed the declared CE tolerance; seed 37 also exceeds the accuracy tolerance. |
| H2: independent source response and zero-gradient span | Passes all fifteen sources. Maximum coordinate error over 32 actual updates is 1.11e-15; maximum relative span residual is 2.00e-14. Saved raw moments change under zero gradients. |
| H3: wide full success versus zero-gradient failure | Passes all six. Even the best of the fixed zero-gradient horizons stays below 99%. |
| H4: wide success without subsequent decay | Passes all six. Their source states already include 400 updates with decay. |
| H5: new parameter directions, including failed controls | Passes all fifteen: relative distance from the initial query/base span is 0.395--0.764. Escape does not separate success from failure. |

H1's P17 CE differences are 0.34223, 0.01828 and 0.14770 for seeds 31, 37
and 41; accuracy differences are 0, 0.7353 and 0 percentage points. The source
query/base/moments/history/sampler are exact in every initial snapshot, and
both continuation arms consume identical samples. A post-outcome check of
original versus full-continuation snapshots at updates 400, 450, 750 and 1500
finds identical samplers and scalar optimizer state in all 60 comparisons.
P17 query discrepancies grow from relative norm 4.36e-8--7.40e-8 at update
450 to 0.466--0.509 at update 1500. This documents trajectory divergence;
it does not establish its cause or turn the failed prediction into a pass.
The checkpoint hashes and tensor errors are in
[`replay_diagnostics.json`](replay_diagnostics.json).

One initial attempt, `full-p193k3-w16-s31`, failed during CUDA graph capture
after its update-400 observation. An unchanged retry from its saved checkpoint
completed update 1500 and passed H1. The failure, retry and full log hashes
are retained in [`training_attempts.json`](training_attempts.json); the logs
remain in `runs/`. It is an infrastructure failure, not evidence of learning
failure. No recipe, scientific threshold or diagnostic horizon was changed.

## What the theory now explains

[`Forcing/Actual.lean`](../../src/Transformer/Rotate90/Forcing/Actual.lean)
connects the actual finite model's sampled-loss gradient and nonlinear logits
to the exact optimizer response. The new theory proves that decay and
averaging alone stay in the initial two-vector parameter span, and that each
new normalized-gradient input contributes through a known sequence of scalar
matrices. The gradient-free score family remains nonlinear; this is not a
rank-two bound on its score matrix. Adam's changing coordinatewise
normalization is included, so a claim about raw gradients would be different.

The experiments distinguish ongoing gradient information from continued
decay in these already trained states. They do not show that decay is
unnecessary from random initialization, or that any new direction is useful.
The remaining learning gap is to derive task-aligned normalized gradients
and their useful accumulation from the sampled training data. Flat spectrum
alone, and the new exact recurrence alone, do not provide that guarantee.

Validation at `9f769888`: all 27 new Lean theorems build and use only the
standard allowed axioms; the full audit retains 36 existing unfinished
statements, zero dependencies on them and zero extra axioms. The Python
suite completes 522 tests with 27 skipped. All 30 recipes pass their definition
check, and an independent compiled CUDA continuation smoke test reproduces
its endpoint exactly. Scientific runs use Python 3.15.2, Torch 2.15.0+cu132
and the RTX 3050 Laptop GPU. Independent recurrence checks and long
zero-gradient reconstructions use CPU float64; training and diagnostic network
evaluations use CUDA float32 with max-autotune. The Lean identities concern
real arithmetic and do not assert bitwise equality of long GPU trajectories.
