# What do new gradients add to Schedule-Free averaging?

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
narrow models, P17 and any failed hypotheses. No outcome has been inspected yet.

The proposed Lean extension splits the actual two-buffer update into a
homogeneous scalar linear map plus the normalized-gradient forcing. Without
gradients, query and base remain in their initial two-vector span even though
decay and averaging keep moving them. New gradients supply the forcing term.
Neither leaving that span nor the finite-network decomposition is assumed
to imply heldout generalization or typical random-start convergence.
