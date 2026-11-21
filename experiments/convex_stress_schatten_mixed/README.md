# E6: the Schatten coupled-moment learner on label shifts of multiplication

The main transformer of DiffGPT fits the training half of `a³ + ab` mod 97 and stays at chance
on the held-out half, at depth 1 and at depth 4 (E1,
`~/python_projects/gpt/research/depth_visibility/README.txt`). Is the barrier the task, the
information in the half split, or the model? `convex_stress_schatten` recovers multiplication
from the same kind of split. Every task below is multiplication with a per-cell class shift,
`f(a, b) = ab + u(a) + v(b)`:

| Task | `f` | `u(a)` | `v(b)` |
| --- | --- | --- | --- |
| `arithmetic-mul` (reference) | `ab` | `0` | `0` |
| `arithmetic-affine-product` (control) | `a + ab` | `a` | `0` |
| `arithmetic-cubic-product` | `a³ + ab` | `a³` | `0` |
| `arithmetic-square-product` | `a² + ab` | `a²` | `0` |
| `arithmetic-quadratic` | `a² + ab + b²` | `a²` | `b²` |
| `arithmetic-quadratic-linear` | `a² + ab + b² + a` | `a² + a` | `b²` |

CM121 in `review/convex/moments.md`: the shift `c ↦ c + u(a) + v(b)` of each cell's classes
maps the product of simplexes onto itself and sends each moment `M_f` to
`diag(ψ(f u)) M_f diag(ψ(f v))`, so it preserves every operator and nuclear budget, and it maps
observed CE to observed CE with shifted labels. The uniform start and the uniform mixture are
fixed by it, and each splitting step commutes with it. The split does not read labels: rows in
`(a, b)` order, one SplitMix64(1) shuffle, the first half for training. So the six tables share
their training cells and their order, and the minibatch order, `Random(seed + 10000)` over
training indices, is the same too. In exact arithmetic the learner's run on `f` is its run on
multiplication with each cell's probability vector rotated by `u(a) + v(b)`.

## Preregistered prediction (2026-11-21, before the first cell)

The learner of `convex_stress_schatten`, unchanged, on the six tasks:

1. The training-only hull selection picks `full_rank` for every task and arm.
2. Every cell (task, arm, budget) stops at the update and for the reason of the
   `arithmetic-mul` cell with the same arm and budget.
3. The selected models' held-out exact accuracy is multiplication's, for every task:
   100 / 99.9575 / 100 percent for full batch / minibatch / forward only, as in
   `convex_stress_schatten`'s pilot.
4. `compare`: in every cell, the largest `|P_f[a, b, c + u(a) + v(b)] − P_mul[a, b, c]|` over
   all cells and classes is below `1e-6`, and no cell's argmax moves.

A mismatch in any of these means the implementation breaks the symmetry; find where before
reading anything else into the result. If the prediction holds, E1's barrier is main's
representation, not the target's spectrum or the information in the half split: for this
learner the half split determines `a³ + ab` exactly as it determines `ab`.

## What changed, and what did not

`src/main.rs` is `convex_stress_schatten/src/main.rs` at `09fc84dd` with the task source
replaced. `selected` reads the 54 original cases of `convex_stress_arithmetic_data`, whose
`dataset::build` checks both historical fingerprints of each split; `task` writes the rows as
the 35-task suite's arithmetic examples, and the data fingerprints recorded are the historical
ones. Six lines of the copied `run`, `replay` and `hull_selection` differ from the original,
all of them the task source. Model, budgets, solver, arms, caps, cadence, stopping and selection
are the original code, compiled from `convex_stress_schatten`'s library. `compare` is new: the
shift `s = f − ab` per cell from the two label tables, a check that it is `u(a) + v(b)`, and the
deviations of item 4.

Checks before the first cell:

- `bridged_tasks_equal_the_suite_tasks`: for multiplication, affine-product and division, the
  bridged splits equal the suite's example by example.
- `planned_tasks_are_label_shifts_of_mul_on_the_same_cells`: every planned task has
  multiplication's cells in both splits, in the same order, and labels shifted by
  `u(a) + v(b)`.
- The bridged multiplication cell (full batch, full rank) equals the unchanged
  `convex-stress-schatten` binary's run bit for bit: model SHA-256 `18b63663…`, diagnostic
  state `317b341a…`, stop at update 19, 4705 of 4705 held out.

The unchanged binary's fingerprint at `09fc84dd` is `99280903…`, not the pilot's
`913357a9…` of `deef98cb`: since then a doc comment of `convex_stress_moments` and the
generation metrics of `convex_stress_minibatch` changed, and arithmetic tasks never reach
those metrics.

## Reproduce

Rust with system FFTW3 and OpenBLAS/LAPACK; no Python. Outputs stay in the ignored `runs/`.

```sh
OPENBLAS_NUM_THREADS=1 cargo test --offline --release --manifest-path experiments/convex_stress_schatten_mixed/Cargo.toml
cargo build --offline --release --manifest-path experiments/convex_stress_schatten_mixed/Cargo.toml
bash experiments/convex_stress_schatten_mixed/run_e6.sh experiments/convex_stress_schatten_mixed/runs/e6-v1
```

The run writes 36 cells with their replays, `summary.json`, `hull.json` and its replay, and
`compare.json`.
