# Partition the shared boundary map

This training-only revision of task335 asks whether a small number of input
regions can represent the discontinuities missed by one global affine map.
The preceding frozen component fits only4494/4704 primary training labels in
forward mode, and its entire3072-configuration bank cannot reach99% conditional
accuracy. Its complete original and independent runs remain immutable. No
original heldout evaluation is allowed here; that is task336.

Lean first at `47ba037f`: `scripts/PartitionedBoundaryAxioms.lean` instantiates
CM55 with fixed descriptor gates and shared parameters in multiple blocks per
leaf. It checks complete coefficient costs, normalized complex features and
ordinary convex full/minibatch CE. The native partition is learned before CE;
this is not a proof that tree discovery is convex or that unseen labels are
identified. The finite witness is not a floating-point refinement theorem.

| Choice | Declared alternatives |
| --- | --- |
| Fixed input descriptors | Whole-grid/nonmodal-box axis or diagonal extrema, as335 |
| Constructor | Uncentered global moments; shared interval regression; interval regression trees with maximum depth1 or2 |
| Ridge / orientation | `[1e-6,1e-4,0.01]` / the same16 generic directions |
| Gain / common offset | `[8,32,128,512]` / `[8,32]` |
| Ordinary CE | Full128 steps or mini8 epochs of127, step0.1; forward0 |

At each tree node, training interval/centroid boundaries supply continuous
regression targets; raw class IDs remain categorical. Every feature threshold
between adjacent distinct fitting-block descriptors is considered. The chosen
split minimizes the two child residual sums after ridge fitting, with fixed
first-choice tie breaking. Both sides must contain a fitting block, and the
split must improve squared error by more than1e-12. Keep every trial and refusal.
Depth2 permits four leaves; a rare leaf can have just one observed binary block.
All nodes and leaf assignments are fixed during the convex CE stage.

Each block's boundary is the shared linear function of its descriptors in its
leaf. Global complex slope coefficients are still shared across all blocks.
The complete8-state Fourier head and all raw-class columns remain charged,
including empty blocks and absent classes. Storage uses at most52 parameters
per role. Eta costs include all block repetitions in that leaf, while the
fixed common offset is charged separately. Use the exact weighted Euclidean
L1 projection, with complete per-class radius1048576. The feature squared norm
bound remains10 because exactly one leaf's descriptor coordinates are active.
Step0.1 is declared before production; actual losses and final iterates remain
visible even on regression.

All6144 configurations use three cell-exclusion folds, retaining2304 states
and18432 gain/common scores. Minimum pooled ordinary CE chooses one refit.
The inherited route was learned on all original training records, so these
are conditional head diagnostics, not unbiased whole-model validation.
The331/332 results, failed335 component and local329 controls remain intact.
A separate frozen comparison must measure any original heldout improvement.

Candidate scoring sums the identical absent-class logits by multiplicity.
This is still ordinary full-vocabulary CE; gradients use the full class vector.
The final state checks both calculations for every original training record.
Because floating-point summation changes, new scores are not claimed byte
identical to335's full-loop scores. Its complete old evidence is retained.

Seven native checks pass before production: complete expanded products and
budgets, input-only geometry, genuine gradients and short batch tails,
categorical renaming and exact replay, full/mini final states, a synthetic
piecewise boundary with unseen inputs, and compact/full CE including absent
classes. Original dimensions are forecast before fitting. The largest new
one-inspection-plus-replay forecast is10,382,354,111,104 scalar units and
13,422,565,376 buffer bytes, before the separate parent-parse allowance.
The12T arithmetic cap includes all attempted splits; explicit buffers stay
below56GiB and processes below64GiB. Serialization, hashes and disk I/O are
measured separately.

Production at `2d5ac6ce` and the independent review both finish all six
inspections. All56 JSON artifacts match exactly. Every mode and role classifies
all4704 original training records correctly. Every selection chooses the same
depth1 tree, Grid frame, Diagonals descriptors, ridge1e-6, direction1, gain512
and common offset8. This uses two leaves and34 stored parameters per role.

| Mode | Primary training CE | Supplement training CE | Effective updates per role |
| --- | ---: | ---: | ---: |
| Forward | 0.006883672052 | 0.006333256670 | 0 |
| Full | 0.006839029108 | 0.006277233057 | 128 |
| Mini | 0.006796859848 | 0.006224551209 | 304 |

The selected conditional folds reach4686/4704 (99.6173%) for the primary role
and4676/4704 (99.4048%) for the supplement. These figures use the inherited
whole-training route and do not measure original heldout accuracy. All full
and mini epoch losses decrease; the final mini batch contains five records.
All254348 primary and257064 supplement split attempts are retained, with no
split refusals. Independent expanded products differ by at most3.980e-13;
compact and full-vector CE differ by at most2.351e-14.

The original run takes17m56.65s and the independent repeat17m55.24s; peak RSS
is3771404/3771324KiB (3.60GiB), under the64GiB process limit. Seven native
checks, formatting, Clippy, release build, full Lean, both applicable Lean
audits, global axioms and the native INDEX check pass at the producer commit.
The existing36 unfinished Lean statements remain explicit. Summary SHA256 is
`6ed37b4db95ea78087b9317628b49c9345657a37fd9866bc8b729d61dcbca041`;
the56-artifact manifest is
`bae8eded33ae9c8541bf3b04769f3f5c01f539b52b09d340c284368d2367397b`.
Complete source, failed affine evidence, original/repeat outputs and review
logs are preserved under the `partitioned-boundary-v1` storage freeze.
Task336 will compare the fixed reviewed component on the original35 cases.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/PartitionedBoundaryAxioms.lean
cargo fmt --manifest-path experiments/convex_stress_partitioned_boundary/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_partitioned_boundary/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_partitioned_boundary/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_partitioned_boundary/Cargo.toml
bash experiments/convex_stress_partitioned_boundary/freeze.sh "$FREEZE" "$CARGO_TARGET_DIR/release/convex-stress-partitioned-boundary"
bash "$FREEZE/source/experiments/convex_stress_partitioned_boundary/run_inspection.sh" "$FREEZE/convex-stress-partitioned-boundary" "$PARENT_CASE" "$OUTPUT"
```

`PARENT_CASE` is the original Fibonacci case under `signed-compare/original-v2`.
The driver verifies all pinned complete references, copies the complete failed
335 evidence, and attempts both roles in every mode. Each inspection saves its
forecast before fitting and replays every state and update exactly. Independent
review rebuilds the committed source, repeats into a fresh directory and compares
every relative JSON hash. Keep frozen drivers/binaries immutable and preserve
all evidence on storage before removing a worktree. Reports remain ignored;
full Lean, global axioms and the native INDEX check are required.
