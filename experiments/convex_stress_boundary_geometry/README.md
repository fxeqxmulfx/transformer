# Share raw-label boundaries through input geometry

Task335 tests whether a common boundary map can reduce the uncertainty left by
331's independent block intercepts. Its separate original35 comparison332
retained35/35 original success flags, but only34/35 raw99 tasks and worse
Fibonacci cross-entropy. This experiment uses original training records only.
Original heldout comparison belongs to336; no improvement is claimed here yet.

Lean first: `12b338cc`, `scripts/BoundarySharingAxioms.lean`, combines CM55's
complete affine expansion and CM51's normalized signed features. The concrete
four-state witness has signed descriptors, nonzero boundary parameters and a
nonzero categorical offset. Generic results cover the actual expanded budget,
ordinary convex full/minibatch CE and bounded features. They do not prove the
native floating-point implementation or unseen raw-label correctness.

The native head keeps331's complete eight-state normalized quadratic Fourier
code and categorical binary decoders. It replaces every free block intercept
with the same affine function of input-only route extrema. All complete
coefficients remain charged, including absent raw classes and empty blocks;
fixed sharing compresses storage to at most25 parameters per role. Raw class
numbers never enter a feature or regression target: only categorical minus/plus
orientation is learned. Discovery and hyperparameter selection precede the
convex ordinary-CE stage and are not asserted to be jointly convex.

| Choice | Declared alternatives |
| --- | --- |
| Input frame | Entire grid; nonmodal-route bounding rectangle |
| Block descriptors | Axis extrema; axis and both diagonal extrema |
| Constructor | Global uncentered class-sign ridge moments; shared regression of observed interval midpoints |
| Orientation | The same16 fixed generic directions as331 |
| Ridge / gain / common offset | `[1e-6,1e-4,0.01]` / `[8,32,128,512]` / `[8,32]` |
| Actual CE updates | Full128 steps at0.1; mini8 epochs of127 at0.1; forward0 |

Every configuration uses the same three deterministic cell-exclusion folds;
repeated cells stay together. The3072 pooled ordinary-CE scores select a single
configuration, followed by full-data refitting. Retain1152 fold states and9216
gain/common evaluations, every refusal, the local329 controls, all real batch
tails, final parameters and complete independent products. Unlike331's centered
moments, global moments retain the block-constant descriptor signal. Interval
regression shares boundary midpoints across blocks; overlapping training
intervals use balanced class-centroid midpoints and retain their signed gaps.

The inherited route was learned from the whole original training set. These
folds diagnose the new head conditionally and are not an unbiased whole-model
generalization test. A class absent from a block's fitting records remains
absent from that block's decoder. Flatness alone supplies neither that label
nor a target-identification certificate.

CE step sizes are0.1, declared before production. The normalized complex row
has squared norm at most1, and the nine bounded descriptor coordinates add
at most9. The categorical contrast takes values in[-1,1], so10 bounds the
smoothness of each record loss in these shared coordinates. This accounts for
the stronger coupling introduced by sharing offsets. Full-batch decreases and
mini final losses are still measured, not inferred from a floating-point proof.

Five checks pass before production: full expanded products and budgets;
input-only descriptor invariance; finite-difference gradients and convexity;
exact forward/full/mini reconstructions with real tails; categorical renaming,
complete candidate accounting, corruption rejection and early bounds. The
prospective primary dimensions give at most961,452,123,648 scalar units for
one full inspection and its internal replay, and4,333,630,976 buffer bytes.
These are conservative dimension forecasts. Hashing/serialization and disk I/O
are measured separately. Explicit buffers stay below56GiB and processes64GiB.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/BoundarySharingAxioms.lean
cargo fmt --manifest-path experiments/convex_stress_boundary_geometry/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_boundary_geometry/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_boundary_geometry/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_boundary_geometry/Cargo.toml
bash experiments/convex_stress_boundary_geometry/freeze.sh "$FREEZE" "$CARGO_TARGET_DIR/release/convex-stress-boundary-geometry"
bash "$FREEZE/source/experiments/convex_stress_boundary_geometry/run_inspection.sh" "$FREEZE/convex-stress-boundary-geometry" "$PARENT_CASE" "$OUTPUT"
```

`PARENT_CASE` is the original Fibonacci case directory beneath
`signed-compare/original-v2`; all three complete parent hashes are fixed in
`inspection-inputs.json`. The driver verifies retained319/329/330/331/332
references, copies complete331/332 summaries, and attempts all six mode/role
inspections. Each saves its forecast before fitting and reconstructs every
candidate, selection, state and update exactly. For independent review, rebuild
the committed source and repeat the frozen driver into another fresh directory;
compare every relative JSON hash. Keep complete evidence on storage, outside
the worktree. Never edit a running driver or executable. Generated reports and
logs remain ignored. Full Lean, global axioms and native INDEX checks apply.

## Reproduced failure

Frozen source `c09c06bdfb20c3d445b287c5b71c3e3fcd5debdb` completes all six
training-only inspections and their internal replays. The independent repeat
reproduces all28 JSON artifacts exactly, including complete331/332 references.
Both complete summaries hash to
`1b40f427e1ec6607d7e49fcccc88ebc42d6b6d350eac466b9975eb7aaceaf18d`.

| Mode | Primary training correct/4704 | Primary CE | Supplement correct/4704 | Supplement CE |
| --- | ---: | ---: | ---: | ---: |
| Forward | 4494 | 0.2504598 | 4300 | 0.2953495 |
| Full | 4513 | 0.2478481 | 4305 | 0.2951052 |
| Mini | 4501 | 0.2454519 | 4310 | 0.2952899 |

Conditional fold scores are4512/4704 primary and4417/4704 supplement, identical
across optimization modes. The best raw primary result anywhere in the entire
3072-configuration bank is4513/4704, below the old331 fold result4623/4704.
The minimum-CE primary choice uses global interval regression on whole-grid
diagonal extrema, direction9, ridge1e-6, gain8/common8. The supplement chooses
global moments on nonmodal-box diagonal extrema, direction1, with the same
ridge/gain/common. Full128 and mini304 steps are effective; forward0. Both
gradient modes lower final ordinary CE but do not repair the representation.
There is no new original heldout evaluation and no improvement claim.

The complete fit takes16m27.27s; repeat16m26.27s. Peak RAM is3,771,064 and
3,770,564KiB, about3.60GiB. Five native checks and full Lean/axiom/INDEX checks
pass independently. Evidence is preserved in storage `boundary-geometry`
and `freezes/boundary-geometry-v1/findings-review`; relative manifest SHA-256 is
`dfb6b2e57ba1a470afb01b103ab39b645f5ee22222025969baa459d61e6bf16b`.
Training-only boundary diagnostics show discontinuities missed by the single
affine map. The next declared revision is
[`convex_stress_partitioned_boundary`](../convex_stress_partitioned_boundary),
which keeps this failure intact and checks fixed partition sharing in Lean first.
