# Locate the remaining raw-label errors

Task305 is reviewed at `5dd63c6e`. Across its 35 primary splits, raw errors are
96 full batch, 97 minibatch and 95 forward only out of 279,952 target tokens
per mode. Fibonacci contributes 96/95/95; minibatch also retains two inherited
sum-square errors. This audit locates the errors before choosing another model
or extending the Lean assumptions.

The protocol pins the exact reviewed summary, inventory, completed manifest
and compiled parent source fingerprint. Every completed artifact is checked.
Both 105-choice policies and all 35 tasks must be present. Raw native counts
are used wherever field metrics also exist; target counts and reported raw
accuracy must agree. The complete task denominators remain visible.

For every fresh raw-region endpoint, the audit loads the actual saved complete
complex state against its external selection seal, reconstructs its parent
and geometry, and reproduces the entire original metric object for every
split. It then counts correct answers, changed predictions, fixes and regressions
by prior/supplement route and observed, covered, uncovered, inactive, unsupported
or overlapping context. This is reconstruction of existing trajectories;
inherited training and verification work are retained in the reported bounds.
The audit introduces no new training run or optimizer trajectory.

Every remaining error records its actual raw prediction, true label, original
parent prediction and CE. It diagnoses the observed point set of the true class:
missing support, affine rank, membership in its observed affine span, and, for
line-shaped classes, intersections of the extended line with other observed
class hulls. Selecting a class by the true label is explicitly a post-evaluation
diagnostic. These properties do not supply a new prediction, establish target
class extrapolation, or follow from spectral flatness. A new model requires a
separate Lean proof and a separately frozen complete comparison.

The two inherited minibatch sum-square errors remain in the full-suite totals.
Their endpoint is outside the fresh raw-region family, so this audit does not
claim to have reconstructed its per-example geometry. Existing original evidence
and every reference remain unchanged.

The executable reads a request with absolute `root`, `review` and `manifest`
paths. It prints one report only after every reconstruction check passes. The
prediction work forecast charges original and parent metric calls plus the
per-example diagnostic calls; its cap is 128 billion per endpoint. Run under
the 64 GiB process limit and keep generated output on storage.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_raw_audit/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_audit/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_raw_audit/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_audit/Cargo.toml
prlimit --as=68719476736:68719476736 -- FROZEN_EXECUTABLE REQUEST.json > FRESH_REPORT.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
```

The request points to `/run/media/misha/storage/transformer-runs/raw-regions-compare/original-v1`,
`.kanban/artifacts/305/review` and `.kanban/artifacts/305/completed.sha256`, using
absolute paths for the latter two. The source, executable and request are frozen
before the diagnostic report is generated. Authored interpretation will be
recorded here after the audit; no new model result is claimed.

## Frozen audit findings

Producer `5a59a9715a0ac26d2988e8e4359f94e1308b33ff` reconstructed every fresh
endpoint and every complete original metric object. The original audit and
its independent rerun completed successfully and produced byte-identical
reports. Exact findings-commit review remains pending. The audit confirms all
previous primary raw totals, including the two inherited minibatch sum-square
errors; it does not reconstruct that older endpoint's individual examples.

The geometry change makes exactly five raw prediction changes per Fibonacci
mode: five fixes and zero regressions. All 3,878 validation queries using the
new region context are correct. All 4,704 observed training records remain
correct, with unchanged original metrics.

| Validation route and context | Queries | Full errors | Mini errors | Forward errors |
| --- | --- | --- | --- | --- |
| Prior, covered region | 3878 | 0 | 0 | 0 |
| Prior, uncovered region | 620 | 91 | 90 | 90 |
| Prior, inactive block | 200 | 1 | 1 | 1 |
| Prior, unsupported input | 3 | 3 | 3 | 3 |
| Supplement, inactive block | 3 | 1 | 1 | 1 |
| Total | 4704 | 96 | 95 | 95 |

The true class is absent from the routed block's observations in five errors
per mode: the three unsupported inputs and the two inactive-block errors.
Among uncovered errors, one true class has a single observed point. Eight
queries lie outside the affine span of their collinear true-class observations;
37 lie on that line beyond the observed class hull. The remaining 45 full-batch
or 44 mini/forward errors have a two-dimensional true-class point set.

These observations reject a blanket inference from collinearity to affine
class support: eight known error queries already lie off their true class's
observed line. For the 37 on-line queries, the extended line intersects another
observed class hull. Extending all lines without domain and class consistency
checks is unsupported. A two-dimensional affine span is the entire coordinate
plane and cannot itself distinguish raw classes.

The next Lean question is convex exclusion. If assigning a candidate label to
an uncovered query makes its augmented class hull contain an observed point
of another label, convex target regions rule out that candidate. Identifying
one remaining label also requires an explicit closed candidate pool (or a
correct many-to-one decoder fiber). Neither that closure nor target convexity
follows from spectral flatness. The five missing-support errors remain a
separate issue. This is a proof proposal, not an evaluated prediction rule.

The audit charges 59,614,507,008 additional prediction-work units per endpoint,
plus the reported inherited reconstruction bounds. It took 70.18 seconds and
70.08 seconds for the two passes. Peak RSS was 3,474,992 KiB, about 3.31 GiB.
No model or confidence changed and no new training trajectory was introduced.

Evidence is local under `/run/media/misha/storage/transformer-runs/raw-audit/original-v1`;
the frozen source, executable and request are in `freezes/raw-audit-v1` under
the same storage root. The report SHA-256 is
`b544ba1a2d9d424e7559eb7c6bc65418a002ce47ebbc3ec13038db82e6bbd927`.
The completed manifest `.kanban/artifacts/306/completed.sha256` has SHA-256
`3f2758aed5010ff40a9d19a973e003d12dad1310a7418b0ec962a586828e817d`.
Generated reports, timings and logs remain ignored; authored interpretation
and reproduction commands are committed here.
