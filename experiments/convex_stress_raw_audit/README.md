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
