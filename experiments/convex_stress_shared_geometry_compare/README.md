# Shared geometry on the complete original35 stress suite

Task332 compares reviewed331 producer `b19b3bc9` with both complete319 and
reviewed330. Original success flags, raw99, native exact99 and ordinary CE
remain separate. The current validated original35 result is35/35 success
flags but34/35 raw99; no new original heldout result exists at this stage.

Lean first: `scripts/SharedGeometryFallbackAxioms.lean` audits CM38 and CM55
before native evaluation. Fixed affine sharing pulls the actual combined
head's convex normalized CE back to its parent-frozen budget domain. Covered
original batches have exactly the candidate objective as a function of shared
parameters, hence the same derivatives, including the real short tail. The
non-Dirac complex witness retains repeated records and nonzero fallback logits.
CM55 separately charges all shared coefficients and offsets. No new theorem
or `sorry`, global feature-discovery convexity or unknown-label certificate
is claimed. The standalone audit passes with standard axioms only.

The Rust adapter preserves all35 cases, both105-choice policies,35 conventional
endpoints,336 depth outcomes, all earlier failures and the separate historical
DiffGPT seeds0/1/2. It verifies all331 complete319 artifacts and156 reviewed330
artifacts. Complete baseline summaries and manifests are copied unchanged.
Eligibility is the same fixed representation prefix as330, not a task name,
arithmetic rule or heldout metric. The exact331 v2 constructor and its six
reviewed outcomes are hash-bound; each fresh fit must reproduce the full
saved outcome, including all candidate states and training history.

All three new endpoint states are sealed before any original metric. Original
prior/supplement routing is preserved. A new shared head acts on blocks covered
by original training; the actual319 predictor remains elsewhere. Original
observed CE is freshly evaluated through that same composite predictor.
Every raw audit records all319/new logits and, additionally, the retained330
logits/prediction/CE for that exact original record. It rejects any mismatch
in IDs, tokens, target, field, role, vocabulary or319 logits. Retained330 raw
files are copied and checked against the reviewed manifest; they are reported
as retained evaluations, not new330 training. Independent expanded products
check arbitrary post-CE shared coefficients. Fixes and regressions against
both prior models are retained. No changed state's previous metric is restored.

The complete radius is1048576 per new role;319/330 had different head budgets,
so this is not a matched-capacity claim. Report both stored shared parameters
and all expanded real head coordinates, plus inherited tensors and radii.
Candidate discovery stays outside the convexity claim. Full/mini retain the
original128 updates or8 epochs of127 records with the actual5-record tail;
forward has zero CE gradients. The final iterate is used even if it regresses.

Before fitting/scoring, each invocation reserves two old constructions, three
new training passes, all candidate folds, original records, gradients,
projections, metric/field calls and independent complete products. The new
component's `one_run_kernel_work` is charged for each reserved pass; repeated
run/replay invocations have their own resource logs. Cached330 arithmetic and
an8MiB row buffer are additionally charged. A separate phase bound covers
both complete summaries without simultaneous live predictors. The limits are
10T scalar kernel units per case,56GiB explicit buffers and64GiB process address
space. File hashing, serialization and disk reads are measured separately.

Seven native checks cover actual old/new states in both roles and every mode,
nonzero unsupported fallbacks, complete coefficient accounting, raw-logit
replay and malformed retained records, unchanged original-record/field-call
forecasts, categorical-label-invariant support, both105-choice policies,
regressions against both319 and330, and failure denominators. No original
heldout result has yet been reported by this adapter.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/SharedGeometryFallbackAxioms.lean
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_shared_geometry_compare/Cargo.toml -- --test-threads=1
cargo fmt --manifest-path experiments/convex_stress_shared_geometry_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_shared_geometry_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_shared_geometry_compare/Cargo.toml
experiments/convex_stress_shared_geometry_compare/freeze.sh "$FREEZE" "$CARGO_TARGET_DIR/release/convex-stress-shared-geometry-compare"
bash "$FREEZE/source/experiments/convex_stress_shared_geometry_compare/run_original.sh" "$FREEZE/convex-stress-shared-geometry-compare" "$OUTPUT" "$REQUEST"
```

Use fresh storage paths for `FREEZE` and `OUTPUT`. The request JSON fields are
`producer_commit`, `parent_root`, `parent_manifest`, `component_root`,
`comparison_root`, `comparison_manifest` and `diff_gpt_history`. They identify
reviewed319 `signed-compare/original-v2`, its complete331-file manifest,
reviewed331 `shared-geometry/inspection-v2`, reviewed330
`shared-head-compare/original-v2`, its156-file manifest, and task318's frozen
historical snapshot. Exact content hashes are in `protocol.json`. Invoke the
driver from the immutable freeze and never edit it during execution. It
attempts all35 cases and every replay; incomplete cases remain in the35-task
denominator. Generated evidence stays on storage. All35 original metrics and
an independent full repeat must pass before any new99% claim.
