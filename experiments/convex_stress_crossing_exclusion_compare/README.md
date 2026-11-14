# Crossing exclusion on the complete original stress suite

Does the reviewed CM50 hull-intersection rule improve raw accuracy and ordinary
cross-entropy beyond task310? This adapter uses component `a6be8048` and ports
the task310 adapter reviewed at `df79e005`. Its source is each exact saved CM49
prior and supplement model. All task310 and task305 outcomes, task301 guarded
CE results, task296 comparisons, task298 lower-CE tradeoff, previous failures
and inherited costs remain in the complete report.

| Quantity | Protocol |
| --- | --- |
| Original scope | All 35 seed1 tasks and both 105-choice policies |
| Training modes | Original full batch, minibatch and forward only |
| Records | Exact original order, repeats, short tails and raw labels |
| Baselines | All 35 conventional transformer endpoints and 336 depth outcomes |
| New model | Exact two-sided rational hull-intersection certificates |
| Eligibility | Current CM49 representation and exact selected model identity |
| Confidence | Fixed at one before scoring |
| New optimization | Zero gradients and optimizer updates in every mode |
| Original success | Unchanged primary `sequence_accuracy >= 0.99` |
| Raw outcomes | Complete primary raw totals, per-task raw counts and ordinary CE |

Only a unique surviving observed-label candidate changes a previously
uncovered feature row. All earlier hull rows, point-exclusion rows and other
parent predictions are retained. Uniform positive same-label source features
multiply every unchanged real and imaginary coefficient of the complete head.
No held-out answer, score or task name selects the geometric rule or confidence.
Convex target regions and membership of the missing answer in the candidate
pool remain explicit unproved assumptions. Flatness supplies feature bounds.
Earlier suite outcomes informed development; this is not a fresh holdout for
choosing the architecture.

All actual endpoints are constructed and sealed on disk before new metrics.
The complete parent metrics must reproduce exactly, including every original
observed metric object. Every split has an independent raw transition audit
of changed predictions, fixes, regressions and errors on newly identified
queries. Its native raw counts must equal the original metric implementation.
All extra complete products are charged. Full-suite totals use native raw
counts when pooled-field metrics also exist, retaining all 35 denominators.

Each of the 210 choices is compared with complete task310, task305 and task301
references. The saved task310 record includes all its outcomes and raw audits;
the earlier records, references and costs remain separately available. Current
metrics never overwrite a stronger reference or hide a regression. Saved-state
replay reconstructs and verifies both rational mixtures, all parent states,
prediction rows, original records, full coefficients and exact metric objects.

The unchanged one-endpoint-at-a-time protocol sums prior and supplement buffer
forecasts and reserves three complete constructions per endpoint. Reviewed
training-only forecasts total 3,020,412,676,608 construction-work units plus
104,700,653,568 units for metrics and raw audits. This exceeds the prior 3T case
allowance. The new prospective allowance is 4T, fixed before original scoring.
Limits remain 256B units per component, 64B for metrics/audits per endpoint,
56 GiB forecast buffers and 64 GiB process address space. At least 16 GiB free
storage is required. Resource refusals and failed cases stay in the denominator.

All 18 adapter tests pass. Native fixtures use actual full/minibatch/forward histories with 1/6/0 updates,
45 repeated records and a five-record tail. They check original identities,
nonzero complete complex products, held-out-target independence, certificate
and source mutations, disk sealing, both policies, complete references, raw
count conventions and explicit regressions. The separately reviewed component
has 13 geometry, complete-head and gradient tests. Fixtures are not original
benchmark gains. Task314 executes and independently reproduces all 35 original
cases only after this adapter's exact-commit review.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_crossing_exclusion_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_crossing_exclusion_compare/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_crossing_exclusion_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_crossing_exclusion_compare/Cargo.toml
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
bash experiments/convex_stress_crossing_exclusion_compare/run_original.sh FROZEN_EXECUTABLE REQUEST.json FRESH_OUTPUT
jq -f experiments/convex_stress_crossing_exclusion_compare/report.jq FRESH_OUTPUT/summary.json
```

The request supplies this producer commit; absolute task310 source directory,
completed manifest and exact review files; task312 component review files;
task301 reference directory, manifest and review files; and the failed task301
attempt's directory and manifest. The protocol pins every identity. Freeze the
source, executable and request before original scoring. Generated states,
reports and logs stay ignored on storage; authored findings follow separately.
