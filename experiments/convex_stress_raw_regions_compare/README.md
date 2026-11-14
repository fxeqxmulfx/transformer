# Observed-region contexts on the original stress suite

Does the CM48 context map improve raw-label accuracy and ordinary cross-entropy
on the original tasks? This adapter wraps the reviewed component `0fee1a9d`.
It is ported from the reviewed complete task301 adapter at `7d858f69`, with the
margin guard replaced by fixed observed-region certificates. The exact CM46
models from task296 remain the source. Task301 is a complete secondary
reference, including its stronger CE result, all task298 lower-CE/higher-error
outcomes, every earlier cost and the failed work-cap attempt.

| Quantity | Protocol |
| --- | --- |
| Original scope | All 35 seed1 tasks; both 105-choice initializer policies |
| Training modes | Original full batch, minibatch and forward only |
| Source records | Exact order, repeats, short tails and all raw classes |
| Baselines | All 35 conventional transformer endpoints and 336 depth outcomes |
| New model | Observed integer hulls, independent rational coordinate certificates |
| Eligibility | Current CM46 representation and actual source-model identity |
| Confidence | Fixed at one before evaluation |
| New optimization | Zero gradient batches and zero optimizer updates in every mode |
| Original success | Unchanged primary `sequence_accuracy >= 0.99` |
| Additional outcomes | Raw accuracy and ordinary CE against both task296 and task301 |

Only uniquely covered, matched unseen inputs receive a different complex
feature row. Intersecting class hulls, missing coverage, unsupported axes and
observed inputs keep the exact parent. Prediction uses the complete unchanged
complex head. Neither task names nor held-out labels select a region, fallback
or confidence. Parent metrics are freshly recomputed and required to match the
reviewed result exactly. Observed ordinary CE must remain unchanged. New raw
predictions and all raw, CE or pooled-field regressions remain in the results.

CM48 proves identification conditional on convex raw-class regions inside the
source block. Disjoint observed hulls do not establish this assumption for
unseen labels. The benchmark measures correctness; it does not certify global
realizability. The earlier suite results motivated this architecture, so this
suite is not an untouched architecture-selection holdout.

Endpoints are constructed one at a time, saved on storage, and released. The
entire model selection seal precedes any new metric. Scoring reconstructs each
model and checks its exact saved hash. Independent replay also loads saved
geometry, rational rows and all parent state through complete reconstruction.
The peak forecast charges the largest live endpoint with both prior and
supplement; the work forecast reserves three complete construction passes.
Actual geometry counters remain separate from conservative work forecasts.

Limits are 56 GiB forecast, 64 GiB process address space, 128 billion scalar
work per component and 3 trillion per complete case. Both old and new metric
products are charged, with a 64 billion guard per endpoint. At least 16 GiB of
free storage is required before each case. Every failed or refused case remains
in the denominator and prevents a complete success summary.

Tests check complete scope, all original mode identities, repeated records,
geometry mutation rejection, eligibility independent of held-out scores,
source/record mutations, exact disk reconstruction and every retained reference
choice and cost. The stronger-reference comparison explicitly preserves a CE
regression even when the source-parent comparison improves. The component has
separate nonzero complex and exact integer geometry tests.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_raw_regions_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_regions_compare/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_raw_regions_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_regions_compare/Cargo.toml
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
bash experiments/convex_stress_raw_regions_compare/run_original.sh FROZEN_EXECUTABLE REQUEST.json FRESH_OUTPUT
jq -f experiments/convex_stress_raw_regions_compare/report.jq FRESH_OUTPUT/summary.json
```

The request supplies the producer commit, absolute task296 source directory,
completed manifest and exact review files; task304 component review files;
task301 reference directory, manifest and exact review files; and the previous
failed attempt's root and manifest. The protocol pins their exact hashes.
Source, binary and request must be frozen before original scoring. Generated
models, metrics and logs stay ignored on storage. Authored findings belong in
`original/README.md` after completion; no new benchmark gain is claimed here.
