# Original stress comparison with raw-margin constraints

This is the separate budget-corrected rerun. The v1 source, executable,
protocol, all35 attempts, failure logs and partial sealed forward model remain
unchanged. Its complete manifest is a required input: preparation checks every
file, reconstructs all35 process statuses, records the unstarted replay after
the work refusal, and seals that history for the new summary/report. Historical
failure metadata never supplies coefficients or chooses a candidate.

This adapter compares the reviewed CM47 component `26af04ec` with all35
original seed1 stress tasks. Task296 supplies the exact stronger raw reference;
task298 supplies a separately retained best-CE comparison, including its raw
regressions and complete earlier reference. Neither held-out scores nor task
names choose a model. Eligibility uses the current selected CM46 representation
and its actual saved-model digest.

Both105-choice initializer policies, all original modes and records,35
conventional-transformer endpoints,336 depth outcomes, prior/supplement routing
and every inherited cost remain present. The fixed reference and all simplex
coefficients are sealed before fresh metrics. Original primary success stays
`sequence_accuracy >= 0.99`; ordinary raw CE and raw accuracy are separate.
Every raw prediction must equal the reference, while probability-pooled field
accuracy and validation CE may change. Those changes remain in the report.

The guarded component has substantial temporary all-query/all-class caches.
This runner constructs endpoints sequentially, saves complete models to disk,
and releases each endpoint before constructing the next one. After the entire
selection seal is written, it reconstructs one endpoint at a time and checks
its exact model hash before scoring. Replay additionally loads each saved model
through full recursive reconstruction. The peak forecast is the largest live
endpoint, not a sum of scratch buffers that are never simultaneously live.
Each endpoint's prior/supplement forecast is checked before either guarded
cache is allocated. Three complete construction passes are reserved in the
execution-work forecast, covering fresh construction, replay loading and the
post-seal scoring pass. Actual optimization and verification counters remain
separate from those conservative forecasts.

Memory limits stay56GiB forecast and64GiB address space. Each guarded component
has a16trillion worst-case scalar-work guard; the complete case allows192trillion
for all original endpoints and reconstruction passes. This increased finite
bound counts potentially rejected backtracking work, not a measured runtime
claim. Metric work remains capped at64billion per endpoint, and disk reserve
is at least16GiB. Coefficients, margin certificates, ordinary CE diagnostics,
gates, profiles, original source identity and costs are all hash-bound.

All thirteen native adapter tests pass. They cover complete scope and failure retention, eligibility
independent of held-out scores, actual full/mini/forward models, repeated original
records, raw/field metric distinction, source and mode mutations, all retained
reference choices and costs, review barriers, and disk reconstruction after
releasing the original model. The source, binary and request must be frozen
before original evaluation. This README makes no new original numerical claim.
Previous suite results motivated the architecture, so this is not an untouched
architecture-selection holdout.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_margin_budget_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_margin_budget_compare/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_margin_budget_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked \
  --manifest-path experiments/convex_stress_margin_budget_compare/Cargo.toml
bash experiments/convex_stress_margin_budget_compare/run_original.sh FROZEN_EXECUTABLE REQUEST.json FRESH_OUTPUT
jq -f experiments/convex_stress_margin_budget_compare/report.jq FRESH_OUTPUT/summary.json
```

The request supplies the producer commit, absolute parent output directory and
completed manifest, exact296 and302 review exit/commit files, the complete
298 reference directory/manifest/review files, and the previous failed attempt's
root and manifest. `protocol.json` pins their exact
hashes. Generated models, metrics, logs and resources remain ignored on storage;
authored findings will be recorded in `original/README.md` after completion.
