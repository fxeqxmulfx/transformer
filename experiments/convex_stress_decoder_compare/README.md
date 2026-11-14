# Original categorical decoder confidence comparison

Does the reviewed CM44 complex head reduce raw-label cross-entropy for saved
categorical phase decoders while preserving the original task outcomes? This
prospective comparison follows Lean `a8f0020a`, component `9ddda0f8` and the
reviewed task 287 comparison at `c5ba15c6`. Its scope and bookkeeping derive
from the frozen CM43 adapter at `123dfc9c`; all changed source imports and
predictions use this separate package and protocol.

| Scope | Treatment |
| --- | --- |
| All 35 original seed-1 tasks, both 105-choice policies | Retain every task and training mode. |
| Selected identified, consequence and fiber phase decoders | Import the exact saved model and construct the reviewed complete complex confidence head. |
| Other representations | Explicitly retain their reviewed models and metrics. |
| Full, minibatch and forward-only modes | Preserve inherited endpoint metadata; add zero optimizer updates. |
| Raw labels | Recompute ordinary full-vocabulary CE and raw-token accuracy. |
| Original success | Preserve primary sequence accuracy at least 99%, including original field pooling. |

Preparation verifies the complete reviewed parent manifest, summary, inventory
and exact-commit review, along with the reviewed component. It retains all
35 conventional transformer endpoints, all 336 depth outcomes and the complete
prior cost records. Eligibility depends only on the saved representation and
selected model identity; it does not filter by task name, raw accuracy or CE.

Saved identified models come from their original `proposal.json`, consequence
models from `bank.json`, and fiber models from `report.json`. Every extracted
typed tensor must have exactly the model SHA-256 already recorded in the
reviewed parent selection. Container hashes and extracted source bytes are
sealed in the preparation inventory. No source is rediscovered from answers.

The unchanged native suite generator verifies its original data fingerprints.
CM44 uses only the regenerated original training records to select a positive
margin. Every repeated record keeps its own ID and weight. All complex code,
coefficients, actual-product caches, margins, budgets and source bindings are
saved before held-out evaluation. The output head's budget is reported
separately from the inherited source's latent-moment budget.

For each transformed endpoint, a fresh evaluation of the old source must first
reproduce every old metric exactly. The new head then receives fresh original
metrics on every split. A replay reconstructs the complete head, validates its
saved loading path, and compares all model, seal and metric bytes. Every
eligible head is transformed regardless of its held-out change; regressions
remain visible. Missing cases or failed processes cannot inherit a success.

The executable uses the optimized test profile with debug assertions. Every
process is limited to 64 GiB of address space, and each case forecasts its
complete head construction and both old/new metric work. All old discovery
costs remain in the parent evidence; no matched-compute speed claim is made.
Generated evidence remains ignored on storage. No original-data gain is
claimed before running the committed protocol.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_decoder_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_decoder_compare/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_decoder_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --profile test --manifest-path experiments/convex_stress_decoder_compare/Cargo.toml
bash experiments/convex_stress_decoder_compare/run_original.sh FROZEN_EXE REQUEST_JSON FRESH_OUTPUT_DIRECTORY
prlimit --as=68719476736:68719476736 -- jq -f experiments/convex_stress_decoder_compare/report.jq FRESH_OUTPUT_DIRECTORY/summary.json
```

The request names the committed producer, the absolute task 287 parent and
completed manifest, the parent and component review files, and the absolute
original identified/consequence/fiber run roots. Freeze the executable and
request hashes before preparation. The `fingerprint` command binds this
adapter, the reviewed CM44 component, the unchanged CM43 adapter and its
complete dependency chain. Greater confidence does not identify missing raw
answers; those still require additional evidence or modeling assumptions.
