# Original confidence calibration comparison

Does the reviewed CM43 confidence rule reduce raw-label CE on the actual
reviewed CM42 stress-suite heads? This prospective native comparison follows
Lean `2dcaa38a`, component `0eb9fd7c` and reviewed task284 at `d173da9f`.
The adapter is based on task284's producer `cb71deb9`, with a new frozen parent,
actual saved context-model import, and a separate protocol and output directory.

| Scope | Treatment |
| --- | --- |
| All35 original seed1 tasks, both105-choice policies | Keep every selected task/mode outcome |
| Stored selected CM42 heads | Apply the reviewed training-only confidence rule |
| Other selected models | Explicitly inherit reviewed metrics |
| Full/minibatch/forward-only | Retain actual saved coefficients and zero extra updates |
| Raw labels | Fresh ordinary CE and raw-token accuracy |
| Original task success | Unchanged primary sequence accuracy at least99%, with original field pooling |

The immutable task284 summary, inventory, all completed artifacts and exact
review are pinned externally. Preparation verifies every original result,
selection seal and eligible model tensor before metrics. It retains all35
conventional transformer and336 depth outcomes. The complete parent remains
available by hash and path; its prior construction costs and CM42 context
forecasts are retained. Original process resource files stay in the pinned
parent manifest rather than being presented as new costs.

The original native generator and suite remain unchanged. Training record IDs,
input cells and raw labels must equal those in each saved context model. CM42's
selection, neighborhoods and source tensor are reconstructed and validated.
CM43 selects only a positive confidence scale using original training records,
independently of actual trained coefficients or held-out targets. Every new
model is saved and sealed before evaluation. The prior/supplement route remains
unchanged, and observed CE must agree with the reviewed parent.

Replay rebuilds the actual scaled complex features and compares complete saved
models, selections and metrics. All failures and regressions remain in scope;
no incomplete process set can produce a complete summary. Positive scaling
preserves exact-real raw ranking, but floating-point rankings and pooled task
metrics are measured afresh. No improvement is assumed before this run.

The executable uses the optimized test profile (`opt-level=2`, debug assertions).
All processes have a64GiB address limit. Timings include reconstruction and
evaluation and are not a matched-compute comparison with the transformer.
Generated evidence stays local and ignored on storage.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_context_scale_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_context_scale_compare/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_context_scale_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --profile test --manifest-path experiments/convex_stress_context_scale_compare/Cargo.toml
bash experiments/convex_stress_context_scale_compare/run_original.sh FROZEN_EXE REQUEST_JSON FRESH_OUTPUT_DIRECTORY
prlimit --as=68719476736:68719476736 -- jq -f experiments/convex_stress_context_scale_compare/report.jq FRESH_OUTPUT_DIRECTORY/summary.json
```

The request names the committed producer, absolute task284 original directory,
completed manifest and exact-commit review files, and the reviewed task286
component files. Copy the executable and request into a frozen directory and
record their SHA256 values before running. The `fingerprint` command binds this
adapter, CM43 component, the reviewed CM42 adapter and its complete source chain.
