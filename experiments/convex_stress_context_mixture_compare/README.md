# Original context-mixture comparison

Does the reviewed CM45 mixture improve raw-label accuracy and ordinary CE while
retaining original stress-task success? This prospective comparison follows
Lean `2f0dc9ba`, native component `14b8ad6e`, the reviewed task290 comparison at
`347f29e1`, and its inherited CM43 context tensors from task287. The adapter
ports the complete-scope bookkeeping of task290's producer `d5ec25b6`.

| Scope | Treatment |
| --- | --- |
| All 35 original seed-1 tasks and both 105-choice policies | Every task/mode stays in the denominator. |
| Currently selected CM43 context representations | Import the exact current model hash and apply the reviewed mixture component. |
| Other selected models | Preserve reviewed predictions, metrics and costs explicitly. |
| Full / mini / forward | 64 full diagnostic steps / 8 epochs at the original batch size / forward vertex selection without gradients. |
| Raw-label metrics | Fresh full-vocabulary CE and raw-token accuracy on every original split. |
| Original success | Unchanged primary sequence accuracy threshold of 99%, with original many-to-one field pooling. |
| Baselines | Keep all 35 conventional transformer endpoints, all 336 depth outcomes and the distinct historical DiffGPT comparison. |

Eligibility depends on the selected representation and identity, never the
task name or held-out score. The current model hash is authoritative; nested
saved-endpoint metadata can describe an earlier tensor. Preparation matches
each eligible reference and hash to the reviewed CM43 selection seal and
completed manifest, then copies the original model bytes unchanged. Both
parent reviews, the component review, summaries, manifests and source
fingerprints are pinned before preparation.

The unchanged native generator checks original data fingerprints. Each
component must contain exactly the regenerated original training records,
including repeated IDs and labels. The minibatch size comes from the original
suite declaration. The mixture diagnostic excludes the query's source state
but does not refit discovery; it uses the fixed canonical head and transfers
the learned mixture to the frozen actual head. All selections and saved
mixture tensors are sealed before any original held-out metric evaluation.

Every changed endpoint first reproduces the old CM43 metrics exactly on all
splits. New actual-complex predictions then receive the same metrics. Original
observed CE must remain equal. Both priors and supplements retain their saved
support route. Full/minibatch inherited source update counts are recorded as
unknown, separately from the measured new mixture gradients, scores and
updates. Forward adds no derivatives or optimizer updates.

Every eligible model is evaluated regardless of improvement. CE/raw-accuracy
tradeoffs and original-success regressions remain visible; no held-out result
selects a fallback. The summary counts each actual new model's work once,
separately from its reuse in both initializer policies. Failed or missing
cases cannot inherit success. Existing discovery and all prior constructor
costs remain in the report; this is not a matched-compute speed comparison.

The frozen executable uses the optimized test profile with debug assertions.
Each process is capped at 64 GiB; aggregate model buffers must forecast below
56 GiB. Component construction/optimization retains its 32-billion work guard,
old-plus-new metrics allow 64 billion per endpoint, and the case total allows
512 billion across all retained prior/supplement endpoints. Replays repeat
construction and loading, and their measured resource costs are retained
separately. At least 16 GiB of disk must remain before preparation. Generated
artifacts live on storage and are not committed.

The eight adapter fixtures pass: original-record and mode binding, held-out
target independence, actual tensor reconstruction, stale metadata rejection,
complete scope and baselines, separate raw/primary success, regression and
failure accounting, work totals and source fingerprints. Original-data results
will be authored separately after the committed protocol runs.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_context_mixture_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_context_mixture_compare/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_context_mixture_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --profile test --manifest-path experiments/convex_stress_context_mixture_compare/Cargo.toml
bash experiments/convex_stress_context_mixture_compare/run_original.sh FROZEN_EXE REQUEST_JSON FRESH_OUTPUT_DIRECTORY
prlimit --as=68719476736:68719476736 -- jq -f experiments/convex_stress_context_mixture_compare/report.jq FRESH_OUTPUT_DIRECTORY/summary.json
```

The request contains the producer commit, task290 parent root/manifest/review,
task287 context root/manifest/review, and task292 component review. Freeze its
hash and the executable before preparation. A complete fresh reproduction
must rebuild all cases and reproduce every JSON model, seal, metric and summary
artifact before task293 review. No original CM45 accuracy gain is claimed by
the synthetic fixtures.
