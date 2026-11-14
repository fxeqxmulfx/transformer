# Original raw-label context comparison

Does the reviewed CM42 context map improve raw-token accuracy and ordinary
cross-entropy on the original selected complex heads? The component at
`c62022c5` follows the Lean proof at `81f4e46e`. This prospective comparison
uses the complete reviewed task281 seed1 result, with unchanged primary success
criteria and all conventional transformer and depth outcomes.

| Scope | Treatment |
| --- | --- |
| All 35 tasks and 105 choices per initializer | Retained in both original policies |
| Stored selected raw/shared-extension pair heads | Apply the reviewed training-only context policy |
| Other selected heads | Preserve reviewed metrics with explicit inheritance |
| Full, minibatch and forward | Reuse actual saved coefficients; zero extra optimizer updates |
| Raw labels | Fresh native-token accuracy and ordinary full-vocabulary CE |
| Original success | Primary sequence accuracy at least 99%, including unchanged field pooling |

Eligibility depends on the frozen saved representation and its selected
reference. It is sealed before metrics. Task281 contains three such endpoints,
all Fibonacci, one in each training mode. The context component's canonical
margin24 selection uses only original training inputs and labels, retaining
record multiplicities and excluding the query state from its own neighborhood.
The full latent discovery is not refitted for that diagnostic. The ordinal
input metric remains an explicit locality assumption; flatness provides a
feature/head bound and does not identify unseen raw labels.

Preparation checks the exact reviewed parent summary, inventory, completed
artifact manifest and review commit. Every imported case and eligible saved
model is hashed against that manifest. The embedded native `parent.jq`
projection retains every selected metric and all 35 transformer and 336 depth
outcomes, with the complete immutable parent retained by its hash and location.
The original native data generator checks its stored train/validation tensor
fingerprints. Original record IDs and raw labels must match the saved heads.

Every changed model is built and written before the first metric call. Shared
extensions keep the exact original prior/supplement support gate. The actual
complex feature products use the stored source coefficients, and original
training CE must remain equal. Independent replay reconstructs each context
map and tensor and compares saved model bytes and complete metrics. A failed
or missing original process/replay prevents a complete summary. Regressions
are retained even when raw CE improves. Unchanged choices remain explicitly
inherited rather than presented as freshly evaluated.

No original numerical improvement has been recorded yet. Outputs, logs and
resource measurements stay ignored on storage. The existing reviewed 35/35
model and the older 255/261/267 comparisons remain frozen.

Build the fixed evaluation executable using the optimized test profile
(`opt-level=2`, with debug assertions), and copy it into a frozen artifact
directory before starting the run. These timings are not a matched-compute
comparison with the historical transformer.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_context_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_context_compare/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_context_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --profile test --manifest-path experiments/convex_stress_context_compare/Cargo.toml
bash experiments/convex_stress_context_compare/run_original.sh FROZEN_EXE REQUEST_JSON FRESH_OUTPUT_DIRECTORY
prlimit --as=68719476736:68719476736 -- jq -f experiments/convex_stress_context_compare/report.jq FRESH_OUTPUT_DIRECTORY/summary.json
```

`REQUEST_JSON` names the producer commit, absolute task281 original directory,
its completed manifest and exact-commit review files, and the task283 component
review files. Protocol hashes are embedded in `protocol.json`. The executable's
`fingerprint` command binds this adapter, the context component and the complete
reviewed parent source fingerprint. A fresh `prepare` with the same request
must reproduce the inventory and parent snapshot byte for byte.
