# Original comparison of five shared class-count gates

Do five shared context-weight rows improve raw-label accuracy and ordinary
cross-entropy? The reviewed component `5e89c616` instantiates the generic CM46
proof `c60816fa`. This adapter ports producer `1b42f411`, keeping the exact CM45
parent reviewed at `f0a19727`. Task296's block-local mixtures improved some raw
counts but increased CE in every mode. Its complete negative evidence remains
pinned and explicitly retained here. That earlier evaluation motivated this
architecture choice; the original suite is not an independent model-selection
holdout across these successive experiments.

| Scope | Treatment |
| --- | --- |
| All 35 original seed-1 tasks, both 105-choice policies | Keep every task/mode and every original denominator |
| Selected CM45 representations | Load exact reviewed model bytes and add five shared gates |
| Other representations | Retain every reviewed endpoint, metric and cost |
| Full / mini / forward | Inherit mode and batch; 64 local full steps / eight epochs / zero gradients |
| Task296 negative reference | Keep all 35 records, both 105-choice policies, every split metric, fresh outcome, change and local work count |
| Baselines | Preserve 35 conventional transformer endpoints and 336 depth outcomes |
| Metrics | Reproduce CM45 metrics exactly; then report original success, raw accuracy and ordinary CE separately |

The component fixes profiles from unique observed source states, excluding the
query's own observed state once even when records repeat. The profile selects
one of five rows by minority-label fraction. Raw label names have no ordering
meaning. Gates are fixed before local optimization. The same radius 0.5,
CM45 anchor, 91-map bank and complete complex head remain. Original source
blocks still index features; the coarse gates index only context weights.
The source-excluded diagnostic is ordinary full-vocabulary cross-entropy.
Discovery is not refitted for each excluded state. Neither the diagnostic nor
the flat-spectrum bound certifies unseen raw labels.

Eligibility uses only CM45's currently selected representation, reference and
actual model hash. Original records, mode, batch and source reference must
match. Every eligible model is evaluated, with all gates and weights sealed
before held-out metrics. There is no score-based fallback to CM45 or task296.
Observed loss and original success rules remain unchanged; lower diagnostic CE
may still worsen raw accuracy or held-out CE.

Preparation validates the reviewed CM45 manifest, summary, inventory and model
bytes. It separately checks task296's exact review, all files in its completed
manifest, summary and inventory, then seals a projection retaining all choices,
metrics, fresh outcomes and costs. The complete unprojected negative summary
remains hash-bound on storage. That projection is included in the new summary
and report but cannot affect eligibility, optimization or predictions.

Native tests cover record/mode binding, held-out target independence, exact
model reconstruction, scope/baselines, success/raw-metric separation, explicit
regressions, work counts, reference projection and tampered reference evidence.
All 35 runs and replays must finish; missing or failed cases cannot inherit
success. New gradient batches, optimizer steps and local-row updates count each
actual model once, separately from policy reuse. All inherited costs remain.
Historical DiffGPT's recorded 35/35 on three seeds remains distinct from these
retained seed-1 conventional transformer endpoints; neither is retrained.

Every process has a 64 GiB address limit, with a 56 GiB model-buffer forecast,
32-billion per-component work guard, 64-billion old/new metric guard per
endpoint and 512-billion complete-case guard. The runner requires a 16 GiB
disk reserve before preparation and each case and retains refusals as exit125.
Generated outputs stay ignored on storage. These caps do not imply an
equal-compute or speed comparison.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_count_gates_compare/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path experiments/convex_stress_count_gates_compare/Cargo.toml -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_count_gates_compare/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --profile test \
  --manifest-path experiments/convex_stress_count_gates_compare/Cargo.toml
bash experiments/convex_stress_count_gates_compare/run_original.sh FROZEN_EXE REQUEST_JSON FRESH_ROOT
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_count_gates_compare/report.jq FRESH_ROOT/summary.json
```

The request supplies the producer commit; task293 parent root, completed
manifest and review files; task297 component review files; and task296 local
reference root, completed manifest and review files. Freeze source inventory,
protocol, request and executable before original evaluation. Review requires
an independent complete preparation, all 35 cases/replays, and byte-identical
JSON/report reproduction. No original improvement is claimed before that run.
