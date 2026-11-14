# Original training cohort for shared-role heads

Does the reviewed CM40 component construct usable complex heads on the original
paired stress tasks? This adapter prepares the same 23 cases and all 55 primes
from the prior grouped-kernel cohort, while retaining metadata for all 35 suite
tasks. It performs training and replay only. Original execution and complete
held-out comparison are separate follow-up steps; this adapter has no benchmark
success count.

The component is pinned to `5d4eee848cacceba911f048ade2d014e2b71cf42`, reviewed in
task 272. Its Lean proof is `ec5f839dd7584fc07765373132d255286f53f052`, task 271.
The adapter ports `convex_stress_readout_kernel/cohort` at the component commit.
It keeps the original data generators, source fingerprints, output partitions,
task order and minibatch sizes. It adds the reviewed shared-role equations and
binds original training records separately from their distinct equations.

Each original training index becomes a record ID. Its raw label and input order
are preserved, including repetitions. Only identical cell/label equations are
coalesced for the linear system; conflicting raw targets are rejected. The last
token is the row and the penultimate token is the column, matching the original
suite. Every case saves `input.json`, `records.json`, the complete declared
`readout.json` and all three policies. Replay regenerates the original inputs
and checks those files, both semantic hashes and artifact hashes, every complete
matrix certificate, actual complex head and update history.

| Mode | Fixed training schedule |
| --- | --- |
| Full | One projected CE update over every original record |
| Mini | One epoch in the saved original order, using the original batch size and actual tail |
| Forward | The same feasible complex initializer, zero updates and zero CE gradients |

All modes use ordinary cross-entropy over the full raw vocabulary. The output
partition changes the construction equations, not the CE labels. No input
arithmetic formula or unseen answer enters the learner. The added shared-role
family and compatible readout remain explicit unknown-target assumptions;
observed fit and flat features do not prove them.

The entire prime catalogue is retained. Cardinality mismatches, missing shared
identities, equation/work/nullity limits, incompatible groups and unavailable
heads remain recorded outcomes. A failed case retains its partial files and
process costs; later planned cases still run. A missing or failed replay cannot
be counted as complete. The runner requires a fresh destination and refuses to
overwrite partial results. Every final summary retains all planned cases and
all 35 original task names.

The native checks cover all 55 stages in a small complete case plus a retained
failed case. They check seven original records with five distinct equations,
the actual minibatch tail, raw aliases, unchanged construction after held-out
labels change, and tampering with sources, scope, models, partitions, record
IDs/order/multiplicity and replay files. Record-file edits still fail after
their ledger hashes are regenerated. Disk-pressure fixtures verify exit 125
without construction and continued processing of later cases.

All numeric processes run with a 64 GiB address-space limit. The component's
combined head, full matrix certificate and record/history forecast must fit
56 GiB; original work caps remain unchanged. The runner checks for 16 GiB of
free disk before each case and stage and streams stages individually. Complete
construction and discovery times are recorded separately, alongside actual
process resource reports and partial failure costs. Source archives, binaries,
logs and run files stay on the storage disk and are not committed.

Build and check from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-roles-cohort-tests
mkdir -p "$TMPDIR"
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
manifest=experiments/convex_stress_shared_roles/cohort/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path "$manifest" -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path "$manifest" --all-targets -- -D warnings
```

After committing and checking the adapter, freeze its source graph, compiled
fingerprint, release runner, test executable and scripts into a fresh directory:

```sh
freeze=/run/media/misha/storage/transformer-runs/freezes/shared-roles-cohort-v1
bash experiments/convex_stress_shared_roles/cohort/freeze_source.sh \
  "$PWD" "$(git rev-parse HEAD)" "$freeze"
```

After tasks 271, 272 and 273 are approved, create a separate original request
from that clean producer checkout and launch the unchanged frozen runner:

```sh
bash "$freeze/freeze_training.sh" "$PWD" "$(cat "$freeze/producer.commit")" \
  "$freeze/request.json"
export SHARED_ROLES_COHORT_BINARY=$freeze/shared-roles-training-v1
out=/run/media/misha/storage/transformer-runs/convex_stress_shared_roles/training-v1
mkdir -p "$(dirname "$out")"
bash "$freeze/run_training.sh" "$freeze/request.json" "$out"
```

Use the frozen executable's `replay-case ROOT TASK` and `summarize ROOT` commands
for fresh review. Keep original artifact paths stable because summaries bind
them. Original model evaluation must later retain both prior initializer arms,
all 105 choices per arm, full/mini/forward outcomes, transformer/depth baselines,
the original primary success criteria and all costs. Frozen comparisons 255,
261 and 267 and components 270 and 272 remain unchanged.
