# Complete comparison with partial-parent fallback

This is the task 276 shared-role comparison with the proved CM41 predecessor
availability rule. The initial task 277 run exposed original long inputs beyond
some saved count parents' domains. Its failures remain separate evidence.
This adapter has no new original benchmark result yet.

The common comparison modules are imported directly from the unchanged
`convex_stress_shared_compare` source at `15761f51`. Its compiled fingerprint
is checked against `protocol.json`. The new native wrapper is the reviewed
`convex_stress_partial_fallback` at `4f6ebe46`, backed by the Lean CM41 proof at
`baf7c2e4`. The change applies to every selected causal head by the same rule.

| Decision | Fixed rule |
| --- | --- |
| Head selection | Existing parent first, then ascending prime; ordinary observed raw CE must improve by more than `1e-12`. |
| Shared-role extension | Same-prime, same-mode CM40 head on a selected raw kernel, with positive added support. |
| Causal prediction | Candidate when supported; saved predecessor on unsupported inputs inside its own domain; candidate when that predecessor is unavailable. |
| Primary success | Original primary sequence accuracy of at least 0.99; auxiliary splits remain diagnostics. |

No held-out answer or score selects a head or route. Parent availability comes
from its checked saved representation: Counts keeps its finite saved context,
and Causal keeps its original prediction domain through 4096. The candidate
also keeps that domain. Both heads and original buffers remain charged, and
there is no new feature discovery or optimizer update.

The four original data sources remain exact reviewed 246, 252, 274 and 237
summaries, inventories, binaries and approval receipts. The adapter retains
all 35 tasks, both independent 105-choice initializer arms, the conventional
transformer and all 336 depth outcomes. Every unchanged metric is explicitly
marked as inherited reviewed evidence. Every changed actual model receives
fresh original metrics and prediction masks after its selection and tensors
are sealed. Fresh replay reconstructs those states and recomputes their scores.
Failed processes retain their artifacts and cannot acquire ancestor success.

Causal masks now distinguish `supported_mask`, `parent_available_mask`,
`parent_route_mask` and `candidate_without_parent_mask`. The last identifies
an unsupported input routed to the candidate because its parent is unavailable.
A supported input may also have an unavailable parent; the support and
availability columns preserve that distinction. These are routing facts,
not correctness certificates for unseen answers.

The original raw/shared histories, original record order and multiplicity,
full/minibatch/forward modes and constructor costs remain bound. Ordinary raw
cross-entropy and its supported-batch derivatives are unchanged. CM41 uses
actual zero-extended parent coordinates and flat component bounds; the fixed
feature library and route are not jointly optimized transformer parameters.

Tests reuse the reviewed selector, import, tensor, metric, failure and disk
fixtures. An additional full-runner fixture reproduces the old short-count-parent
refusal and checks the corrected actual model through length 4096, saved-state
replay and every route mask. Changing only its held-out label changes its score
while leaving the sealed construction unchanged. Both complete initializer arms
retain all 35 outcomes in the synthetic comparison.

The 64 GiB process limit, combined 56 GiB buffer forecast, uniform 32-billion
metric-work cap, 8 GiB artifact limit and 16 GiB disk reserve are unchanged.
Every changed endpoint is checked before any endpoint receives a metric.
Outputs stay on the storage disk. This remains the original seed-1 comparison;
DiffGPT's recorded 35/35 on seeds 0, 1 and 2 is separate historical evidence.

Build and check from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
mkdir -p "$TMPDIR"
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
manifest=experiments/convex_stress_domain_compare/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path "$manifest" -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path "$manifest" --all-targets -- -D warnings
```

After committing, freeze the clean source graph, CLI, tests and protocol. After
task 280 and all declared parents pass review, freeze the original request:

```sh
freeze=/run/media/misha/storage/transformer-runs/freezes/domain-compare-v1
bash experiments/convex_stress_domain_compare/freeze_source.sh \
  "$PWD" "$(git rev-parse HEAD)" "$freeze"
bash "$freeze/freeze_original.sh" "$PWD" "$(cat "$freeze/producer.commit")" \
  "$freeze/request.json"
export DOMAIN_COMPARE_BINARY=$freeze/domain-compare-v1
out=/run/media/misha/storage/transformer-runs/convex_stress_domain_compare/original-v1
mkdir -p "$(dirname "$out")"
bash "$freeze/run_original.sh" "$freeze/request.json" "$out"
```

Fresh original review uses the frozen CLI's `replay-case ROOT TASK` on every
completed case and `summarize ROOT`, comparing the complete saved bytes.
Keep generated JSON, tensors, logs and archives local. Commit only source,
reproduction inputs and later authored findings. Frozen 255/261/267/277
comparisons and their historical outcomes remain unchanged.
