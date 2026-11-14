# Complete comparison with shared-role coverage

This adapter tests the shared-role model on the original 35-task suite. It
starts from reviewed comparison 246, preserves both 105-choice initializer arms,
and keeps the conventional transformer and all 336 depth outcomes. Its new
model construction is fixed before original metrics. Adapter tests are
synthetic; this source has no new original benchmark result yet.

The original stages are already complete and independently reviewed. Task 252
supplies all 23 raw complete-kernel training cases and all 55 primes. Task 274
supplies the corresponding shared-role cohort, including original record IDs,
order and multiplicity. Task 237 supplies the original causal histories.
Task 275 binds actual CM34 and CM40 certificates to the conservative CM39 gate.
The fixed CM38 predecessor fallback is the reviewed native component at
`027237cd`, with the importer/evaluator ported from `3852e0b0`.

| Step | Rule fixed before scoring |
| --- | --- |
| Raw head selection | Existing parent first; ordinary observed raw CE, then ascending prime, with a strict improvement greater than `1e-12`. Declared output-readout compatibility is required. |
| Shared-role extension | Only a selected raw head can use its available same-prime, same-mode CM40 head, and only with positive added support. Keep prior logits wherever the prior is supported or both heads refuse. |
| Causal fallback | Only a selected causal head receives the exact saved predecessor identified by the original ancestry and checkpoint digest. Keep the candidate on its existing support. |
| Unavailable transformation | Retain the selected head and record the exact refusal. |

The ordinary selector is imported directly from `convex_stress_kernel_compare`.
Its inputs contain original training CE and source bindings. No held-out label,
score or mask chooses a head, prime, structural transformation or initializer
arm. Both transformations preserve the selected observed objective. There are
no new optimizer steps. All source full/minibatch/forward histories are bound;
actual selected histories receive fresh numerical replay before evaluation.

The model uses the original categorical inputs and raw CE labels. Shared-role
alignment adds the actual CM40 matrix equations, with a free common role offset;
it does not inspect an arithmetic answer formula. The declared output partition
remains separate from the raw vocabulary used by CE. Global shared-role target
realizability and compatible readout remain explicit unproved premises. Neither
observed fit, extra support nor flat Fourier features certify an unseen label.

The complete parent summary and exact approval receipt are immutable inputs.
Every parent producer, compiled source fingerprint, executable, summary,
inventory and review commit is pinned in `protocol.json`. Preparation and final
summary verify the complete imported artifact inventories. Each case verifies
the imported files it actually uses, and freshly replays selected model histories.

An unchanged model's metrics are explicitly marked `inherited_reviewed_parent`.
Every changed model is scored afresh on every original split, with provenance
`fresh_changed_model`. Historical unchanged metrics are never described as a
fresh rerun. This reuses reviewed evidence without recursively executing completed
ancestor experiments. Frozen comparisons 255, 261 and 267 remain unchanged.

Every case saves its fixed selection, actual complex tensors and a selection
seal before the first score. Fresh replay reconstructs the selected models from
the bound original histories, checks the saved tensors and every structural
choice, and recomputes changed-model metrics and masks. Raw masks distinguish
encoded axes, supported vector matches, exact observed pairs and the supplemental
route. Causal masks retain candidate support, parent routing and exact original
prefixes. A mask is a support diagnostic, not an unseen-correctness certificate.

Success follows the original primary **sequence accuracy of at least 0.99**.
Auxiliary failures remain diagnostics. Each initializer arm still has 35 outcomes
per mode, with missing models and process failures retained. A failed changed
case does not acquire a success flag from its ancestor. This remains the original
seed-1 comparison; DiffGPT's recorded 35/35 on each of seeds 0, 1 and 2 is separate
historical evidence.

All original constructors are charged once across the two arms. Raw and shared
cohort construction costs are retained in full, including unused fields and
refusals. The causal cohort cost is already in the complete parent and is not
charged twice. Verification, routing, inference and serialization are new costs,
separate from optimization. Original forward-only histories still have zero
updates; checking another saved mode's history is verification work.

Numeric processes have a 64 GiB address-space cap. Saved heads, source histories,
certificates and original record buffers retain the combined 56 GiB forecast.
All changed endpoints receive the uniform 32-billion-operation metric preflight
before any endpoint is scored. Original prefixes are checked through length
4096. The existing bounded artifact reader/writer enforces 8 GiB per file. The
runner checks a 16 GiB disk reserve before preparation and each case, keeps every
partial artifact or exit 125 refusal, continues later planned cases, and requires
a fresh destination. Output paths remain stable on the storage disk.

Native tests cover the original selector, both complete arms, primary versus
auxiliary success, metric provenance, exact review receipts, source and checkpoint
mutations, real raw/shared and causal histories, actual saved-case replay,
repeated causal records, long prefixes, resource refusals and disk pressure.
The paired fixture carries genuine selected-field histories and a complete field
catalogue, with unused upstream endpoints explicitly mocked as unavailable.
Original fully reviewed cohorts supply the complete original numerical evidence.

Build and verify from the repository root:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
mkdir -p "$TMPDIR"
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
manifest=experiments/convex_stress_shared_compare/Cargo.toml
prlimit --as=68719476736:68719476736 -- cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked \
  --manifest-path "$manifest" -- --test-threads=1
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path "$manifest" --all-targets -- -D warnings
```

After committing, freeze the clean source graph, release CLI, test executable,
protocol and scripts. After task 276 and all declared parents are approved,
freeze a separate original request and launch a fresh output root:

```sh
freeze=/run/media/misha/storage/transformer-runs/freezes/shared-compare-v1
bash experiments/convex_stress_shared_compare/freeze_source.sh \
  "$PWD" "$(git rev-parse HEAD)" "$freeze"
bash "$freeze/freeze_original.sh" "$PWD" "$(cat "$freeze/producer.commit")" \
  "$freeze/request.json"
export SHARED_COMPARE_BINARY=$freeze/shared-compare-v1
out=/run/media/misha/storage/transformer-runs/convex_stress_shared_compare/original-v1
mkdir -p "$(dirname "$out")"
bash "$freeze/run_original.sh" "$freeze/request.json" "$out"
```

The request generator uses the preserved original roots and approval files from
this repository. It does not search for another predecessor or choose by score.
Fresh original review uses the frozen CLI's `replay-case ROOT TASK` for every
completed case and `summarize ROOT` for the full scope, with byte comparisons to
the saved results. Keep generated JSON, tensors, resource logs and source archives
local; commit only this source, protocol, reproduction commands and later authored
findings.
