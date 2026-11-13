# Training stage from preserved categorical discoveries

This adapter binds the reviewed CM31/CM32 component at `c55fe210` to the original
pair-task training data. It constructs and independently replays refined heads
without scoring a held-out split or selecting against a previous model. Original
execution was frozen after task242 review and is now complete. No new benchmark
success follows from the adapter's synthetic tests or training-only results.

## Complete original training results

Task243 retains every original stage: all23 cases run and independently replay,
with1265 stage records and zero process failures. Six stages admit full, mini
and forward modes, all at modulus97: arithmetic-cube-plus, arithmetic-sub,
arithmetic-add, arithmetic-square-sum, arithmetic-half-sum and
arithmetic-perm-add. Each of their18 endpoints has zero observed raw errors and
CE approximately3.62e-9. All three modes start from the already fitting refined
initializer; these losses do not demonstrate optimizer convergence.

| Outcome | Stages | Interpretation |
| --- | ---: | --- |
| Actual full/mini/forward states available | 6 | Original observations fit; no held-out evaluation |
| Fourier shape cap | 331 | Explicit constructor refusal retained |
| Source admitted no latent code | 692 | Original discovery refusal retained |
| Source stage returned an error | 236 | Original failed stage retained |

Fibonacci has no admitted source code and gains no endpoint. Construction costs
185.750755602 recorded seconds and120 CE gradient batches (six full steps and
six19-batch mini epochs); forward uses zero gradients. These constructor totals
exclude inherited source discovery and the separately measured preparation,
replay and summary work. The latter takes10m46.84s for the original summary;
run/replay peak RSS is272068KiB. All processes use the64GiB cap. Original
summary and inventory files remain under `cohort/runs/training-v1` in the root.

The frozen producer is `4f210282986efaa7ce2a9d2c491588ba35a2255f`, source
fingerprint `f044d46f75bbf84e6f0b10addcac5ba28cb3987bdb604d4f6310ae12f7ccdf76`
and executable digest
`badc5dc48abd5036a5889374dc921067d7b5398025560c79e242faff00fa73eb`.
The original summary digest is
`e52725e3bd4ce92c9e5841892530e1d2729b07f43f47f3056c1c026cead934ef`,
and inventory digest is
`113d440c9d1ad17af420a047349c3c2d3ef4ead0ad3a8cd39dfd9f114e5512b5`.
The request, protocol, frozen executable and hashes remain in
`.kanban/artifacts/243/freeze-v1`. Rebuild the summary with that executable's
`summarize` command and compare it byte for byte; `replay-case` reconstructs
the actual numerical states for an individual case. These are training
availability counts, not additions to the35-task success total. The complete
comparison still requires the reviewed parent chain.

## Frozen inputs and complete scope

`protocol.json` pins the task232 producer `49105730`, its source fingerprint and
executable digest, all23 original pair tasks and the unchanged35-task suite. The
request also pins the actual source inventory and executable files. Preparation
requires a process outcome for every source case and, for successful cases, its
complete matching replay evidence. It can run after the catalogue closes and
before the separate comparison review; it cannot consume an unfinished case.

Each source case retains every stage file, its hash and size, the complete
55-prime ledger, all discovery/model work counters and stage times, process
logs, resource observations and any partial artifacts. Failed source cases get
an explicit source refusal with zero new models. A source stage that returned
an error or admitted no code also retains an explicit refusal. The adapter does
not rerun discovery or discard a stage using its readout compatibility or score.

The inherited native generator verifies the original data fingerprints. Only
the ordered training pairs, unchanged raw labels, original vocabulary and
original minibatch size enter the new component. They must exactly match the
saved discovery input. Task names route files; they supply no answer rule.
The three fixed policies remain one full update, one original minibatch epoch
including its tail, and zero forward CE gradients, after complete-data
initialization. These differ prospectively from task232's16/256 update policies.
The32B per-mode work forecast and64GiB address-space cap are unchanged from241.

## States, replay and costs

Every eligible stage writes the actual initialization, three mode outcomes or
individual refusals, complex coefficients, raw CE audits, Gram/SVD evidence,
gradient work, and hashes of every updated table and head. A stage error is
retained with its reason and measured duration. Replay reconstructs the same
original input and policy, rehashes the complete source case, independently
checks all complex states/updates through the component, and exactly reruns
construction. It also replays refusals. File and summary hashes alone do not
substitute for the numerical replay.

The summary requires every planned process outcome. It retains partial files
and resource logs for failed runs/replays and reports counts of available
training endpoints without interpreting them as benchmark passes. Source
discovery, validation, preparation, construction and replay remain distinct
costs. There is zero *new* discovery work; the old discovery and replay costs
remain in the pinned source records. A later comparison charges construction
once across both complete parent arms. Elapsed times are recorded observations;
replay verifies their bindings and finite nonnegative values, not their physical
truth by reproducing wall-clock duration.

Seven adapter tests cover a complete actual55-stage synthetic catalogue and its
replay, a failed source, a new process failure, retained partial work, original
input/order/batch and scope mutations, source/executable/replay corruption, and
numerical corruption after recomputing both hashes and the cheap retained ledger.
The complete fixture keeps35-task metadata and all three modes. A separate test
checks the fixed source pin against the current unchanged parent source.
The14 component tests cover derivative correctness, feasible defaults, aliases,
serialization and per-mode cap refusals. No original model is trained by these
fixtures.

## Reproduction

```sh
cd experiments/convex_stress_refined_quotient/cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked -- --test-threads=1
prlimit --as=68719476736:68719476736 -- \
  cargo clippy --offline --locked --all-targets -- -D warnings
bash -n run_training.sh freeze_training.sh
cargo build --offline --locked --release
```

After task242 approval, use `bash freeze_training.sh REPOSITORY_ROOT
PRODUCER_COMMIT SOURCE_ROOT SOURCE_BINARY NEW_REQUEST` to create the request.
Preserve the committed source fingerprint, exact executable and SHA256 digest,
request and protocol together outside the worktree before starting. Set
`REFINED_QUOTIENT_BINARY` to that frozen executable and run
`bash run_training.sh REQUEST FRESH_ROOT`. The wrapper checks board approval,
runs serially with single-threaded BLAS and64GiB limits, keeps a16GiB disk reserve,
and saves each exit/resource record before producing the complete summary.
Use the identical executable for `replay-case ROOT TASK` and `summarize ROOT`.
All generated evidence stays ignored in the root's experiment `runs/` directory.

This stage contains no held-out evaluator or candidate selector. Comparison
integration remains a separate task requiring the reviewed232 parent and other
relevant approvals. It must retain both105-choice parent arms, all35 tasks,
the conventional transformer,336 depth outcomes, causal237 evidence, failures
and costs. Use the original declared primary success rule; keep all other splits
as diagnostics. DiffGPT105/105 across seeds0/1/2 remains the existing separate
reference. The current34/35 full,34/35 mini and33/35 forward counts are unchanged.
