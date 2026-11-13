# Training stage from preserved categorical discoveries

This adapter binds the reviewed CM31/CM32 component at `c55fe210` to the original
pair-task training data. It constructs and independently replays refined heads
without scoring a held-out split or selecting against a previous model. Original
execution is a separate frozen follow-up after task242 review. No new benchmark
success follows from the adapter's synthetic tests.

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
