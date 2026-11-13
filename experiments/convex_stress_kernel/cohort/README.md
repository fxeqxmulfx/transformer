# Original training stage for complete-kernel complex heads

Does the CM34 model admit useful complete kernels on the original stress tasks
when every observed equation is retained? This adapter binds the reviewed
task250 component at `9b10d880` to the original categorical training inputs,
without a held-out evaluator or parent selector. Original execution requires
review of this adapter and a separate prospective freeze. Its fixtures are not
benchmark passes.

## Fixed scope and inputs

`protocol.json` fixes all23 original pair tasks, all55 supported primes in
increasing order, and the unchanged35-task suite. Task names route files and
choose the original data generator; they are never supplied as features or
answer rules to discovery. The inherited native generator checks the original
data fingerprints. Only the ordered training pairs, their raw labels, the full
vocabulary and the original batch size enter the new component.

Every original observation enters the matrix. Inputs beyond the equation cap
receive an explicit refusal; no equation prefix replaces the original data.
The two legitimate shift gauges and full basis checks are unchanged from250.
Earlier one-direction discoveries are not reused as complete kernels. This
stage charges its new discovery work and preserves every certificate, outcome
and cost. Readout compatibility is never a candidate filter.

| Mode | Initialization | Subsequent ordinary CE |
| --- | --- | --- |
| Full | All original observations | One full update at rate1 |
| Mini | All original observations | One original-order epoch, including its tail, at rate1 |
| Forward | All original observations | No CE derivatives |

The source core retains the actual vector range, separate unknown/unmatched
masks, margin24 initialization, fixed defaults, the64N column budget and all
shape/work forecasts. Its prospective56GiB buffer forecast and the user's64GiB
process cap remain unchanged. Stage JSON is streamed with an8GiB per-file
limit, one prime at a time, with a16GiB disk reserve checked before each stage.
A case refused before starting records exit125 and zero new construction.
A write failure retains the partial file, process
status, logs and resource evidence. It is not converted to an available model.

## States, independent checks and costs

Each stage stores its complete original-system discovery report and either the
actual complex initialization and full/mini/forward states or explicit errors
and refusals. Original raw CE, real/imaginary coefficient coordinates, default
and Gram/SVD audits, every update hash and work counter are retained. The small
ledger binds these actual files by hash and size and records each model's hash;
hash agreement does not substitute for numerical replay.

Independent replay regenerates the original input and checks every field
equation, complete basis, vector-state dictionary, complex tensor, gradient,
projected iterate and saved trajectory through the reviewed component. It
also reruns discovery and construction exactly. Discovery and head errors
must reproduce; masks and unknown-target/readout scope flags are checked.
Recorded elapsed times must be finite and nonnegative; replay binds them but
cannot prove a historical physical duration by repeating it.

The complete summary requires a process outcome for every planned case. It
retains partial artifacts and resource logs for failed execution or replay,
all35 task names, all original mode outcomes and every prime stage. Counts of
available endpoints concern training only. Discovery and head construction
are counted separately; preparation, replay, hashing and summary work remain
separately measured costs. A later comparison charges each construction once
across its two complete105-choice parent arms. It must retain the conventional
transformer, depth results, original primary success rules, diagnostic splits
and earlier failures.

Seven native tests cover a complete55-stage multidimensional fixture, forced
raw aliases, an unknown class bank, all three modes, unequal mini tails,
original label/order/batch and scope binding, changed held-out labels and task
names, complete-basis and numerical corruption after recomputing file hashes
and ledgers, equation-cap refusals, process failures, missing stages and forged
replay evidence. The20 reviewed core tests additionally check the actual
derivatives, finite differences, feasible budgets and capacity forecasts.

## Reproduction and prospective freeze

```sh
cd experiments/convex_stress_kernel/cohort
prlimit --as=68719476736:68719476736 -- cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked -- --test-threads=1
prlimit --as=68719476736:68719476736 -- \
  cargo clippy --offline --locked --all-targets -- -D warnings
bash -n run_training.sh freeze_training.sh
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release
```

After250 and251 are approved, call `bash freeze_training.sh REPOSITORY_ROOT
PRODUCER_COMMIT NEW_REQUEST` from the reviewed clean checkout. Preserve the
committed source fingerprint, exact executable, request, protocol and SHA256
digests outside the temporary worktree before any original construction. Set
`KERNEL_COHORT_BINARY` to that executable and call
`bash run_training.sh REQUEST FRESH_ROOT`. The wrapper runs serially, checks
the board gates, sets single-threaded BLAS, enforces64GiB and a16GiB disk
reserve, and retains every exit/resource record. Use the same executable for
`replay-case ROOT TASK` and `summarize ROOT`.

This adapter's input binding, file I/O, resource retention and summary layout
are attributed ports of `convex_stress_refined_quotient/cohort` at `4f210282`.
The main change is fresh complete-kernel discovery instead of importing old
scalar stages. Frozen232/237/243/244/248 protocols and actual states, and the
reviewed250 core, stay unchanged. Generated data and reports remain ignored in
the root experiment's `cohort/runs/` directory. No original run has started
under this adapter yet, and the current benchmark counts remain34/34/33.
