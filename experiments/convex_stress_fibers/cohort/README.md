# Original-task complete-fiber follow-up

Can CM26's categorical fiber equations improve the same nine pair tasks after
the completed tagged follow-up still passed 3/9? This harness ports the tested
task219 producer at `69c83182`. It rebuilds and retains that complete parent,
including all earlier proposals, failures, states, policies and candidates.
The original data, full CE vocabulary, splits and 99% success rule stay fixed.
The new learner receives only training cell/label pairs and a common policy.

| Component | Frozen policy |
| --- | --- |
| Complete tagged parent | All CM15/21–25 candidates, original input and earlier proposal; parent precedes every new candidate on observed CE ties within 1e-12. |
| Equations | Prime domain equal to input category count; four observed group sizes from 2..8, largest first; first 4096 constraints per size, plus three categorical phase gauges. |
| Exact elimination | At most 500 million scalar updates and 2 million row operations per guess; retain original system, every operation, final matrix, affine basis and caps. |
| Enumeration | Nullity at most three, 65536 tuples per guess, 16 distinct admitted output tables; retain every inspected tuple and concrete rejection/duplicate witness. |
| Admission | Both phase maps injective; original observations fit one fully covered decoder; every size-k group's actual decoder fiber has cardinality k and the proposed common mass. Check even groups beyond the equation cap. |
| Evaluation | Seal observed-only selection and all artifact digests before measuring every admitted candidate on unchanged original splits. Reconstruct all parents, equations, states and work for replay. |

The independent checker verifies full invertible row operations and every
original endpoint equation; it reconstructs actual complete-fiber images and
latent coverage from the original observations. Every admitted model receives
all-frequency raw DFT/SVD and ordinary full-vocabulary CE audits. The new model
and decoder are proposals in the convex spectral hull; these checks do not
prove unknown unseen labels or global search success.

Four harness tests pass: default policies and frozen shared-data fingerprint;
complete fixture replay and earlier-parent CE ties; rehashed tampering with
balanced, tagged and fiber ledgers; and explicit refusals/input/byte limits and
all 27 mode references when all nine processes fail. The separate native core
has ten checks, including 793 exhaustively enumerated finite systems and
masked pair/triple/four-fiber recovery, incomplete-fiber refusals and ambiguity.

The combined parent/new report forecast must fit within 48 GiB, each artifact
within 512 MiB, and each numerical process within the user's 64 GiB limit.
Cases run serially with one BLAS thread and a 16 GiB disk reserve. Every
process failure is retained while later planned cases continue. Generated
evidence remains ignored and rooted outside temporary worktrees.

```sh
cd experiments/convex_stress_fibers/cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --locked --release
cd ../../..
FIBERS_PRODUCER_COMMIT=<committed-producer> FIBERS_BINARY=<frozen-binary> \
  bash experiments/convex_stress_fibers/cohort/run_cohort.sh \
  experiments/convex_stress_fibers/runs/original-v1
```

Commit and freeze the tested producer, complete source fingerprint, binary,
protocol and launcher before preparing original data. The parent summary hash
is bound in the protocol. All original parent reports must compare exactly to
their retained copies. No original fiber score exists yet. The 27 mode records
refer to nine forward constructions; they are not three optimizer runs. All35
short/long tasks, full/mini/forward comparisons, stronger earlier families and
the conventional transformer remain required for final integration.
