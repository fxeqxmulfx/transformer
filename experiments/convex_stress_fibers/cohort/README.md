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

The tested producer, complete source fingerprint, binary, protocol and launcher
were frozen before preparing original data. The parent summary hash is bound
in the protocol. All original parent reports compare exactly to their retained
copies. The 27 mode records
refer to nine forward constructions; they are not three optimizer runs. All35
short/long tasks, full/mini/forward comparisons, stronger earlier families and
the conventional transformer remain required for final integration.

## Frozen original-v1 findings

Producer `0e0c2d66cb88ba3bf4d982810505d43237ba8eba`, source fingerprint
`660de1fd395b9d82583f16c37fa132b32f59eb5beb0ea7a146a4b4fa1b6ebd81`, and binary
`ff19a7a111482184bdb8353b529221b8519e4448c5800be679d2914119d6aaa9` completed
all nine cases and their full replays without process failures. Every input,
earlier proposal and complete tagged parent is unchanged. The summary digest is
`d6b12e04b9bb7c08de7d8cba81619ee3534dc7f692d51aa0229819f0050d5777`.

| Task | Selected candidate | Observed errors | Primary accuracy |
| --- | --- | ---: | ---: |
| Permuted addition | Earlier proposal | 0 / 4704 | 100% |
| Addition | Earlier proposal | 0 / 4704 | 100% |
| Subtraction | Earlier proposal | 0 / 4704 | 100% |
| Division | Earlier tagged candidate | 1793 / 4656 | 55.67% |
| Multiplication | Earlier proposal | 1198 / 4704 | 72.60% |
| Sum-cube | New fiber candidate | 0 / 4704 | 100% |
| Sum-power-four | New fiber candidate | 0 / 4704 | 100% |
| Absolute difference | Earlier proposal | 4061 / 4704 | 3.78% |
| Blind absolute difference | Earlier tagged candidate | 1671 / 4704 | 61.40% |

The result improves from 3/9 to **5/9 primary and all-split passes**. Sum-cube
and sum-power-four each contribute one new model, selected by observed CE
before held-out evaluation. Their ordinary unit-logit CE is
3.5922701555123604. Both recover all 4705 held-out cells, completing their
9409-cell tables. No arithmetic interpretation of the class names enters
the constructor. Sum-cube's size-three system has one free coordinate and
enumerates all 97 coefficients; fourth-power's size-four system has two free
coordinates and enumerates all 9409 pairs. Other guessed sizes and failed
systems remain in their reports.

Across nine cases, the new fiber stages perform 49,952,939 scalar updates,
725,770 row operations and 9507 coefficient trials; two admitted models receive
194 raw frequency audits. They perform zero CE gradient evaluations. Total
construction time, including rebuilding every older parent, is 903.440 seconds;
peak measured process RSS across construction and replay is 175,388 KiB
(171.3 MiB), under the 64 GiB cap. Existing stronger division, multiplication
and absolute-difference families remain required in the full comparison. This
standalone cohort is not a full35 result or three separate optimizer runs.
