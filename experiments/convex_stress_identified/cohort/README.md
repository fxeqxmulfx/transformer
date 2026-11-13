# Original comparison with fixed-view identification certificates

This separate producer applies the already declared CE/certificate rule to all
23 original pair-table tasks. It reuses every task222 modular proposal without
changing its predictions. The complete earlier105-choice comparison is pinned
to producer `f661532eb606bf450309033d5276773ea45b6037`, source fingerprint
`8f2dc75151dd0246eb5609b872af6361db0c047e65ce8adfd2806cea39cda2e6`, and summary
`5d3aa99d91efdab222fc3bd5c1c8ea68f16c8f632808b22f84c62213f4355139`.
Preparation copies that summary byte for byte, verifies its complete original
replays and proposal hashes, and regenerates all original input fingerprints.
No original task has been certified by this producer yet.

CM28 and Lean commits `4fee3d09`, `9cd95708`, `6a8fb49e` precede native code.
The constructor receives only saved observed data, code proposals and fixed
view maps. It checks the complete modular/strip report and raw spectral/CE
audits, then certifies the actual anchored matrix through bounded ring row
operations. Missing or padded axes are refused. Every admitted and refused
view stays recorded. Identification remains conditional on a fixed bijective
cyclic interior and the view's constant-strip structure. It does not identify
arbitrary ground truth over the larger convex prediction hull.

Each mode starts with its earlier selected model, then the saved modular
candidates. A CE decrease greater than1e-12 takes priority. Within a1e-12 CE tie,
a verified fixed-view certificate takes priority; otherwise earlier order is
kept. The same certified model hash also certifies an earlier selection.
Neither held-out scores nor primary split names enter this decision. The
runner seals training-only choices before recomputing any held-out metric.
This structural preference is an explicit inductive assumption; the old
earlier-first run and its wrong equal-loss completions remain immutable.

All23 tasks must reach a terminal outcome. A certificate refusal preserves
the candidate as an uncertified proposal; a failed run or replay retains its
three complete earlier choices. The summary preserves all105 choices, all
earlier candidates/failures, and the12 tasks outside this certificate's scope.
One certification is charged per task with three mode references and zero
CE gradients. Prior construction costs remain in the complete parent.
Full replay repeats the old modular constructor and every new certificate,
checks all original artifacts, and recomputes raw spectra, CE and every split.
Retained depth and the fresh conventional transformer remain required.

Ten core, five adapter and six cohort tests pass on synthetic inputs. The
cohort tests cover the exact original protocol and data-source fingerprint,
CE-tied fixed-view selection, complete old/new replay, rehashed certificate
tampering, altered original inputs, all23 failures preserving all105 choices,
a successful three-mode replacement, byte limits and immutable-parent checks.
These tests do not establish any new original benchmark pass.

Freeze the committed source fingerprint, binary, protocol and launcher before
original preparation. Native Rust/shell processes are serial with one BLAS
thread and a64GiB address-space limit. Artifacts are capped at2GiB, combined
parent/certificate/parser forecasting at56GiB, and disk reserve at16GiB.
The certificate defaults retain4096 equations and permit two billion scalar
updates, two million row operations, twenty million pivot/combination tests
and one million inverse tests. All original observations are checked even
beyond the elimination equation cap. Generated evidence stays local/ignored.

```sh
cd experiments/convex_stress_identified/cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo build --release --locked
cd ../../..
IDENTIFIED_PRODUCER_COMMIT=<committed-producer> IDENTIFIED_BINARY=<frozen-binary> \
  bash experiments/convex_stress_identified/cohort/run_cohort.sh \
  experiments/convex_stress_identified/runs/original-v1 \
  experiments/convex_stress_modular/runs/original-v1
```
