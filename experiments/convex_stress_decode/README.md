# Categorical decoding of bounded latent spectra

Can a learned many-to-one decoder recover tasks whose output classes do not
admit an injective phase code? Lean CM15 (`b73b3e3f`,
[`review/flat_spectrum.md`](../../review/flat_spectrum.md)) comes first: fixed
decoding is linear, preserves probability rows and convex ordinary
cross-entropy, and the hull over all decoders remains convex. The spectral
bounds apply to latent moments. They need not hold for decoded output moments.
Any feasible exact observed fit attains the global CE minimum in this hull.
The proved three-cell counterexample also has two such minima with different
unseen answers, despite connected matching coverage.

## Native proposal

`Model` stores a one-hot latent table and its materialized decoded table, along
with the phases and decoder needed to check their construction. The latent
phase at a cell is the sum of learned categorical row and column phases. Model
evaluation only reads the saved decoded table. It uses the full original class
vocabulary with unit-scale logits; classes absent from training remain in the
softmax denominator. No input-token arithmetic answer formula is supplied.

The prototype search uses only observed pairs and categorical labels:

1. Give observed classes random complex codes, in first-occurrence order, and
   center their observed sketch. Power iteration proposes leading singular
   vectors, whose angle orders propose permutations of latent phases.
2. Fit a many-to-one decoder by observed counts. Alternately solve exact
   minimum-cost row and column assignments and refit that decoder. At unit
   scale, wrong one-hot predictions contribute exactly one unit of ordinary
   CE excess each, so these coordinate updates minimize that same objective.
3. Retain the lowest-error candidate with observed coverage of every proposed
   latent state. Stop on exact observed fit, repeated state or the preset cap.
   Record every attempted sketch, including failed phase proposals.

This is a heuristic search for vertices of the proved convex prediction class;
it is not an efficient global optimization oracle. A one-class observed sketch
is an explicit failure of this constructor even when a fitting model exists.
No continuous full-batch or minibatch optimizer is implemented here yet.

| Quantity | Prototype limit or interpretation |
| --- | --- |
| Input side, latent states, original output vocabulary | Each between 2 and 193 |
| Search latent count | Equal to input side; row/column proposals are permutations |
| Random code trials | At most 64 |
| Power iterations / assignment refinement rounds | At most 128 each per trial |
| Retained endpoint | Best covered candidate; unsuccessful candidates are explicit |
| Spectral audit | Reconstruct every Fourier matrix from saved latent labels and inspect its LAPACK singular values, one matrix at a time |
| Memory guard | Conservative established dense-buffer bound, at most 2 GiB; actual compact storage is quadratic |
| Cost records | Validation/class/sketch/template/verification observations, label comparisons, assignment sweeps, power iterations and all matrix-vector products, including failed ones |

The library guard bounds its own numerical buffers. The original-data launcher
additionally enforces an 8 GiB process cap, serial execution, single-thread BLAS
and a 16 GiB disk reserve. Its bounded JSON reader and one-frequency-at-a-time
audit avoid materializing a bank of dense Fourier matrices.

## Frozen original-data follow-up

The original eight constructions and their replays finished, but summary
verification stopped the downstream queues: preparation rebuilt the phase
parent with a relative path, while later verification used its saved absolute
root. Exactly nine `model_path` strings differed; all scores, choices, hashes,
costs and other fields were equal. Original failure records and artifacts stay
unchanged. `recover_paths.sh` stages a separate canonical phase view whose
symlinked artifacts resolve to the same original files, regenerates its summary
with the frozen binary, checks every other field exactly, and verifies a second
summary byte for byte. It rejects equal-byte files at a different actual path.
The recovery uses no new numerical producer or model policy.

```sh
bash experiments/convex_stress_decode/recover_paths_test.sh
bash experiments/convex_stress_decode/recover_paths.sh view \
  /absolute/original/phase /absolute/fresh/phase-view /absolute/frozen/phase-binary
```

The frozen decoded launcher must then receive absolute paths for both its fresh
output and canonical parent. Preserve the first failed decoded run; a fresh
eight-case construction/replay may proceed with its unchanged source, protocol
and binary, followed by the unchanged transformer and full-suite comparisons.
The canonical view and all downstream numerical processes are capped at or
below the user's 64 GiB. This path repair is not an accuracy improvement.

The recovery at`5e1ff865` is complete. All eight cases reconstructed and replayed
with byte-identical proposals and models. Every non-timing/non-parent field in
their results is unchanged; seven returned endpoints and one refused proposal
remain. The sealed comparison has15/24 primary and all-split passes, three
exact-fit proposals, and peak replay/run RSS12844KiB. Its summary SHA256 is
`f72a3398f92d196ccbdf840d5efafd34128a2fcb67891f3d3103adc5f7d78b4c`.
The same path-view helper, using each producer's `summarize` command, also
repairs paired and certified parents without retraining. This unblocks the
27-choice general-bank integration and the original35-task transformer queue.
Rooted evidence is`runs/cohort-canonical-v1` and`.kanban/artifacts/221`; the
first failed run and both failed comparison gates remain preserved.

The complete eight-task comparison has five passes in each of its full-batch,
minibatch and forward-only parent-selection arms. These are 24 comparisons
with seven newly constructed endpoints and one explicit refusal; the new
constructor itself uses no CE gradient. The canonical recovery closes the
original-data obligation of task192 and supplies the decoded parent for
task195's paired integration. The unchanged first run still records exit1;
successful recovery is separate evidence, not an overwritten exit status.

[`protocol.json`](protocol.json) freezes eight cases on the same unchanged
datasets as the preceding class-coding/phase cohorts: permuted addition,
addition, subtraction, division, multiplication, sum-cube, sum-power-four and
absolute difference. They include positive controls and known failures. The
model receives categorical training pairs and labels, not the task name or an
arithmetic operation. This constructor does not call the earlier bijective
class-code discovery, which rejects unobserved output classes.

All cases use the previously tested 16 random code trials, 64 power steps,
24 assignment refinement rounds and seed 73. Save the best covered endpoint
even if it fails to fit; save an explicit rejection if none exists. Save every
trial and its cost, full original metrics, and both compact probability tables.
An independent process rebuilds the proposal from training labels, checks its
exact coordinates and trace, reconstructs each latent Fourier matrix directly
from saved labels, and recomputes original CE and split metrics. A summary
requires all eight sealed results, with no omitted failures.

Prepare only after the complete, unchanged 16-case phase cohort and its
cold/graph parents. For each of its 24 task/mode selections, compare `[new
decoded candidate, previous selected candidate]` by original training CE:
first candidate within `1e-8` of the common ideal, otherwise the minimum.
Preserve the previous selection and every earlier candidate in the report.
The new endpoint is constructed once and referenced in each full-batch,
minibatch and forward-only comparison, with its full standalone cost. These
references are not three gradient training runs. An ideal training fit may
still cause an accuracy regression; all original held-out metrics remain in
the comparison, with the unchanged 99% sequence-accuracy success rule.

```sh
cargo build --release --locked --manifest-path experiments/convex_stress_decode/Cargo.toml
DECODE_PRODUCER_COMMIT=$(git rev-parse HEAD) bash experiments/convex_stress_decode/run_cohort.sh \
  /absolute/fresh/decoded-run \
  /absolute/canonical/phase-parent
```

The run destination must be fresh. Preserve incomplete attempts instead of
silently restarting them. Generated evidence remains ignored in the root
checkout's linked runs directory. No original-data score has been observed at
this protocol freeze.

## Findings and checks

Ten native tests pass. A shuffled eleven-state fixture with a many-to-one
decoder is recovered, including its withheld cells. Four of its eleven output
classes never occur, but all eleven remain in CE. Renaming output classes
preserves the proposal trace and appropriately renames every saved prediction.
This deliberately small fixture is a correctness check, not original-task
generalization evidence or a guarantee for arbitrary missing cells.

Other checks reproduce the CM15 capacity/ambiguity witness, detect corrupted
saved tables, compare assignment costs against exhaustive small permutations,
verify the full-vocabulary CE excess, and retain bounded failure records.
The decoded witness exceeds the old direct binary nuclear budget while its
latent moments satisfy their budget. This distinguishes an extended feasible
class from an invalid claim that decoding preserves every original moment
bound.

End-to-end runner checks independently replay successful, imperfect and absent
proposals. Altered saved coordinates, cost ledgers or metrics are rejected even
when their digests are recomputed. A missing parent keeps preparation stopped,
and oversized saved input is rejected before reading it into memory.

```sh
cargo fmt --manifest-path experiments/convex_stress_decode/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 cargo test --locked --manifest-path experiments/convex_stress_decode/Cargo.toml -- --test-threads=1
cargo clippy --locked --manifest-path experiments/convex_stress_decode/Cargo.toml --all-targets -- -D warnings
```

The fixture checks establish implementation behavior. The separate original
cohort above supplies the eight-task, 24-choice evidence. For a fresh review,
use the frozen binary's `replay-cell ROOT CELL` for every entry of `cells.txt`,
compare each returned seal with `ROOT/CELL.replay.json`, then regenerate
`summarize ROOT` and require byte identity with `ROOT/summary.json`. Use the
absolute canonical root, one process at a time, with
`prlimit --as=68719476736:68719476736 --` and single-thread BLAS. This follow-up
does not establish a full-suite pass or a transformer comparison; the complete
35-task comparison is reported separately in `EXPERIMENT_PLAN.md`.
