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

The library guard bounds its own numerical buffers. Original-data execution
must additionally use an 8 GiB process cap, run serially, freeze a protocol and
independently replay saved predictions before reporting stress results.

## Findings and checks

Six native tests pass. A shuffled eleven-state fixture with a many-to-one
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

```sh
cargo fmt --manifest-path experiments/convex_stress_decode/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 cargo test --locked --manifest-path experiments/convex_stress_decode/Cargo.toml -- --test-threads=1
cargo clippy --locked --manifest-path experiments/convex_stress_decode/Cargo.toml --all-targets -- -D warnings
```

No original stress run, full/mini/forward comparison, transformer comparison
or full-suite pass is claimed by this prototype. The previously frozen cohorts
continue unchanged.
