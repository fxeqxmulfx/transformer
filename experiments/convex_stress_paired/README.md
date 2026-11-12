# Equal probabilities on opposite latent states

Does a fixed decoder that depends on the vocabulary alone, optimized by ordinary
cross-entropy over a convex set of latent probability tables, complete tasks
that the bijective class codings and the searched decoders of the preceding
cohorts did not? Lean CM16 (`3447d663`,
[`review/flat_spectrum.md`](../../review/flat_spectrum.md)) comes first. A latent
table on the cyclic group with equal probabilities on opposite states is a
linear restriction of the CM15 domain; for a decoder with `d(-z) = d(z)` it
loses no prediction (`pairedFeasible_iff`). The symmetric phase table of any
row and column phases meets the operator and nuclear budgets `N`
(`pairedPhase_budget`) and, when it fits the observations, minimizes ordinary
CE over the whole domain (`pairedPhase_minimum`). The same file proves that two
such minima can disagree at an unseen cell under connected matching coverage
(`paired_unseen_ambiguity`). Nothing proved says that an unseen answer is
recovered, and nothing proved says that the iterations below converge.

## Domain and decoder

Three sizes are separate in the code: the input side `N`, the latent orbits
(latent states `q = 2 * orbits - 1`) and the original output classes `V`.

| Quantity | Here |
| --- | --- |
| Learned coordinates | One probability vector over the `q` latent states per ordered input pair, equal on `z` and `-z`; stored by halves, one number per orbit `{c, -c}` |
| Constraints | Probability rows; operator and nuclear norm at most `N` for the moment of every nonconstant frequency. The constant moment of probability rows is the all-ones matrix, of nuclear norm `N` (`constant_moment_nuclear`) and of the same operator norm, so this is the Lean domain at `R = B = N` |
| Decoder | Fixed and even, given by the class of each orbit; summed over fibers. The cohort uses `foldDecoder`: `q = 2V - 1`, the states `c` and `-c` decode to the class `c` |
| What the decoder reads | The number of classes and their given order, as every Fourier code of the preceding cohorts does. No training label, no input token |
| Logits | The decoded probabilities at scale one, over all `V` classes; a class without a latent state stays in the softmax |
| Inputs | Table indices only; no arithmetic function of token values |

With 97 classes and the frozen cap of 193 latent states, an even decoder of an
odd cyclic group reaches every class only with 193 states and one orbit per
class. The cohort matches orbits to classes in the given class order and
proposes nothing else. The library accepts any even decoder of an odd cyclic
group, with larger fibers or with classes that no state reaches; the tests use
both.

An even table has real moments that are even in the frequency, so a real
`orbits x orbits` cosine matrix replaces the complex transform. In the Euclidean
metric of the full table, where an orbit of two states counts twice, the two
projections are exact: rows onto even probability vectors by the simplex
projection of the row written out state by state, and each cosine moment onto
the two budgets by projecting its singular values.

## How the runs differ

| Arm | Update | Derivatives |
| --- | --- | --- |
| `full_batch` | `x = P_A(z)`, `z += P_B(2x - z - g) - x` with `g` the CE gradient of all observed cells at `x` | Every observation, every update |
| `minibatch` | The same, but only the 256 selected observations refresh their part of a retained `g` (incremental aggregated gradient, the original shuffle and tails) | One batch per update |
| `forward_only` | `A` also confines each observed row to the fiber of its label; `g = 0` | None |

The latent gradient is the class gradient read through the decoder,
`g(z) = (softmax(Q) - e_y)(d(z))`. The step is one whenever no class has more
than two latent states, as for the fold decoder. All arms start from the
uniform latent table, observe every 19 updates as the original suite row does,
and stop at the first positive-update observation within `1e-8` of the common
CE lower bound `ln(e + V - 1) - 1`, or at the caps of the CM12 cohort: 128, 512
and 128 updates. At each observation the iterate is mixed with the uniform
latent table until both budgets hold; that mixture is what is evaluated and
saved.

[`protocol.json`](protocol.json) freezes 24 cells: the three arms on the eight
unchanged datasets of the preceding follow-ups (permuted addition, addition,
subtraction, division, multiplication, sum-cube, sum-power-four and absolute
difference; modulus 97, 4704 training pairs each). They are positive controls
and known failures of earlier candidates. The model receives categorical pairs
and labels, never the task name.

Before any buffer is allocated the three sizes are validated and an explicit
bound on the numerical buffers (273 MB for these tasks) is checked against
2 GiB. The launcher runs one process at a time under an 8 GiB address-space
cap with single-thread BLAS and a 16 GiB disk reserve, never overwrites a
cohort and never resumes a cell.

```sh
cargo build --release --locked --manifest-path experiments/convex_stress_paired/Cargo.toml
PAIRED_PRODUCER_COMMIT=$(git rev-parse HEAD) bash experiments/convex_stress_paired/run_cohort.sh \
  experiments/convex_stress_paired/runs/cohort-v1
```

Every cell is replayed by a second process. It restores the saved splitting
point, reproduces the saved latent and decoded tables exactly, and rechecks the
schedule, exposure and cost records. A raw audit then uses none of the
producer's decoder, transform, projection, loss or metric code: it reflects the
saved half table to all 193 latent states, recomputes the decoded table, builds
the Fourier matrix of each of the 192 nonconstant frequencies by direct complex
sums before a shared LAPACK SVD, and recomputes the full-vocabulary CE and every
split metric. The spectral checks are floating-point, not interval certificates.

The standalone summary selects nothing and drops nothing. Integration with the
earlier candidates is a separate step that requires the complete, unchanged
decoded-phase cohort, whose summary retains every earlier candidate:

```sh
experiments/convex_stress_paired/target/release/convex-stress-paired integrate \
  experiments/convex_stress_paired/runs/cohort-v1 experiments/convex_stress_decode/runs/cohort-v1
```

For each of its 24 task and mode selections it orders `[paired candidate,
earlier selection]` by original training CE: the first within `1e-8` of the
lower bound, otherwise the lower one. The earlier choice is retained whole. No
held-out metric chooses, stops or sets a budget; the 99% sequence-accuracy rule
is only reported.

## Findings and checks

Fifteen native tests pass. The cosine moments equal the complex Fourier sums
of the table written out state by state. On a table whose side equals the
number of latent states, the spectral projection agrees with the complex CM5
projector applied to that written-out table, which knows nothing of the
symmetry. The latent gradient matches finite differences of the
full-vocabulary CE through a decoder with a merged fiber and two unreachable
classes. The Lean witnesses are reproduced: the folded three-class table is
feasible at the budgets 3 and reaches the CE lower bound, and the two tables of
`paired_unseen_ambiguity` share the observed fit and differ at the unseen cell.
A probability table whose first moment has nuclear norm 4.24 is rejected by the
raw audit alone, as are altered decoded entries, row masses, metrics and
reported norms. Runner checks replay all three arms on the folded fixture with
one withheld cell and reject forged tables, splitting points, schedules, costs
and metrics even when their digests are recomputed.

Before the freeze the solver was run on that fixture and on one synthetic table
of the original sizes: random row and column phases in the 193 latent states,
decoded by the fold, half of the 9409 cells observed. All three arms reached
the CE lower bound within 57 updates and predicted the 4705 withheld cells, at
about 1.1 s per update on one thread. This table is realizable in the domain by
construction; it measures the solver, not the original tasks.

```sh
cargo fmt --manifest-path experiments/convex_stress_paired/Cargo.toml --check
OPENBLAS_NUM_THREADS=1 cargo test --locked --manifest-path experiments/convex_stress_paired/Cargo.toml -- --test-threads=1
cargo clippy --locked --manifest-path experiments/convex_stress_paired/Cargo.toml --all-targets -- -D warnings
PAIRED_MODE=full PAIRED_UPDATES=57 OPENBLAS_NUM_THREADS=1 cargo test --locked \
  --manifest-path experiments/convex_stress_paired/Cargo.toml -- --ignored --nocapture synthetic
```

No update had been computed on the original data at this freeze. No
original-task score, comparison with earlier candidates, transformer comparison
or full-suite pass is claimed here.
