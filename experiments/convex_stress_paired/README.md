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
(`paired_unseen_ambiguity`). Lean CM18 (`5c0fe9bc`) proves that a certificate
checked on training samples alone fixes the decoded answer of every completion
at the cells it reaches (`accepted_decoding`), and that this program can still
have a global CE minimum with a wrong answer at such a cell
(`budget_wrong_minimum`). Nothing proved says which minimum the iterations
below approach, or that they converge.

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
difference; modulus 97, 4704 training pairs each, 4656 for division). They
are positive controls and known failures of earlier candidates. The model
receives categorical pairs and labels, never the task name.

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

No update had been computed on the original data at this freeze.

## Original data

The frozen cohort ran once, on 2026-11-12, serially, from the binary built at
`a56cf4f0` (source fingerprint `40aa7b2c…7a4439`). The 24 producer processes
and the 24 replay processes all ended with exit status 0. Every cell was
replayed and passed the raw audit: largest decoding defect 0, largest
imaginary part of a moment `6.5e-16`. The audit recomputed the operator and
nuclear norms of all 192 nonconstant frequencies of every saved table, accepted
them against the budgets 97 at relative tolerance `1e-9`, and found them within
`1e-7` of the producers' records, whose largest nuclear norm is
`96.9999999998` and largest operator norm 79.2. Each task has two splits,
4704 training and 4705 validation pairs (4656 and 4656 for division); the
accuracies are sequence accuracies in percent. `CE excess` is the mean
training CE above the lower bound. `Mixture` is the weight of the iterate in
the saved table, to four digits; below one, the iterate exceeded a budget and
the rest of the weight is the uniform table.

| Task | Arm | Stop | Updates | CE excess | Train | Validation | Mixture | Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arithmetic-perm-add` | `full_batch` | cap | 128 | 0.692541 | 100 | 0.64 | 0.9996 | 176 |
| `arithmetic-perm-add` | `minibatch` | cap | 512 | 0.692437 | 100 | 0.64 | 1 | 684 |
| `arithmetic-perm-add` | `forward_only` | cap | 128 | 0.745167 | 100 | 1.13 | 0.2401 | 168 |
| `arithmetic-add` | `full_batch` | cap | 128 | 0.208610 | 96.92 | 95.26 | 0.9807 | 147 |
| `arithmetic-add` | `minibatch` | cap | 512 | 0.201054 | 96.92 | 95.26 | 0.9927 | 580 |
| `arithmetic-add` | `forward_only` | cap | 128 | 0.644249 | 100 | 96.54 | 0.3424 | 153 |
| `arithmetic-sub` | `full_batch` | cap | 128 | 0.206311 | 96.28 | 94.79 | 0.9806 | 156 |
| `arithmetic-sub` | `minibatch` | cap | 512 | 0.198649 | 96.17 | 94.79 | 0.9927 | 588 |
| `arithmetic-sub` | `forward_only` | cap | 128 | 0.641786 | 100 | 96.11 | 0.3449 | 156 |
| `arithmetic-div` | `full_batch` | cap | 128 | 0.687460 | 100 | 1.03 | 0.9987 | 167 |
| `arithmetic-div` | `minibatch` | cap | 512 | 0.687096 | 100 | 1.01 | 1 | 651 |
| `arithmetic-div` | `forward_only` | cap | 128 | 0.746273 | 100 | 1.12 | 0.2390 | 159 |
| `arithmetic-mul` | `full_batch` | cap | 128 | 0.689254 | 100 | 0.83 | 0.9986 | 164 |
| `arithmetic-mul` | `minibatch` | cap | 512 | 0.688868 | 100 | 0.83 | 1 | 650 |
| `arithmetic-mul` | `forward_only` | cap | 128 | 0.748372 | 100 | 0.79 | 0.2369 | 168 |
| `arithmetic-sum-cube` | `full_batch` | cap | 128 | 0.692780 | 100 | 0.55 | 0.9995 | 173 |
| `arithmetic-sum-cube` | `minibatch` | cap | 512 | 0.692647 | 100 | 0.55 | 1 | 690 |
| `arithmetic-sum-cube` | `forward_only` | cap | 128 | 0.766034 | 100 | 0.32 | 0.2189 | 170 |
| `arithmetic-sum-pow4` | `full_batch` | cap | 128 | 0.683539 | 100 | 24.34 | 0.9992 | 176 |
| `arithmetic-sum-pow4` | `minibatch` | cap | 512 | 0.683320 | 100 | 24.42 | 1 | 710 |
| `arithmetic-sum-pow4` | `forward_only` | cap | 128 | 0.758950 | 100 | 32.71 | 0.2261 | 190 |
| `arithmetic-abs-diff` | `full_batch` | cap | 128 | 2.07e-5 | 100 | 100 | 1 | 151 |
| `arithmetic-abs-diff` | `minibatch` | cap | 512 | 1.63e-8 | 100 | 100 | 1 | 597 |
| `arithmetic-abs-diff` | `forward_only` | CE bound | 38 | 4.7e-11 | 100 | 100 | 1 | 45 |

Three cells pass the 99% rule, the three arms of `arithmetic-abs-diff`; one
cell meets the stopping rule; 23 stop at their caps.

**The positive control.** With the given class order, `|a - b|` is a table of
the fold decoder: the row phases `a` and the column phases `-b` in `Z/193`
decode to it. The model is given no phase. The uniform start is right on 94
of the 4705 withheld pairs; at the first observation after it, update 19, all
three arms are right on all 4705, and stay so. The forward-only arm reaches
the CE lower bound and stops at update 38. The minibatch arm ends at its cap
`1.6e-8` above the bound, just over the tolerance, and the full-batch arm
`2.1e-5` above it: correct on every pair, and not converged in CE within the
caps. The structure is supplied by the decoder and the class order, not found
by the optimization: this is a control of the solver on a table that its
domain contains.

**Addition and subtraction are fitted in part.** Validation accuracy is 94.8
to 96.5%, below the 99% rule. The CE arms do not fit the training set (96 to
97%) and stay about 0.2 above the bound; the forward-only arm fits the
training labels by its largest class, and its saved table carries only a
third of the iterate. Every validation error has its answer near the wrap of
the modulus: for addition the classes 0 to 11 and 85 to 96 in the two CE arms
and 0 to 7 and 89 to 96 in the forward-only arm; for subtraction 0 to 14 and
78 to 96, and 0 to 6 and 88 to 96. All other classes are right on every
withheld pair.

Addition modulo 97 is not a phase table of this decoder. Shift the phases to
`r(0) = s(0) = 0`; the cells with `a = 0` or `b = 0` give `r(a) = ±a` and
`s(b) = ±b`; the cell `(1, 1)` and a reflection give the sign `+` at 1; the
cells `(a, 1)` and `(1, b)` then give it up to 95; and the cell `(50, 50)`
has the state 100, which decodes to 93, not to 3. (This remark is not
formalized.) Whether another table of the domain fits addition exactly is not
known.

**The other five tasks.** `arithmetic-sum-pow4`, which has 25 answers,
reaches 24 to 33% on validation, against 4.6% for its most frequent answer;
what it gets right was not analyzed. Permuted addition, division,
multiplication and sum-cube stay at 0.3 to 1.1%, each below the share of its
most frequent answer, while the largest class is right on every training
pair. The CE arms keep 0.68 to 0.69 above the bound on all five. These are
failures of this candidate, kept as such.

**Costs.** The producer processes took 7681 s of wall clock in total and the
replays 137 s; an update took 1.1 to 1.5 s on one thread. Peak resident
memory was 88 MiB under the 8 GiB address-space cap. Another serial cohort
queue ran on the same machine throughout, so the seconds are indicative only.
The whole cohort, replays included, took 2 h 10 min.

**What is not shown.** The 100% on the control is a measurement of saved
tables. Whether the coverage conditions of CM18 hold for that split was not
checked here, and no saved table was compared with a certificate; CM18 itself
shows that a CE minimum of this program need not have the certified answer.
The follow-up [`convex_stress_certified`](../convex_stress_certified) searches
for such certificates on the same eight tasks and reports its own results.
No solver convergence, no stability at the tolerance and no interval
certificate of the spectra is claimed. One seed and one split per task.

**Integration was not obtained.** `integrate` requires the decoded-phase
cohort `experiments/convex_stress_decode/runs/cohort-v1` complete and
unchanged. That cohort had not started when this one finished: it is queued
behind earlier cohorts of the same serial machine. `summary.json` records
`integrated_with_older_candidates: false`. No ordering against an earlier
candidate is claimed, and no earlier candidate is dropped or replaced by these
numbers. The command above remains to be run once that cohort is complete.
The `integrate` step of `convex_stress_certified` calls it on the same two
roots and refuses a changed `summary.json` of this cohort (sha256
`11ebba65…ad11068`), so the comparison is recomputed there as well.

These are eight of the 35 tasks of the suite. No full-suite result and no
comparison with a transformer is claimed.
