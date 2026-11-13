# Identification certificates for categorical completions

Can actual observed constraints justify choosing between models with equal
ordinary training cross-entropy? Task223 starts from CM28 and Lean commits
`4fee3d09` and `9cd95708`, before native implementation. They prove that a
cyclic joint module identifies an unknown bijective-code phase table over a
composite domain, including repeated input phases. Missing axes cannot be
silently pinned for this claim. The identification is conditional on the
stated categorical model, not a consequence of flatness or exact observed fit.

The first native component constructs an exact ring row certificate. It uses
actual swaps, unit scalings and row additions. If a pivot column contains
nonunits, bounded coefficient search combines rows to reduce the gcd with the
modulus. Only a checked unit is inverted. The result retains the full matrix,
every operation, completed pivots, stop reason and work. A successful replay
has every coordinate row, proving that the original anchored kernel is zero
by the Lean theorem `kernel_zero_of_rows`.

The independent checker uses i128 row arithmetic; the producer uses bounded
u32 residues. It checks each complete row operation, unit scaling, final matrix,
actual coordinate rows and arithmetic work. Full deterministic replay separately
checks every pivot/search decision and control-work counter. Partial reductions,
nonunit obstructions, missing pivots and caps remain explicit refusals. There
are no CE gradients in certificate construction.

Five tests exhaustively compare all 2274 two-variable square matrices modulo
2 through 6 with their actual finite kernels. They also replay the exact
four-instruction nonunit-pivot example from Lean, exercise larger rectangular
systems through modulus257, every work cap, invalid inputs, JSON round trips,
row/state mutations and changed control-work counters. These are synthetic
checks; no original task has been evaluated by this producer.

The categorical layer now binds the certificate to actual observed cell/label
identities and a proposed code. It checks class-code injectivity and every
original equation before shifting the two gauges. It constructs the exact
matrix from up to4096 observed equations, the two shift pins and one actual
unit class anchor. It never inserts pins for missing axes. The independent
checker rebuilds the partition, signed shifts and every coefficient separately
before replaying the ring certificate. A full replay also binds all decisions
and work. Acceptance means uniqueness within the declared bijective cyclic
model; it does not assert that arbitrary ground truth has that structure.

Five additional tests certify masked domains2–16 with repeated phases and
unknown shifted codes, reproduce the exact six-state Lean witness, and check
categorical renaming. A missing-axis example has two different predictions
with identical observed fit; both are refused. Other checks retain collisions,
conflicts beyond the elimination equation cap, budget stops, original-matrix
and endpoint mutations, and a realizable model with a noncyclic joint module.
The last refusal is a limitation of this sufficient certificate, not a proof
that decoded recovery is impossible. All ten native tests pass.

Matrices have at most1024 variables and8192 rows. The declared default permits
two billion scalar updates, two million row operations, twenty million pivot
tests and combination tests, and one million inverse tests. Storage forecasting
includes matrix, instruction, checker/replay and JSON copies and must fit8GiB.
Numerical processes obey the user's64GiB limit:

```sh
cd experiments/convex_stress_identified
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The adapter in `cohort/` now checks the complete saved modular report, including
every admitted/refused view and all raw spectral/CE audits. Each certificate is
bound to the actual observed class order, phase/code point, model hash, and every
proposed full-table prediction. Lean `6a8fb49e` proves the fixed-view scope: a
strip view also requires its constant-strip structural assumption. An interior
certificate alone does not identify arbitrary strip answers. Unequal padding
and missing axes are not given additional identification pins.

The declared training-only rule compares ordinary CE first. Within a1e-12 tie,
it prefers a verified fixed-view certificate, then preserves earlier order.
This is an explicit structural inductive assumption for selecting a model,
not an unconditional truth guarantee over the larger prediction hull. An
earlier selected model with the same certified model hash keeps its priority.
Five adapter tests cover repeated-phase constant strips, earlier missing-axis
pins, CE ties and improvements, changed held-out scores, empty banks, budget
refusals, and mutations of the actual model, matrix, scope and parent evidence.
All five pass; these are additional to the ten core tests.

```sh
cd experiments/convex_stress_identified/cohort
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The [original cohort harness](cohort) adds six checks and binds the complete
task222 parent, all23 original pair inputs, saved proposal artifacts and all105
mode choices. It seals the same declared rule before fresh original metrics
and replays every old constructor and new certificate. All21 native checks
pass. Producer `a2ae1a35` now completed all23 original cases and full replays
with zero process failures. Sixteen saved candidates receive the actual
fixed-view certificate; their unchanged predictions score100% on every
original training and held-out split. Seven cases have no admitted modular
candidate and retain the complete earlier result.

| Comparison mode | Earlier all-split passes | Certified-selection all-split passes | Primary passes |
| --- | ---: | ---: | ---: |
| Full batch | 26/35 | 30/35 | 31/35 |
| Minibatch | 30/35 | 30/35 | 31/35 |
| Forward only | 25/35 | 29/35 | 30/35 |

The four changed cases are power division atp97 andp193 withk2 andk3. Their
saved candidate models already passed in task222, but full/forward comparison
kept earlier wrong completions on equal ordinary CE. The declared training-only
certificate priority selects the passing model in those eight mode references.
Minibatch already selected them and retains its result. This is an observed
selection repair, conditional on the stated model family, with no changed
candidate predictions and no held-out score used to choose a model.

All105 choices and the complete immutable task222 parent remain. The five
remaining all-split failures shared by full/mini are Boolean AND shift,
square-times, both basis-depth tasks and recurrence. Forward also retains the
branch-amplitudes failure. The retained depth results await fresh review and
integration; the conventional transformer is still running its original cohort.
Full pass has not been achieved.

Certification plus its independent checks took553.734 seconds, charged once
per task with69 mode references and zero new CE gradients. Original proposal
construction costs remain attached in the complete parent. The largest recorded
prepare/run/replay RSS was4,256,420KiB, about4.06GiB, under the64GiB limit.
The source fingerprint is
`92e20503f66ea08cfa9a5791374e1fb8eba1e52b422fecad34e79251612038fa`;
the complete summary digest is
`5c13439e55af498340977c6b27c2492ccc840d4bfb9940568e7ad73fe4cf311d`.
Generated artifacts stay in root `runs/original-v1`; the frozen binary,
protocol, manifest and extracted work/findings stay in root
`.kanban/artifacts/223/original-v1`. Fresh board review is next.

All35 tasks, full batch, minibatch, forward-only, retained depth and the
conventional transformer comparison remain in scope.
