# Categorical codes in composite latent domains

Can observed categorical equations learn a phase code when the proposed cyclic
domain is composite? CM27 / Lean `98b12abc` proves exact residual lifting,
checked Bezout recombination and unit gauge identities before this native work.
It also proves that a valid base solution may have no lift, and that arbitrary
noncoprime moduli do not justify multiplying the modulus. These results do not
identify unseen labels or guarantee successful discovery.

The native integer core now passes five checks. It retains the original matrix
and right side, canonical base representatives, exact divided residual, reduced
prime-field system, every elimination operation and its complete affine space,
selected digits, next representatives, final original residuals and work.
When the affine space is nontrivial, it keeps every direction and explicitly
chooses its free-zero representative; that is not a uniqueness claim. Caps and
inconsistency leave no selected next point.

The independent checker uses i128 original-row arithmetic while the producer
uses bounded i64 arithmetic. It verifies coefficient reduction, full invertible
row operations, actual digit/residual identities and every final congruence.
CRT recombination retains and checks the actual Bezout witness, each component
residue, the explicit representative and all original equations. Unsupported
noncoprime components remain refusals. Deterministic replay additionally binds
every input, intermediate state, stopping decision and work counter.

Tests compare all 97 small scalar lift systems modulo four/nine against direct
enumeration of original integer solutions, and over 2000 small CRT residue pairs
against exhaustive product-domain search. They include CM27's nonzero matrix
example, a nonunique digit space, its failed base lift, each solver cap, invalid
representatives/domains, JSON round trips and tampering with residuals, matrices,
chosen points, Bezout witnesses and work. No original task was used in these
checks.

The `coding` layer now proposes a cyclic domain from observed class count and
uses observed relations `r[i]+s[j]=t[label]`, with unknown class codes and
repeated input phases allowed. Classes are ordered by first occurrence; their
token numbers carry no arithmetic meaning. It retains up to the first 4096
equations, pins one observed row and the first class to zero, and factors the
proposed domain. Each prime component must have exactly one kernel direction
after these pins. Its first nonzero class coordinate is normalized to one by
an explicitly checked unit. Each prime-power digit must then have a unique
solution in this normalization. Larger kernels and failed lifts are explicit
refusals of this proposal policy, not proofs of global impossibility.

Checked CRT combines the complete components. The resulting class codes must
be distinct, and every original observation is checked, including those beyond
the equation cap. A complete decoder and actual phase table receive all-frequency
raw DFT/SVD and ordinary full-vocabulary CE audits. No CE gradients are used in
this constructor. Missing axes are listed and explicitly assigned phase zero;
this arbitrary choice does not identify their unseen answers. All component
matrices, anchors, residuals, operations, endpoint and refusal witnesses, caps
and work counters remain in the report and deterministic replay.

Seven additional tests recover masked categorical tables for every domain size
2 through 16, including prime powers, CRT components and repeated phases. They
check renamed classes with unused vocabulary, a missing-axis example that fits
all observations but gets unseen answers wrong, all stage/work caps, noncyclic
kernel refusal, actual class collisions, omitted-equation conflicts, invalid
inputs and fourteen certificate mutations. The complete suite has twelve tests.
These are synthetic checks, not an original benchmark score.

The `search` wrapper now retains the existing CM25 categorical strip views
around this code learner. It proposes the unchanged full table first, followed
by supported constant-strip views in observed class order. Its default protocol
can attempt at most four views; every additional, empty, failed or duplicate
view remains recorded. Interior axes may repeat and are padded when their
counts differ. Admitted interiors lift to full tables in the existing tagged
latent domain, with independent full-table map and raw spectral/CE checks.
Three wrapper checks cover composite interiors of sizes 6/8/12, strips,
padding, class renaming, view/stage caps, small interiors and nine mutations.
All fifteen tests pass. The wrapper forecast must fit within 48 GiB. The
parent-preserving [original harness](cohort) now passes five further checks and
covers all23 original eligible pair tables, including recurrence, retaining the full105-choice parent. Its
parent digest now pins the complete checked integration:26/35 full/mini and25/35
forward all-split passes. Preparation refuses a changed parent.

The original follow-up froze producer `f661532e` before preparation. All 23
cases completed and replayed, with zero process failures. Sixteen produced a
new candidate, each with 100% accuracy on every original split. Seven refused:
square-times, sum-square, both absolute-difference cases, sum-cube, sum-power-four
and recurrence. Earlier families retain their existing passes for five of these.
Square-times exposes a larger component kernel; recurrence remains outside the
accepted proposal policy. These refusals do not prove the tasks impossible.

The complete comparison preserves all 105 choices and the frozen CE tie rule:

| Mode | Parent all-split passes | New all-split passes | New primary passes |
| --- | ---: | ---: | ---: |
| Full batch | 26/35 | 26/35 | 27/35 |
| Minibatch | 26/35 | 30/35 | 31/35 |
| Forward only | 25/35 | 25/35 | 26/35 |

The four added minibatch passes are power-division p97/k2, p97/k3, p193/k2
and p193/k3. Their new candidates also pass in the other modes, but tie the
earlier exact observed fit; the declared rule retains the earlier failing
completion. No held-out score changes that selection. New construction costs
574.527 seconds, charged once with 69 mode references and zero CE gradients.
The largest recorded run/replay RSS is 2,501,016 KiB, about 2.39 GiB, under the
64 GiB limit. Complete parents, rejected views and all original results remain
in ignored `runs/original-v1`; the summary digest is
`5d3aa99d91efdab222fc3bd5c1c8ea68f16c8f632808b22f84c62213f4355139`.

CM28 / Lean `4fee3d09` now identifies an unknown bijective-code table when
the full observed joint module is cyclic, including composite domains and
repeated phases. It also proves that arbitrary pins for missing rows cannot
justify identification. Native certificates and a separately frozen structural
selection rule are task223; the original run above remains unchanged. Full
35-task success, retained depth integration and the fresh transformer comparison
remain unfinished.

The core restricts integer matrices to 1024 variables and 8192 rows, coefficients
of absolute value at most 1024, right sides at most 2^30, and total modulus at
most 257. The lift forecast covers report, checker, replay and serialization
and must fit within 8 GiB. Coding has a separate 16 GiB forecast, eight total
stages by default, at most 500 million scalar updates per stage and two billion
across stages; every stage has a two-million-row-operation cap. Numerical
processes use the user's 64 GiB cap:

```sh
cd experiments/convex_stress_modular
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

Source: [`review/flat_spectrum.md`, CM27](../../review/flat_spectrum.md#cm27-composite-categorical-codes-require-checked-modular-lifts).
Prime-field row operations reuse the checked finite-system core at `05b89afc`;
integer residual lifting, original-system checks and CRT witnesses are new.
All35 short/long tasks, full batch, minibatch, forward-only, retained stronger
families and the conventional transformer comparison remain required. Any
original follow-up needs its own tested and committed producer freeze.
