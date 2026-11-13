# Categorical discovery with colliding class codes

Can observations discover an additive latent code when several raw output
names share a latent state? CM30 in `review/flat_spectrum.md` and Lean commit
`e02b7c49` prove that original row certificates can identify codes and compatible
quotient answers. They also show that a certified code need not identify a
unique raw answer, even when every raw label occurred in training.

This native component proposes noninjective codes and binds them to a bounded
complex latent model with an ordinary cross-entropy readout. The complete state catalogue
is every prime from2 through257, fixed independently of the data. One `Request`
runs one catalogue stage; callers must retain each outcome and release its
large field report before proceeding. The original cohort has completed all23
cases and initial independent replays. Its complete comparison remains34/35
full-batch,34/35 minibatch and33/35 forward-only on the declared primary metric;
the fresh complete review is pending. See the cohort README for every outcome
and cost.

| Step | Rule |
| --- | --- |
| Names | Compact only observed row, column and output names in first-observed order; preserve inverse maps |
| Equations | Original observed equations `row + column - class = 0`; up to4096 by default |
| Gauges | First observed row and first observed class set to zero; no unseen-axis pins |
| Proposal | Require exactly one remaining field direction and a nonconstant class component; normalize its first nonzero class coordinate |
| Collisions | Keep all raw names and every code fiber; no class-code bijection requirement |
| Validation | Replay all actual field operations and check every original observation, including those outside the discovery prefix |
| Scope | Discrete discovery followed by convex fixed-feature CE; conditional code certification, no raw-label identification or original benchmark pass |

The finite-field engine is the existing independently checked
`convex_stress_fibers::finite_system`. It records elementary operations, rank,
free coordinates, caps and contradictions. The additional checker reconstructs
categorical names and uses signed128-bit arithmetic to normalize and check the
actual proposed coordinates. It does not use the proposal's query method for
its original-equation check. Unknown input axes and unseen raw labels return
`None`, including axes omitted by compaction. Ambiguous or capped stages retain
their entire unsuccessful report.

Five discovery tests pass. A synthetic table has14 observed row names,14 observed column
names and14 raw class names, embedded in17 input categories and29 output
categories. Its hidden seven-state code has two raw names per code. The checked
stages2,3,5,7,11 admit only7; the discovered phases match every observed and
held-out target code. Separate class/axis permutations and reversed observation
order retain code correctness. Other checks cover unused axes/classes, work
caps, ambiguity, incompatible observations beyond the elimination prefix,
invalid inputs, serialization and seven altered reports. These tests concern
latent codes; they do not establish raw-label generalization.

Defaults allow500 million scalar updates and2 million row operations per
stage. The existing field engine caps variables at1024 and rows at8192. A
conservative single-stage construction/check/serialization forecast must fit
8GiB; commands retain the user's64GiB process cap. The full55-stage catalogue
has up to27.5 billion default scalar updates across separate stages, before
independent replay; it must not be described as free or as a complete search
over composite moduli or higher-dimensional solution spaces.

```sh
cd experiments/convex_stress_quotient
cargo fmt --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  prlimit --as=68719476736:68719476736 -- cargo test --locked -- --test-threads=1
cargo clippy --locked --all-targets -- -D warnings
```

The actual model and training binding now pass eighteen tests in total. Every
admitted code materializes its full latent table. Missing axes receive a zero
phase to make prediction total; their identification mask remains false and
their row certificates refuse. For each latent frequency, the auditor constructs
the raw complex matrix from the actual saved table and computes its SVD. Both
operator and nuclear norms must be at most the input category count, with
numerical tolerance1e-7. These bounds concern latent moments, not the output
logit matrix, and are numerical checks rather than exact-real certificates.

The readout learns one tensor `W[latent,class]` on latent one-hot features.
Every original vocabulary class remains in the softmax denominator; raw labels
are unchanged. Each class column has L1 radius `32*vocabulary`. CD8's
`JointHead.lean` already proves this domain and ordinary CE convex for arbitrary
fixed features. It does not make discovery or jointly learned phase coordinates
convex. No new Lean premise is needed for this fixed-feature specialization.

Initialization counts observed raw labels at each latent state. A positive
count gets logit `24 + max(log(count)-log(total),-24)` and a zero count gets
zero. This is an observed-frequency proposal, not a claim of optimal CE or
unique raw decoding. Counts, logarithms and materialized phase cells are charged
once and retained. The mode defaults are explicit:

| Mode | Updates | Batch | Step size |
| --- | ---: | --- | ---: |
| Full batch | 16 | All observations | 1 |
| Minibatch | 256 | 64, cyclic original order | 0.1 |
| Forward only | 0 | None | None |

Each update uses the actual usual-CE derivative and Euclidean projection onto
the class-column L1 balls. Saved steps retain batch offsets, CE, derivative
norms and subsequent training CE. An independent checker recomputes derivatives
with separate accumulation and uses a bisection projection; exact replay then
reproduces all saved states and counters. A finite-difference check uses a
nonoptimal head, and weighted minibatch gradients reproduce its full gradient.
The forward endpoint is byte-equivalent to its initializer and computes no CE
derivatives. Numerical audits and replay costs remain additional work; Fourier
matrix counts, audited entries and feature-dot-product terms are explicit.

The CM30 certifier ports the original-row engine from
`convex_stress_consequences` at70cf9ab5 and removes only its injective
class-count restriction. It keeps one variable for every original raw class,
not a substituted learned code. In the seven-state fixture, all196 known-axis
cell codes are correct and certified in all three modes, including held-out
cells. The remaining padded axes stay uncertified. The head's raw training CE
is below0.7, but raw training mistakes remain because aliases share features.
No raw-label pass is inferred from the successful code test.

The exact CM30 witness with four observed raw classes and two codes certifies
every query while one held-out raw answer is wrong. A declared evaluation
readout can be checked for constancy on the *candidate* code's observed fibers.
That check cannot prove compatibility with the unknown target code. Changing
the readout changes neither model construction nor any training endpoint.
Both target compatibility and additive target realizability remain explicit
premises of the Lean quotient theorem. Caps, unknown classes, missing axes and
discovery refusals retain their reports. Fifteen binding mutations are rejected
in addition to the original field and certificate mutation checks.

The complete binding preflights all three stored endpoints and certificates
below56GiB, inside the64GiB process cap. Training separately caps a conservative
scalar-work forecast at2 billion per mode. A single raw-SVD matrix is resident
at a time; no input-by-input-by-latent-by-vocabulary tensor is allocated.
Stages must still be released between primes. Task232's original evaluation
now uses task226's reviewed complete parent comparison and a separate
source/binary/protocol freeze preserving all35 tasks, full/mini/forward and
the conventional transformer. Original model construction follows the full
data and parent verification. The completed cohort retains every prior success
without a new task pass; see the complete counts and limitations above.

The separately tested `cohort/` component now streams all55 stages to saved
reports and replays them independently. Five catalogue tests retain two successful
quotients in one fixture and choose by observed CE, along with every refusal,
returned error and work counter. Three selector tests preserve both105-choice
parent arms; eight additional original-adapter fixtures bind full parent evidence,
actual states, unchanged raw CE and complete original metrics. Compatible decoded
answers can pass while unseen raw-token answers fail. The adapter and complete
parent are independently reviewed. The frozen original cohort is complete,
and its fresh review will replay every original stage and the full summary.
See the cohort README for the complete outcome and approval boundary.
