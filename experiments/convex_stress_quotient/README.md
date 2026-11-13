# Categorical discovery with colliding class codes

Can observations discover an additive latent code when several raw output
names share a latent state? CM30 in `review/flat_spectrum.md` and Lean commit
`e02b7c49` prove that original row certificates can identify codes and compatible
quotient answers. They also show that a certified code need not identify a
unique raw answer, even when every raw label occurred in training.

This native component proposes noninjective codes. The complete state catalogue
is every prime from2 through257, fixed independently of the data. One `Request`
runs one catalogue stage; callers must retain each outcome and release its
large field report before proceeding. No original suite has been evaluated.

| Step | Rule |
| --- | --- |
| Names | Compact only observed row, column and output names in first-observed order; preserve inverse maps |
| Equations | Original observed equations `row + column - class = 0`; up to4096 by default |
| Gauges | First observed row and first observed class set to zero; no unseen-axis pins |
| Proposal | Require exactly one remaining field direction and a nonconstant class component; normalize its first nonzero class coordinate |
| Collisions | Keep all raw names and every code fiber; no class-code bijection requirement |
| Validation | Replay all actual field operations and check every original observation, including those outside the discovery prefix |
| Scope | No raw-label identification, ordinary-CE optimization, spectral model or benchmark pass yet |

The finite-field engine is the existing independently checked
`convex_stress_fibers::finite_system`. It records elementary operations, rank,
free coordinates, caps and contradictions. The additional checker reconstructs
categorical names and uses signed128-bit arithmetic to normalize and check the
actual proposed coordinates. It does not use the proposal's query method for
its original-equation check. Unknown input axes and unseen raw labels return
`None`, including axes omitted by compaction. Ambiguous or capped stages retain
their entire unsuccessful report.

Five tests pass. A synthetic table has14 observed row names,14 observed column
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

Next, bind proposals to an actual complex latent model and a bounded readout
with ordinary full-vocabulary cross-entropy. CM30 certification must use the
original raw-label rows, without replacing them by an assumed correct learned
code. Compatibility with a benchmark's existing readout stays explicit. Only
after those checks may a separate original freeze compare all35 tasks and all
three modes while preserving the reviewed parent and transformer results.
