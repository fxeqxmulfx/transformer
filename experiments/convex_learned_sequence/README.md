# Complete learned-sequence update kernels

Can native updates retain the entire CM75–CM78 complex sequence family while
satisfying the activation domain required by its ordinary cross-entropy proof?
The first component implements the exact projection certificate proved in
CM79, commit `945603c5`, under
`src/Transformer/Grokking/FlatSpectrum/ConvexChannel/LearnedHinge/Projection`.

The native model now executes the full learned complex forward and projects
the whole parameter array with retained component certificates. Complete cone
centres, direction bases and feasible queries now have native generators and
independent exact certificate checks. Mini and
forward-only Mini proposal rules and training remain unfinished. Known feasible
points used by controls are not learned outcomes; no stress accuracy is claimed.

## Complete domains and controls

| Object set / context cap | All prefixes | All scalar patterns | Coordinates per complex vector | Complete real parameter count |
| --- | ---: | ---: | ---: | ---: |
| 2 objects / 2 positions | 7 | 128 | 5 | 5,120 |
| 3 objects / 1 position | 4 | 16 | 4 | 768 |

Object integers are serialization identities, not arithmetic features. Inputs
remain sequences of original immutable objects. The raw row contains the bias
coordinate and one position/object address per visible position. CM79 proves
that its dot product with learned weights is the original raw lookup. The
rows here describe update constraints; they do not supply an inference model.

`DomainPlan` counts every bounded prefix, all scalar masks, both complex
components, every original output object and both readout signs before domain
allocation. An oversized declaration is refused. Inputs exceeding the context
cap, invalid objects and invalid masks are refused without truncation. Empty
prefixes and all original rows remain present.

`ProjectionProblem` generates candidates by enumerating active row subsets
and solving their Gram systems with exact rational Gauss–Jordan elimination.
Singular active systems cannot be inverted; their original inequalities remain
in every verification. A separate verifier checks all four CM79 conditions:
nonnegative multipliers, primal feasibility, complementarity and stationarity.
An accepted certificate proves global nearest-point optimality, so candidate
enumeration can stop once it has one. If no certificate is found, it refuses.
Solver completeness and practical full-suite cost are not inferred from this
finite search or from the controls.

Finite binary64 proposals are imported as their exact dyadic rational values.
Projection outputs stay rational, including denominators such as five. There
is no rounding back to an approximately feasible floating-point model in this
component. Nonfinite input, shape mismatch, failed certificate conditions,
arithmetic overflow, coefficient-bit limits and search/storage limits produce
explicit errors. Rational arithmetic uses the pinned `num-bigint` and
`num-rational` crates. This Rust implementation is tested against the Lean
specification; it is not itself a compiled or formally verified Lean program.

## Actual complex model and atomic updates

`ComplexModel` retains all rational coordinates in pattern/class/readout-sign/
component/coordinate order. Native sign zero denotes Lean's positive (`true`)
readout, and component zero denotes the real (`true`) component. Constructors
check every original prefix inequality before accepting the parameter array.

Each complex vector `u` recovers amplitude `sqrt(norm(u))` and representation
`u/amplitude`. The norm is the maximum coordinate complex magnitude, matching
Lean's function-space instance; the projection metric remains Euclidean in
joint coordinates. The forward reads the recovered learned bias and visible
position/object values, rectifies both components, multiplies by the complex
readout `amplitude*(1+i)` and sums signed neurons. It executes both learned
stages. A gated linear expression is not substituted for inference.

The physical-to-joint constructor follows CM76's exact component-pattern
aggregation. The optional CM78 paired forward calls the same model twice,
renames every input object, aligns every output class and averages complete
complex outputs. The renaming must be an involutive permutation of the entire
original object set. It is an explicit task restriction, not a universal fix.

Real logits enter stable full-class softmax and ordinary cross-entropy. Only
the loss accepts a target. The predictor returns an original object index;
ties select the first index while retaining every output class. Rational
parameters remain authoritative. Factor recovery, complex multiplication and
probabilities use binary64 and are qualified numerical approximations to the
Lean model. Nonfinite conversion/evaluation and detected underflow are refused;
no strict numerical-error enclosure or exact binary64 equivalence is claimed.

Whole-model projection checks every pattern/class/sign/component, retains every
full dual array and verifies that every certificate point matches the new
parameter array. Construction and independent replay must succeed before a
new model is returned. Failure leaves the old immutable model untouched. One
cone workspace is reused sequentially; no model pattern is removed. A
conservative working-memory reservation charges old/proposed/new state,
certificates, rational work and factor caches, with external RSS guards retained.

## Complete geometry for feasible forward queries

`ConeGeometry` implements CM80 and CM81, Lean commits `eb1634bd` and `2d2ca3ae`.
It projects every original normal, retains every full dual certificate and sums
the projected points. Independent replay checks the projections, complete centre
sum, all row slacks and forced-equality flags. Zero and redundant original rows
stay present. The auxiliary equality matrix zeros only non-forced rows; the
original cone remains intact in every feasibility and query check.

Exact Gauss–Jordan elimination retains the entire equality matrix and its full
row-operation matrix. It proposes full-coordinate basis vectors, free-coordinate
indices and recovery coefficients against every equality row. A separate checker
tests all three CM81 identities: kernel membership, identity on free coordinates,
and reconstruction of every original coordinate residual. The Lean theorem
then establishes coverage and uniqueness, rather than trusting a generated rank.
Deleting a feasible basis direction fails the reconstruction check.

Queries accept a complete direction and a positive rational step cap. Every
original row supplies its absolute-response bound; a tight row with nonzero
response refuses the query. The final two points and every bound are independently
replayed. Positive centre mixtures are available to open non-forced boundary
rows. Mixing changes the iterate and can increase loss, as recorded in CM80.
All returned points remain rational. To execute them, the full model constructor
checks the whole parameter array before actual recovered complex inference.

`GeometryPlan` admits the number of normal projections, the worst-case total
active-subset candidates, rational entry count and conservative byte reservation
before further construction. The reservation includes projection workspace,
normal certificates, elimination transforms, basis/recovery arrays and replay
buffers. It describes one complete cone, not an all-pattern training-memory or
runtime guarantee. Full-family caching and optimizer state need their own total
admission; the 110 GB aggregate RSS and both-disk guards remain applicable.

## Findings and validation

All ten native controls pass. They include an independent 125-point orthant
clipping reference, a slanted projection with exact nonbinary coordinates,
redundant and degenerate constraints, individual corruptions of each certificate
condition, exact binary64 import, complete raw-coordinate lookup, input/budget
refusals and all-pattern coverage. Two distinct proposals are certified for
every one of the 128 two-object patterns; a further proposal is certified for
each of the 16 three-object patterns. These 272 checks retain every row in their
respective domains. They are projection controls, not learning outcomes.

Nine additional model controls pass, for nineteen total. Actual recovered
forwards reproduce CM76's complex outputs `-7+7i` and `4+10i` with differing
component masks, as well as signed class readouts and empty-input behavior.
The CM78 feasible witness returns its two observed objects and the renamed
unseen object, with all three class CEs and whole complex equivariance checked.
This is construction of the proved witness, not a trained model or an accuracy
measurement. A full 768-coordinate projection retains and replays 192 component
certificates. Corrupted or incomplete certificates, infeasible arrays, invalid
objects/renamings and numerical/resource failures are rejected atomically.

Nine further geometry controls pass, for twenty-eight total. They cover the
degenerate diagonal cone, a slanted plane with rational nonorthogonal basis
vectors and recovery, zero and unconstrained cones, lineality, boundary
mixing, malformed queries, resource refusals and individual certificate
corruptions. Every one of the 144 patterns across the two complete domains
gets checked geometry. Across these controls, all 960 normal projections carry
their full certificates, and every certified basis direction is tested with
feasible queries. A full model with
all sixteen three-object patterns executes two actual complex query forwards:
the selected real and imaginary score changes are both `1/16`, with ordinary
full-class CE changing in the expected directions on every original prefix.
This is a known feasible control, not training or a new accuracy measurement.

Development tests, strict all-target Clippy and a release build pass. The Lean
projection/query components independently pass the full build, fifty-seven
direct axiom checks, global audit, native INDEX verification and forbidden-source
scan. Existing 36 `sorry` declarations remain; no new `sorry` or extra axiom was added.
Evidence and failed development logs stay ignored under
`.kanban/artifacts/366/`; generated output is not committed.

From the repository root, with build output on storage:

```sh
export CARGO_BUILD_JOBS=1
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/learned-sequence-v1
cargo fmt --check --manifest-path experiments/convex_learned_sequence/Cargo.toml
prlimit --as=4294967296:4294967296 -- cargo test --offline --locked --manifest-path experiments/convex_learned_sequence/Cargo.toml -- --test-threads=1
prlimit --as=4294967296:4294967296 -- cargo clippy --offline --locked --all-targets --manifest-path experiments/convex_learned_sequence/Cargo.toml -- -D warnings
prlimit --as=4294967296:4294967296 -- cargo build --offline --locked --release --manifest-path experiments/convex_learned_sequence/Cargo.toml
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ProjectionHingeAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/QueryGeometryAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/QueryBasisAxioms.lean
```

The development/review wrappers check aggregate RSS plus the process allowance
against 110,000,000,000 bytes and require free space on both disks before each
heavy command. Matrix-entry, rational-bit and active-subset limits are internal
representation/work limits, not a claimed end-to-end RAM forecast.

## Remaining work

Implement proposals and training for both required minibatch modes. A
forward-only rule must query the actual model and respect its domain; a gated
surrogate evaluated outside that domain does not inherit the CE theorem.
CM80 derives a centre by summing certified projections of every original
constraint normal. Its zero-slack rows are exactly the universally forced
equalities; their joint nullspace describes the feasible two-sided query
directions. Degenerate cones retain every coordinate and constraint. A positive
mixture of the current point with this centre admits those directions, with
explicit displacement and convex-loss bounds because mixing can increase loss.
The native centre and complete direction basis now implement these certificates;
no finite-difference gradient identity or convergence follows from domain safety.
CM81 supplies a finite basis-completeness certificate: every proposed direction
annihilates every forced row, its selected free coordinates form an identity,
and every original coordinate residual is an explicit combination of retained
equality rows. The Lean reconstruction theorem proves full kernel coverage and
unique coefficients from those checks. Native elimination generates and
independently verifies them. Next connect the complete family of these queries
to actual minibatch loss evaluations and update proposals. A nonorthogonal
basis requires recovery of a direction from its measured pairings; raw directional
values cannot simply be treated as Euclidean gradient coefficients.
CM79's quantitative descent result retains its actual-loss upper-bound premise.
Complete training, all-training sealing before heldout scoring, a separate
producer freeze, independent repeat and full raw replay precede any accuracy
claim. This small kernel does not replace the 99% raw/exact target on all 165
stress cases or solve the exponential full-family storage and search cost.
