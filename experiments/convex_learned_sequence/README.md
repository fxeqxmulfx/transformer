# Complete learned-sequence update kernels

Can native updates retain the entire CM75–CM78 complex sequence family while
satisfying the activation domain required by its ordinary cross-entropy proof?
The first component implements the exact projection certificate proved in
CM79, commit `945603c5`, under
`src/Transformer/Grokking/FlatSpectrum/ConvexChannel/LearnedHinge/Projection`.

This component projects real component vectors. Lean proves how certificates
for every component combine into a feasible update of the whole complex model.
The native whole-model forward, Mini trainer and forward-only Mini trainer are
not implemented here yet. No stress case or model accuracy is claimed.

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

## Findings and validation

All ten native controls pass. They include an independent 125-point orthant
clipping reference, a slanted projection with exact nonbinary coordinates,
redundant and degenerate constraints, individual corruptions of each certificate
condition, exact binary64 import, complete raw-coordinate lookup, input/budget
refusals and all-pattern coverage. Two distinct proposals are certified for
every one of the 128 two-object patterns; a further proposal is certified for
each of the 16 three-object patterns. These 272 checks retain every row in their
respective domains. They are projection controls, not learning outcomes.

Development tests, strict all-target Clippy and a release build pass. The Lean
component independently passes the full build, sixteen direct axiom checks,
global audit, native INDEX verification and forbidden-source scan. Existing
36 `sorry` declarations remain; no new unproved input or extra axiom was added.
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
```

The development/review wrappers check aggregate RSS plus the process allowance
against 110,000,000,000 bytes and require free space on both disks before each
heavy command. Matrix-entry, rational-bit and active-subset limits are internal
representation/work limits, not a claimed end-to-end RAM forecast.

## Remaining work

Implement the unchanged actual complex forward with recovered learned factors,
the full parameter array and proposals for both required minibatch modes. A
forward-only rule must query the actual model and respect its domain; a gated
surrogate evaluated outside that domain does not inherit the CE theorem.
CM79's quantitative descent result retains its actual-loss upper-bound premise.
Complete training, all-training sealing before heldout scoring, a separate
producer freeze, independent repeat and full raw replay precede any accuracy
claim. This small kernel does not replace the 99% raw/exact target on all 165
stress cases or solve the exponential full-family storage and search cost.
