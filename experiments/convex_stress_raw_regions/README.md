# Raw-label contexts from observed input regions

Can observed input geometry repair raw-label mistakes which a margin guard
must preserve? This component uses the reviewed CM48 proof at `71d96c93`.
It wraps the exact CM46 model from `a8deaa63`, preserving its complete complex
head, records, mode metadata and original gradients. The spectrum supplies
the feature bound. Convex raw-class regions within a fixed source block are
additional structure; training data do not establish that assumption on
unknown labels.

For each ambiguous source block, the model constructs a convex hull of the
observed two-token coordinates for each raw class. The coordinates are the
original ordinal token IDs. No task name or arithmetic answer formula enters
construction. Different-class hulls must be disjoint; any intersection refuses
the entire block and keeps the parent. A matched unseen input receives a new
context only when exactly one class hull contains it. Observed inputs, missing
source matches, single-label blocks and uncovered inputs keep the exact parent.
This is an ordinal geometric bias, not invariance under arbitrary token renaming.

Membership produces at most three observed source states, positive integer
numerators and a positive denominator. An independent verifier checks source
IDs and raw labels, unit total mass and both original input coordinates using
checked integer arithmetic. It does not trust hull order or a floating-point
inclusion tolerance. Points, line segments, triangle boundaries, shared edges
and intersecting polygons are handled explicitly. The mathematical certificate
is exact; prediction converts its rational masses to floating point and is
tested separately for numerical feature and product errors.

The new feature row is the fixed weighted sum of the actual complex source
rows. Prediction multiplies that row by every original real and imaginary
head coefficient. Confidence scale is fixed at one in the protocol. There
is no separately trained output decoder, altered target or alternate loss.
With the context fixed, ordinary full-vocabulary CE is convex in the complete
head. Because every observed row is unchanged, the original observed objective
and derivatives remain exact. This does not assert joint convexity of feature
discovery and head training.

| Mode | Existing head training | New context construction |
| --- | --- | --- |
| Full batch | Retained exactly | Fixed observed geometry; no new gradient or update |
| Minibatch | Original order, repeated records and short tail retained | Same construction; no new gradient or update |
| Forward only | Original zero-update endpoint retained | Same construction; zero gradients and updates |

Eleven native tests cover integer membership and independently verified
coordinates, overlapping/degenerate hulls, exact boundaries, overflow,
uncovered inputs, unknown axes, actual complete complex products and bounds,
ordinary CE, source identity and saved-state mutations. The clean 47-record
fixture retains a repeated record and a seven-record tail. All three actually
trained head modes cover its three unseen inputs and predict their raw labels
correctly, with CE below 1e-8. The forward result equals the seven-class
margin 24 CE. A second fixture covers four unseen queries in two distinct
source blocks. A noisy fixture produces intersecting class hulls and keeps
the original model on those blocks. These are component checks, not new
original-suite results.

The preallocation forecast includes inherited costs, hull storage, rational
rows, scratch and serialization reserves, pairwise hull checks and all finite
input membership checks. Prediction forecasts include the longest allowed
prefix scan and complete source/head products. The component has a 56 GiB
buffer limit and 128 billion scalar-work guard; every test and experiment process
uses the 64 GiB address-space limit. Actual integer-work counters remain separate
from these conservative forecasts. No new geometry/head optimization cost is
hidden in an inherited counter.

The source, protocol and actual parent digest bind the saved model. Loading
reconstructs the complete parent, hulls, certificates, fallback decisions,
counts and bounds. A changed certificate, geometry, work count or rehashed
invalid parent is rejected. The native coverage example reads a saved source
envelope only; it never builds an evaluation dataset or reports held-out scores.

CM48 identifies covered raw labels only if the target has the stipulated
convex regions. Disjoint observed hulls do not prove that realizability, and
uncovered labels can remain ambiguous. A separate frozen comparison must retain
all 35 original tasks, both 105-choice policies, all modes and baselines, plus
the task 296 raw reference, task 301 guarded CE improvement and task 298 lower-CE
tradeoff. Original primary success, raw accuracy and CE remain distinct.
Prior benchmark results motivated this proposal; the suite is not an untouched
architecture-selection holdout. No original numerical gain is claimed here.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/shared-roles
export TMPDIR=/run/media/misha/storage/transformer-runs/shared-compare-tests
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cargo fmt --manifest-path experiments/convex_stress_raw_regions/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_regions/Cargo.toml -- --test-threads=1 --nocapture
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked \
  --manifest-path experiments/convex_stress_raw_regions/Cargo.toml --all-targets -- -D warnings
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
prlimit --as=68719476736:68719476736 -- cargo run --release --offline --locked \
  --manifest-path experiments/convex_stress_raw_regions/Cargo.toml --example coverage -- SOURCE_MODEL.json
```

Generated models, checks and coverage reports stay ignored on storage. Original
evaluation requires an exact reviewed component and a separately frozen adapter.
