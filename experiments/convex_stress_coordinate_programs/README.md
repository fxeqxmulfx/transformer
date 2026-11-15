# Joint convex coordinate programs

This native core implements CM57 at Lean commit
`258dd055995a5dc1f00cc8842d2f5490afc4decc`. It asks whether fixed input-only
program features can support joint decoder/mass training through the actual
complete complex head, with ordinary cross-entropy and fully charged budgets.
Task 342 validates the core; task 343 supplies the stress-test feature families
and frozen benchmark driver. This crate alone adds no catalog pass. The last
measured scope remains 58 of 165 cases, with 107 unmeasured.

Each program supplies five fixed features. The six learned coordinates are
five scaled decoder coefficients and a nonnegative program mass. The model
does not multiply separately learned decoder and routing factors. Two exact
four-point Fourier blocks produce all 16 real/imaginary coefficients for every
program and raw class. Prediction evaluates the expanded tensor. An independent
complex multiplication checks its equality with the CM57 algebra.

The supplied injective class coordinates and identity-copy coefficient are
explicit priors. Homogeneous anchor equations use training labels only. One
common input-only normalizer covers all programs. The complete weighted cost
is `(4,4,4,4,4,2)` per program; global and perspective budgets constrain every
coefficient. A zero-mass program cannot retain uncharged decoder coefficients.
The public prediction API accepts input features without reference labels.

The initializer projects a mass-only parameter proposal onto the training
constraints. Subsequent CE steps update every joint coordinate and project
again. There is no separate fitted decoder held constant during this stage.

| Mode | Initialization | CE updates | Batch handling |
| --- | --- | --- | --- |
| Forward | Training-constraint projection | None | No gradient records |
| Full | Same projection | All six coordinates per program | All original records |
| Mini | Same projection | All six coordinates per program | Original IDs, repeated inputs, actual short-tail denominator |

Projection uses bounded Dykstra iterations over four convex sets: the observed
linear subspace, the perspective cones, the positive-total-mass halfspace and
the complete weighted absolute-sum ball. The cone and weighted-ball projections
use scalar KKT multipliers. Two-pass modified Gram-Schmidt constructs observed
row bases, with a numerical rank threshold of `1e-11`; every original anchor
is independently checked after projection. The iteration cap is at most
20,000 cycles and failure to meet the requested tolerance returns an error.

Projection reports both its maximum normalized constraint violation and its
maximum normalized change across the individual projections. Anchor errors
are scaled by that program's mass and the anchor row magnitude, so a large
coefficient in another program cannot conceal a violated anchor. Global and
perspective errors use their respective budget scales. The full observed
feature-span residual is reported separately from target correctness.
Deserialized heads and domains must pass their `validate` methods before use;
saved outcomes are checked against a complete deterministic replay.

These are floating-point checks, not a Lean refinement or convergence theorem.
The finite feature family is supplied. Observed feature-span coverage does not
prove unknown-target realizability; an uncovered role remains ambiguous. The
hard anchor domain may be empty, including for identical input features with
contradictory labels. Such cases are failures, never silently successful fits.

The meaningful checks cover the two-active-program CM57 witness, its rejected
constant program and unseen class 3; complete products and a nonzero imaginary
coefficient; every CE derivative by central finite differences; weighted
minibatch tails; independent KKT/cone solutions and feasible-grid variational
checks of the joint projection; copy coverage and opposite uncovered-role
predictions; mode replay and trace corruption; shape/memory limits; and an
explicit contradictory-label refusal. Numerical witness results do not replace
heldout stress-test measurements.

All eight checks pass, along with formatting, clippy with warnings denied and
the release build. The direct CM57 and global Lean audits remain unchanged:
36 existing `sorry`, zero resting on them and zero extra axioms. The contradictory
input control is refused within its declared 128-cycle test budget. Both CE
training modes make effective updates on the nonconstant witness, and all
three modes retain its unseen-class and copy predictions. The complete native
trace, including each projection, replays exactly.

The entry points bound dimensions, scalar work, trace storage and conservative
serialization copies before allocation. The forecast limit is 56 GiB, leaving
headroom under the user's 64 GiB process limit. Features are bounded by 4096
before their common normalization; coordinates lie in [-1,1]. No Python is
executed. Build output and generated evidence belong on storage, not in Git.

From the repository worktree:

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/coordinate-programs
export CARGO_BUILD_JOBS=2
manifest=experiments/convex_stress_coordinate_programs/Cargo.toml
cargo fmt --manifest-path "$manifest" --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path "$manifest" -- --test-threads=2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path "$manifest" --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path "$manifest"
LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2 lake build
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/CoordinateProgramAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean -j 2 scripts/Axioms.lean
scripts/native_index/target/release/lean-index --check
```

Before any new heldout measurement, commit and freeze the complete task 343
driver, this core, its native dependencies, configuration and data inputs.
Retain the whole reviewed task 339 report, the separate original-35 reference,
all transformer histories and the independent-label control.
