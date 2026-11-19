# Shared decision geometry with an actual convex complex head

Task331 follows the failed shared-head improvement in330. That complete
comparison retains35/35 original success flags but only34/35 raw99 outcomes
in every mode. Fibonacci remains97.94–97.96%, with higher ordinary heldout
CE than319. Of forward's96 errors,91 have a true raw label already observed
in their routed block. The next question is whether shared decision geometry
can use observations across blocks to constrain their uncertain boundaries.
This is a hypothesis to test, not an identification or accuracy guarantee.

Lean first: `scripts/SharedGeometryAxioms.lean` audits CD8/CFt5/CM38 in
`review/convex_depth.md` and `review/convex/`, `ConvexDepth/{JointHead,ComplexHead,Fallback/Batch}`
and Mathlib's `ConvexOn.comp_affineMap`. For a fixed affine map from compact
parameters to every real/imaginary raw-class coefficient, ordinary CE remains
convex on the preimage of the complete coefficient budget. Original records
remain distinct even when their inputs repeat; normalized CE uses each actual
batch cardinality, including a short final batch. The audit constructs a
nonconstant map sharing two parameters across two complete four-state complex
blocks and both raw classes, with a nonzero fixed coefficient offset. An
explicit nonzero parameter point satisfies the expanded budget. The actual
Fourier features and expanded budget give the same logit bound as CD8.

Existing theorems suffice for that exact preimage contract. CM55 now adds a
sufficient weighted parameter domain suitable for native projection. For
`H[k,c] = B[k,c] + sum_p A[p,k,c] * theta[p]`, each weight bounds the complete
column mass `sum_k abs(A[p,k,c])` for every raw class. If the fixed offset
has radius `T` and `sum_p w[p]*abs(theta[p]) <= R`, the actual complete head
has radius `T+R`. The new Lean module proves this bound, domain convexity,
the actual logit bound and ordinary full/minibatch CE convexity. A nonzero
two-parameter example attains the charged radius. No `sorry` is introduced.
Native work must check every actual coefficient/logit. Numerical
projection, gradients and state replay remain separate implementation checks.
Both the feature library and the raw-class decoder/orientation map are fixed
during ordinary CE optimization. Discovering those maps is not jointly
convex and must use only training records. Flat component spectra supply
bounds, never missing raw labels.

The prospective component will compare a declared generic shared feature
library with the reviewed329 local-head control. Any choice of geometry,
decoder orientation or initializer must use original training records and
training-only folds or record exclusion. Raw labels remain categorical;
permuting their names must permute decoder outputs consistently. No task name,
arithmetic identity, generator answer or heldout label may choose a prediction.
Full and minibatch modes differentiate the actual ordinary full-vocabulary CE
through the fixed coefficient map; forward-only performs zero CE gradients.
All old/new candidate states, refusals, tails, work and CE increases remain.
Declare complete constructor/fold/gradient/projection/prediction and storage
costs before fitting, under64GiB, and verify exact saved-state reconstruction.

The reviewed329 source supplies the mathematical audit. Task330's
independent repeat has passed. Task332 depends on
both330 and331: it must freeze a separate all35 comparison and retain the
stronger319 result and330 regression before reporting any new99% attainment.
The Rust component implements three fixed generic feature banks: quadratic
features alone, or with separate uniform positive hinges on each ordinal input
axis at15 or31 knots. Complete8/64/128-state Fourier features are bounded after
absolute-mass normalization. All binary blocks share one actual complex head;
each has a separate real intercept and a learned categorical orientation. A
fixed one-state feature supplies common192 logits on observed classes. Empty
blocks remain zero, singleton blocks retain that fixed common score, and a
block with more than two labels is explicitly unsupported. Neither missing
classes nor missing block coverage is certified by the model.

Every real/imaginary coefficient is the stated affine expansion of the stored
parameters. With `M` Fourier states, each intercept contributes `beta/sqrt(M)`
to every real Fourier coordinate. The CM55 weights charge each shared coordinate
by the maximum number of binary blocks containing any raw class, and each
intercept by `ceil(sqrt(M))`. The separate common offset is charged by its
complete class multiplicity. The complete radius is1048576, larger than329's
65536: this is a different capacity budget, not a matched-budget superiority
claim. An independent evaluator expands all raw classes and complex products.

The analytic constructor solves group-centered class-code covariance equations
with three declared ridges and16 categorical orientations. V2 retains each
moment intercept and also tests an interval midpoint when the projected training
classes strictly separate. On overlap it keeps the moment intercept. V2 also
tests16 fixed shared linear directions, with interval midpoints and balanced
class-centroid midpoints on overlap; ridge0 explicitly denotes this constructor.
These are analytic initializers, not CE gradient updates. Four fixed gains are tested.
Three input-hash folds retain repeated records together and exclude each held
cell from the head constructor. The minimum pooled ordinary fold CE selects
one of1344 configurations;1008 fold fits and4032 gain scores are saved, along
with refusals, coefficient states, hashes, expanded norms and independent
product residuals. A final all-training refit precedes ordinary CE training.
The inherited route already used all original training observations, so these
are conditional head diagnostics, not unbiased estimates for the entire model.
The old329 initializer is reconstructed separately on each fold and retained.

Full and mini use128 full updates or8 epochs of127 original records (actual
tail retained), followed by the final iterate without checkpoint selection.
Forward performs zero CE gradients, including selection and refit. Each CE
step uses the exact weighted Euclidean L1 projection. Native checks compare
its KKT conditions with an independent solution, gradients with finite
differences, unequal batch sums with the full objective, arbitrary complete
complex products, categorical permutations, excluded-label independence,
new linear/nonlinear synthetic queries and complete serialized-state replay.
An additional calibration check verifies correct separation, unchanged
overlap behavior and independence from excluded labels. The driver compares
every retained v1 state, decoder, numerical score, gap-independent product
audit and local control exactly using `retained_candidates.jq`.

The prospective forecast includes every fold/constructor/gain, three old local
heads, complete implied tensors, all original records, gradients, projections,
independent full products and an exact run replay. It refuses more than1T
kernel units or56GiB explicit buffers; every process has a64GiB address-space
limit. Parent parsing is separately charged before fitting. Generated source
freezes, complete states, reports and resource logs remain on storage and are
not committed. No original heldout measurement is part of this component.

The first frozen producer `0a55b83f` completed all six original training-only
inspections and exact internal replays. Its summary is `06590309ffca1513dbf99bc2d7e1a9610142863f15afaf8016f2c4fb9a87141b`
under `shared-geometry/inspection-v1`, with sources/binary under
`freezes/shared-geometry-v1`. It took5m03.71s and3.60GiB peak RSS under64GiB.
All modes selected Hinges16, ridge0.01, direction1 and gain128. Primary
conditional folds score4534/4704 (local329 control4606/4704); the supplemental
head scores4526/4704. These folds include unavailable decoder labels and are
conditional on the inherited route, as described above.

| V1 mode | Primary training hits/4704 | Primary final CE | Supplemental hits/4704 | Supplemental final CE |
| --- | ---: | ---: | ---: | ---: |
| Forward | 4618 | 0.04338673458 | 4620 | 0.04530137585 |
| Full | 4624 | 0.04207754423 | 4620 | 0.04366581517 |
| Mini | 4623 | 0.04293233185 | 4621 | 0.04458912826 |

The v1 moment constructor therefore underfits observed labels despite the
correct affine convexity contract. CE training reduces its own loss slightly;
it does not resolve the representation error. V2 adds generic calibration and
shared directions before any original heldout evaluation, preserving v1's
complete run and all original moment candidates. This is a new hypothesis to
test, with no claim of99% heldout accuracy. Its prospective work allowance
includes the expanded search and all calibration record passes.

Producer `b19b3bc9` freezes651 inputs under `freezes/shared-geometry-v2`.
All six original training inspections and exact internal replays exit0 under
`shared-geometry/inspection-v2`; summary
`826b4fa770a13daa1a584c7cec15c2be3c6e3ec9a75646a59b5e1d44be74cb5c`.
The run takes8m51.92s with3770788KiB (3.60GiB) peak RSS under64GiB. Every
v1 constructor parameter, decoder, score, product audit and local control
matches exactly in all six `retained-v1.json` checks.

All six selections choose the quadratic bank, the fixed direction `[0,1]`
(ridge0), interval calibration and gain128. Training intervals separate in
all93 primary and94 supplemental binary blocks. Every mode gets4704/4704
training labels correct in each role. Primary and supplemental conditional
folds both score4623/4704 (98.2781%). Primary's local329 control scores4606/4704;
the new conditional score is an improvement, but remains below99%.
Six primary fold targets lack a decoder label (including one in an empty
block);75 of the81 errors have a label present in the fold decoder. These
are training-fold diagnostics, not the original heldout pool.

| V2 mode | Primary final CE | Supplemental final CE | Effective updates per role |
| --- | ---: | ---: | ---: |
| Forward | 0.01354427549 | 0.01386963798 | 0 |
| Full | 0.01348541478 | 0.01381566995 | 128 |
| Mini | 0.01352660142 | 0.01385678559 | 304 |

Every real minibatch tail has5 original records. The maximum discrepancy
between compact logits and independently expanded complete complex products
is1.1369e-13. The model stores405/113 parameters in the primary/supplemental
roles; its actual expanded tensors have1692928/422144 real coefficients.
The prospective bank allowance still charges up to25593088/6381824 expanded
coefficients,1009 constructors including refit,4032 candidate gain scores and
6322176 conditional scoring records per role. Original heldout scores remain
unmeasured until task332.

Task331 is approved at producer `b19b3bc9`. The independent `review-v2` run
reproduces all32 JSON artifacts exactly, including the complete candidate
states, final heads, histories, retained-v1 checks and summary826b4fa7. It
takes8m47.46s with3770496KiB peak RSS and exits0. All eight native checks,
fmt/clippy/release, full Lean build, direct axiom audits, generated INDEX and
forbidden checks pass; the rebuilt binary equals the frozen one. Review
evidence and board history are preserved under the v2 freeze's `findings-review`.

```sh
export LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
prlimit --as=68719476736:68719476736 -- lake build
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/SharedGeometryAxioms.lean
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/Axioms.lean
prlimit --as=68719476736:68719476736 -- scripts/native_index/target/release/lean-index --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_shared_geometry/Cargo.toml -- --test-threads=1
cargo fmt --manifest-path experiments/convex_stress_shared_geometry/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_shared_geometry/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_shared_geometry/Cargo.toml
experiments/convex_stress_shared_geometry/freeze.sh "$FREEZE" experiments/convex_stress_shared_geometry/target/release/convex-stress-shared-geometry
bash "$FREEZE/source/experiments/convex_stress_shared_geometry/run_inspection.sh" "$FREEZE/convex-stress-shared-geometry" "$PARENT" "$OUTPUT"
```

`PARENT` is the reviewed319 `recurrence-fibonacci-k1-c2-lgt` directory. The
three complete model hashes and the separate baseline/329/330 summaries are
pinned in `inspection-inputs.json`. The driver is invoked from the immutable
freeze, attempts all six mode/role inspections, and preserves each exit code.
It must not be edited during execution. Set `FREEZE` and `OUTPUT` to fresh
storage paths. These are reproduction commands, not an accuracy claim.
