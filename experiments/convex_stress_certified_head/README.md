# Certified evaluation of the complete complex head

Can evaluation of the actual saved tensor resolve the real/complex numerical
disagreements in task347 while preserving ordinary cross-entropy and reporting
genuine ties? This is task349's separate numerical component. The frozen347
producer, coefficients, training trajectories, scores and refusals stay intact.
This component does not identify an unseen target or improve program discovery.

Lean came first: CM63 at18308305 proves18 results in
`ConvexDepth/ExactScoring`, including both complete contractions, common positive
normalization, strict ordering, equality and common-shift ordinary CE. A feasible
nonconstant example fits its observed labels but permits two conflicting unseen
completions; another theorem shows that arbitrarily small logit residuals can
coexist with opposite strict winners. The full Lean build and axiom checks passed.

| Evaluation path | Decision and numerical checks |
| --- | --- |
| Separated enclosures | Evaluate every saved real coefficient, enclose each operation with outward binary64 rounding, and require one lower numerator bound to exceed all other upper bounds. Independently multiply the complete complex tensor. Require both normalized vectors inside the shifted enclosures, identical winners, residual and maximum enclosure width at most1e-7. |
| Exact fallback | Convert the finite coefficients and input features to exact binary rationals. Independently contract the complete real tensor and the complex tensor in different orders, requiring exact equality. Retain every exactly maximizing class and use the existing first-index tie convention. |
| Conversion refusal | Refuse nonfinite inputs, invalid dimensions, exceeded precision/resource bounds, or a strict exact gap that disappears when converted to binary64. No tolerance declares unequal numerators tied. |

Both paths subtract a common maximizing numerator before the common positive
normalization. This changes the stored logit origin while preserving the model's
ordinary softmax probabilities and cross-entropy in exact arithmetic. It is an
evaluation representation, not a different loss or a different feasible set.
Numerical enclosure widths are reported separately. Lean does not prove the Rust
IEEE754 implementation or the num-rational conversion correct. Native checks
compare interval containment and winner certificates with independent exact
products, including imaginary terms, subnormals, cancellation and genuine ties.

The model adapter accepts only the saved model and input token prefix. It uses
both program interpreters, preserves the original feature map, categorical
unknown-entry default, and coverage diagnostics. Targets enter a later scorer.
No target-program AST, task ID, expected class, heldout label or preferred winner
enters the evaluator. Program discovery still lies outside joint convex CE.

Admission limits are64 programs,1024 classes,16 real coefficients per program and
class, absolute input features at most4096 and coefficients at most1e14. The
forecast accounts for at most seven coefficient multiplication passes (the exact
complex path distributes over four source rows), an8192-bit integer cap and
conservative buffer space below56GiB. These are explicit arithmetic-operation
and memory limits, not measured time or a proof of big-integer CPU cost. Processes
run under the user's64GiB address-space cap. Complete experiment costs must also
include discovery, training, serialization, scoring and replay.

All nine native component checks and strict Clippy pass. The checks include the
unchanged forward/full/minibatch trajectories on a handwritten sparse-label
fixture, an unsupported categorical entry, exact coefficient reconstruction,
ordinary CE, all-class interval containment and first-index tie semantics.

Native checks (no Python):

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/certified-head-v1
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_certified_head/Cargo.toml -j2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_certified_head/Cargo.toml --all-targets -j2 -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --offline --locked --release --manifest-path experiments/convex_stress_certified_head/Cargo.toml -j2
```

After committing and freezing the executable, `diagnose MODEL_JSON PREFIXES_JSON
OUTPUT_JSON` compares old and corrected complete products on explicitly supplied
input prefixes. It consumes no target labels and produces no accuracy measure.

No heldout numerical-correction result is claimed here. A complete producer must
be committed and frozen before the new13-case, six-endpoint comparison, then
independently rebuilt and repeated. The77/165 verified coverage remains unchanged.

## Frozen input-only diagnosis, 2026-11-15

Component commit `d1b10e2e` was frozen before diagnosis; binary SHA256 is
`8d15b1b08237a70022eb002b1b83308c94425d6c1a7560c0c7143ef7d59d8c27`.
For each failed depth3 endpoint (Coordinates Forward and Mini), take the final
successfully logged input prefix and all36 possible one-token continuations.
Only the saved model and these37 token prefixes enter each diagnosis. No target
labels enter it, and no accuracy is measured. This includes invalid-grammar
continuations and does not estimate the fallback rate on benchmark queries.

| Saved endpoint | Old inconsistent queries | Maximum residual on those queries | Exact raw winner |
| --- | ---: | ---: | ---: |
| Forward | 3 | 4.547473508864641e-13 | 16, uniquely |
| Mini | 2 | 1.1368683772161603e-12 | 15, uniquely |

The old real and complex products chose15/16 in opposite orders. Independent
exact contractions agree on a unique maximum in all five cases: these particular
failures are floating-point ordering errors, not genuine mathematical ties.
All74 queries complete. Each37-query diagnosis takes about4.1 seconds and4.5MiB
peak RSS under64GiB. The separate frozen-source rebuild passes all nine tests and
Clippy and reproduces all query results exactly. Its executable differs when
built from the different checkout path, so the artifact comparison excludes only
the explicitly recorded executable hash; both hashes remain in the evidence.

Evidence is retained outside the worktree in
`/run/media/misha/storage/transformer-runs/certified-head/diagnostic-v1/` and
`.kanban/artifacts/349/`. These numerical checks do not remedy the separate
depth3 generalization failure or add catalog coverage.
