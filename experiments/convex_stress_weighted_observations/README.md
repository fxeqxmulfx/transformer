# Exact weighted observations

Can the complete CM57 head train on more than200000 original observations
without changing ordinary CE, its joint gradients, or the minibatch sequence?
CM60 in Lean, committed as819d4547 before this implementation, proves exact
multiplicity grouping. This native library retains the original observations
while evaluating repeated complete feature/observed-label pairs once per batch.

| Component | Behavior |
| --- | --- |
| Original pool | Every original ID, its order and group assignment are retained; declared count must match the complete stream |
| Group key | All five features of every fixed program and the observed raw label; no rounding, current-logit equivalence, or label repair |
| Full batch | Each group has its complete original multiplicity; denominator is the original record count |
| Minibatch | Consecutive original ranges are counted separately; a short tail uses its actual size |
| Forward | Same joint constraint projection, zero gradient updates |
| Convex stage | Existing complete complex coefficients, ordinary full-vocabulary CE, joint scaled decoders and program masses, copy constraints and charged budgets |
| Noisy labels | Conflicting labels remain separate groups; incompatible exact anchors remain a visible refusal |

`corpus::Corpus::build` consumes an iterator of original records, so the
caller need not allocate every repeated feature vector at once. IDs must be
strictly increasing. Canonical positive zero replaces negative zero; every
other finite feature is keyed by its exact representation. All labels remain
observed labels. Counts are integers, not learned weights. A streaming SHA256
binds the IDs, order, complete representatives, labels and multiplicities.
Training traces retain original batch ranges and their exact group counts;
the bound corpus recovers every original ID without copying IDs every epoch.

CM60's active-constraint equivalence justifies passing one representative per
group to the existing CM57 domain constructor. This does not alter its exact
anchor requirement. A single distinct group is currently refused by that
constructor. In particular, this library does not yet supply a noise-tolerant
domain or prove that supplied features identify unknown answers. Discrete
program discovery stays outside the convex claim.

Prediction and the derivative's softmax use every complete stored real/imaginary
coefficient and every raw class. The derivative applies the CM57 affine product
identity; no sampled classes or approximate logits are substituted. Actual
complete-product counts distinguish weighted group evaluations from represented
original records. Forecasts include key storage, original IDs, constraint copies,
complete coefficients, all optimizer traces, bounded projection cycles and
record processing. They cap a pool at two million original records, groups at
200000, buffers at56GiB and scalar work at600 trillion. The caller must also
forecast data generation, feature discovery, scoring and retained artifacts.
The process still runs under the user's64GiB address-space limit.

Six native checks compare grouped and independently retained ungrouped CE,
every joint derivative, all three training modes and complete complex-product
finite differences. They also check unequal frequencies, actual short tails,
corrupt counts/IDs/labels, deterministic replay and contradictory noisy anchors.
A200003-record stream exceeds the old cap while preserving every ID: three
groups have counts66668/66668/66667, and a127-record schedule ends with105
records. Its two full updates represent400006 gradient records while evaluating
six weighted groups. These are implementation checks, not new stress-test scores.
Floating-point summation order can differ; numerical comparisons use explicit
tolerances, while replay of this implementation is exact.

```sh
cargo fmt --manifest-path experiments/convex_stress_weighted_observations/Cargo.toml --check
CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/weighted-observations \
  CARGO_BUILD_JOBS=2 prlimit --as=68719476736:68719476736 -- \
  cargo test --offline --locked --manifest-path experiments/convex_stress_weighted_observations/Cargo.toml
CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/weighted-observations \
  CARGO_BUILD_JOBS=2 prlimit --as=68719476736:68719476736 -- \
  cargo clippy --all-targets --offline --locked \
  --manifest-path experiments/convex_stress_weighted_observations/Cargo.toml -- -D warnings
```

Task346 still needs complete Basis feature construction, treatment of noisy
constraints, and a committed/frozen eight-case producer with an independent
repeat before any new accuracy can be counted. The existing69/165 result and
all prior baselines remain unchanged.
