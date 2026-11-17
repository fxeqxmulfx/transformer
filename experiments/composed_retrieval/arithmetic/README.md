# Original two-operand arithmetic with the learned object model

How many of the 54 original arithmetic and powered-division cases generalize
with the same next-object predictor that learned modular addition? Each example
supplies its two original objects and a single supervised answer. The model
learns its object representations, interactions and readout with ordinary
full-vocabulary cross-entropy. Each case gets a fresh fit at seeds 0, 1 and 2.

| Training mode | Measurements per update | Updates per scalar |
| --- | --- | ---: |
| Standard minibatch | Analytic minibatch CE derivative | 1,024 |
| Forward-only minibatch | Two feasible full-minibatch CE evaluations per scalar | 1,024 |

The source inventory covers 40 arithmetic and 14 powered-division cases.
Both builds check the complete original input hashes and regenerate all 48
available pairs of training and validation splits before fitting. Six cases
require missing fixed random tables or sets. Their fingerprints and case names
remain in the inventory. The three power-zero constant controls are included.
The original arithmetic criterion is at least 99% exact on validation.
With one supervised target per pair, raw-object and exact counts coincide.
[Source inventory](../../convex_stress_arithmetic_data/README.md)

Two independent builds seal all 576 endpoints before scoring. Each writes
initial and final full-vocabulary logits for every training and validation
pair. The runner checks raw replay and equality between builds. All training
batches, learning rates, capacity and parameter bounds follow the preceding
four-case experiment. [Training protocol](protocol.json)

```sh
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  <producer-commit> <fresh-output> arithmetic <original-arithmetic-input-directory>
cargo run --release --locked --offline --features arithmetic-cohort \
  --manifest-path experiments/composed_retrieval/Cargo.toml --bin audit-raw -- \
  "$PWD" /run/media/misha/storage <fresh-output>
```
