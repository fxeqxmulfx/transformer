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

## Results on the original source splits

Four of the 48 reproducible cases crossed 99% on held-out pairs at every
seed in both training modes. Addition and subtraction were exact on every
pair. Modular division at modulus 97 scored 4,612–4,616 correct out of
4,656. The fourth pass was the modulus-97 power-zero constant control.
Together with the preceding four-case study, six of 51 measured cases meet
their original criterion and five reach 99% raw-object and exact-sequence
accuracy on every applicable held-out split. Six original datasets remain
unavailable and 108 cases fall outside these two cohorts.

| Selected case | Validation pairs | Correct pairs at seeds 0 / 1 / 2 in each mode |
| --- | ---: | --- |
| Addition and subtraction, each | 4,705 | 4,705 / 4,705 / 4,705 |
| Division, modulus 97 | 4,656 | 4,613 / 4,612 / 4,616 |
| Multiplication, modulus 97 | 4,705 | 4,620 / 4,613 / 4,616 |
| Division, modulus 193 | 18,528 | 18,441 / 18,294 / 18,443 |
| Constant output, modulus 17 | 136 | 126 / 130 / 124 |
| Constant output, modulus 97 | 4,656 | 4,650 / 4,652 / 4,648 |
| Constant output, modulus 193 | 18,528 | 97 / 18,523 / 18,525 |
| Permuted addition, modulus 97 | 4,705 | 170 / 171 / 113 |

The original multiplication case fell short at all three seeds. Division at
modulus 193 crossed the threshold at two seeds. Constant output also depended
on initialization and vocabulary size. At modulus 193 the seed-zero model
got 97 of 18,528 validation pairs correct while the other two seeds missed
only five and three. The success on addition did not transfer to the
original permuted-addition case.

Producer `dfa5ee4c` independently fit and sealed all 576 endpoints. Both
builds replayed 2,304 complete raw-logit files totaling 13,520,004,288 bytes.
On the overlap with the four-case study, all six addition model files per
build, optimizer receipts, original metrics and all 48 raw score files
were identical. Both modes had the same validation counts at all 144
arithmetic case/seed combinations. Comparing their final predictions on
every training and validation row found one different next object in
1,631,091 decisions, on the sum-to-the-twelfth-power case. The largest
absolute logit difference across the family was 0.0263.

The complete experiment took 910.76 seconds on the shared CPU. Forward-only
fits took 3.90–92.49 seconds and issued 899,072–9,910,272 complete minibatch
loss queries per fit. Standard minibatch fits took 0.44–3.93 seconds.
Peak per-fit resident memory was 23,676 KiB. The raw runs are in
`/run/media/misha/storage/transformer-runs/composed-retrieval/arithmetic54-v1`.

This command reruns both archived raw decoders, verifies the overlapping
models and logits, and reports per-case outcomes, per-seed counts, measured
costs and the combined 165-case ledger.

```sh
base=/run/media/misha/storage/transformer-runs/composed-retrieval
cargo run --release --locked --offline --features arithmetic-cohort \
  --manifest-path experiments/composed_retrieval/Cargo.toml --bin reconcile-arithmetic -- \
  "$PWD" /run/media/misha/storage "$base/matched-visits-v1" "$base/arithmetic54-v1"
```
