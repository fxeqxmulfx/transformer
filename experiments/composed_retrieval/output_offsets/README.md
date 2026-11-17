# Learning one output offset per original object

Can the two-read successor learn tasks where its angular readout consistently
underpredicts one object? On the original training records for modulus 97,
every mistake on division, multiplication and affine product had target
zero at every seed of the physical-depth reference.

| Training case | Rows with target zero | Errors at seeds 0 / 1 / 2 |
| --- | ---: | --- |
| Modular division | 52 | 31 / 39 / 37 |
| Modular multiplication | 98 | 51 / 64 / 67 |
| Affine product | 81 | 47 / 54 / 53 |

The new model learns a scalar for each original output object. It adds the
scalar to the scaled angular score on every prefix. All offsets start at
zero, use the existing scalar learning rate of 0.06 and stay between −16
and 16. The keys, tied values and readout, rotations, sequence interactions
and gates continue to train jointly from ordinary full-vocabulary
cross-entropy. With all offsets zero, every starting prediction agrees with
the log-odds reference. [Frozen protocol](protocol.json)

CM116 bounds every logit by 32 and gives the exact strict-margin conditions
for a one-class change at fixed base scores. The derivative is predicted
class probability minus the observed indicator. Each forward-only query
computes the full partition after moving a feasible coordinate. A common
shift is invisible to observed class contrasts. The coverage and loss
claims for the fitted joint model are empirical. [Flat-spectrum notes](../../../review/flat_spectrum.md) · [Lean statements](../../../src/Transformer/Grokking/FlatSpectrum/ConvexChannel/ComposedRetrieval/OutputOffsets.lean)

One cohort contains the three retrieval controls and the 48 available
original arithmetic tasks, including the shared addition control once.
Six unavailable original arithmetic sources retain their entries. Each
build fits the same 51 cases, two modes and three seeds for 1,024 updates.
Both builds must seal all 306 endpoints before either scores a held-out
record. The independent decoder checks the original pass criterion and
99% object and exact-sequence accuracy on every applicable held-out split.

```sh
base=/run/media/misha/storage/transformer-runs
git show <producer-commit>:experiments/composed_retrieval/run.sh | bash -s -- \
  "$PWD" /run/media/misha/storage <producer-commit> "$base/composed-retrieval/<fresh-output>" \
  output-offsets "$base/arithmetic-data/inputs-v1"
"$base/composed-retrieval/<fresh-output>/original/target/release/audit-raw" \
  "$PWD" /run/media/misha/storage "$base/composed-retrieval/<fresh-output>"
```

## What a single offset can repair on the training rows

The training-only check uses the frozen log-odds reference and selects
each endpoint's most frequently missed target class. At seed 0 the
following observed strict bounds apply when changing only that class.

| Training case | Offset must exceed | Offset must stay below |
| --- | ---: | ---: |
| Modular division | 9.1813 | 2.8324 |
| Modular multiplication | 6.0803 | 1.6426 |
| Affine product | 15.9597 | 1.4443 |

All three intervals were empty at every seed in both modes. Among all
288 arithmetic endpoints, 50 already made every training prediction
correct. Four could make every training target a strict winner by shifting
one class. These were seeds 0 and 2 in both modes of the modulus-193
constant task. The other 234 had conflicting strict margins or an error
involving additional classes. The new fit lets all parameters move.

```sh
base=/run/media/misha/storage/transformer-runs/composed-retrieval
manifest=experiments/composed_retrieval/Cargo.toml
cargo run --release --locked --offline --manifest-path "$manifest" --features output-offsets \
  --bin offset-intervals -- "$PWD" /run/media/misha/storage "$base/logodds-arithmetic54-v1"
cargo run --release --locked --offline --manifest-path "$manifest" --features output-offsets \
  --bin compare-output-offsets -- "$PWD" /run/media/misha/storage "$base/logodds-depth-v1" \
  "$base/logodds-arithmetic54-v1" "$base/<fresh-output>"
```

## Joint training results on 51 cases

The learned-offset model passed **nine original criteria and eight strict
criteria** at every seed in both minibatch modes. Its log-odds reference
passed six of each on these same 51 cases. Division modulo 193 and the
modulus-17 and modulus-193 constant controls gained strict passes.
Division modulo 97 lost one. Multi-query recall regained its original
95% pass with 31/32 novel validation sequences at its weakest seed.

| Original held-out split | Log-odds reference, seeds 0 / 1 / 2 | Learned offsets, seeds 0 / 1 / 2 |
| --- | --- | --- |
| Division modulo 97, 4,656 pairs | 4,613 / 4,612 / 4,616 | 4,624 / 4,591 / 4,617 |
| Division modulo 193, 18,528 pairs | 18,441 / 18,294 / 18,443 | 18,429 / 18,358 / 18,436 |
| Multiplication modulo 97, 4,705 pairs | 4,620 / 4,613 / 4,616 | 4,596 / 4,612 / 4,577 |
| Constant, modulus 17, 136 pairs | 126 / 130 / 124 | 136 / 136 / 136 |
| Constant, modulus 97, 4,656 pairs | 4,650 / 4,652 / 4,648 | 4,656 / 4,656 / 4,656 |
| Constant, modulus 193, 18,528 pairs | 97 / 18,523 / 18,525 | 18,528 / 18,528 / 18,528 |
| Multi-query recall, 32 novel sequences | 31 / 32 / 30 | 32 / 32 / 31 |

Both modes produced the same object and exact-sequence counts on every
split at all three seeds. Key/value binding remained 512/512 on novel
validation and on test. Two-hop lookup remained 64/64 at each test length
through 96. Modular addition and subtraction remained 4,705/4,705. The
48 arithmetic members include the repeated addition control once. Six
other arithmetic cases have unavailable original sources and 108 cases
lie outside this cohort.

Two independent builds of `e60c0c43` trained all 612 endpoints before
held-out scoring. They matched every model and optimizer receipt and
replayed 2,928 complete raw score files. The run took 1,352.59 seconds.

All 732 initial split score files matched the log-odds reference exactly.
The sampled training rows and coordinate visits matched too. For the
modulus-193 constant control, removing the output offsets after training
reduced correct training predictions from 18,528/18,528 to 85–97/18,528
across the six fits. The target class had the largest learned offset at
every seed. With the same removal, binding stayed at 2,048/2,048 and
two-hop lookup at 256/256 training targets. On modulus-97 multiplication,
removing offsets from the three standard minibatch fits cost 80, 69
and 99 correct training predictions. Their held-out accuracies still
missed 99% at every seed.
