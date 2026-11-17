# Original synthetic prefix tasks

Does the same jointly trained two-read model generalize across the nine
synthetic cases that score supplied input prefixes? Seven cases extend
the 51-case output-offset study. Multi-query recall and two-hop lookup
repeat its original fits as controls. Each model sees only the sequence
of immutable object IDs and predicts one next original object.

| Cases | Vocabulary | Longest input | Supervised positions in 128 training records |
| --- | ---: | ---: | --- |
| Multi-query recall, two-hop lookup | 164, 100 | 96 | 256 each |
| One-type brackets, alternating blocks | 36 | 64 | 1,624 and 1,420 |
| Two- and three-type brackets | 38, 40 | 64 | 1,704 and 1,684 |
| Counting formula depths 1, 2 and 3 | 36 | 64 | 1,400, 1,440 and 1,370 |

Each input adapter reproduces the reviewed native split and tensor
fingerprints, target masks, changes, clean training and novel validation.
The retained first-cohort and remaining-synthetic inventories provided
the nine metadata hashes before training. The original per-example
historical fingerprints were unavailable. Target-generating formulas
belong to the data adapters. The learner receives the original training
examples and the object dimensions. [Frozen source and case hashes](protocol.json)

The original plan specifies batch size 32, data seed 1 and a 95% exact
sequence target on novel validation for each case. All nine use the
CM116 architecture and optimizer with 1,024 updates in both minibatch
modes at model seeds 0, 1 and 2. Each independent build fits 54
endpoints. Both seal before either scores its first held-out split.
The raw decoder also checks 99% object and exact-sequence accuracy on
every applicable held-out split. [Training protocol](../output_offsets/protocol.json)

```sh
base=/run/media/misha/storage/transformer-runs
manifest=experiments/composed_retrieval/Cargo.toml
cargo run --release --locked --offline --manifest-path "$manifest" --features prefix-cohort \
  --bin prefix-inventory -- "$base/arithmetic-data/inputs-v1"
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  <producer-commit> "$base/composed-retrieval/<fresh-output>" prefix-cohort \
  "$base/arithmetic-data/inputs-v1"
"$base/composed-retrieval/<fresh-output>/original/target/release/audit-raw" \
  "$PWD" /run/media/misha/storage "$base/composed-retrieval/<fresh-output>"
cargo run --release --locked --offline --manifest-path "$manifest" --features prefix-cohort \
  --bin reconcile-prefix -- "$PWD" /run/media/misha/storage \
  "$base/composed-retrieval/output-offsets51-v1" "$base/composed-retrieval/<fresh-output>"
```
