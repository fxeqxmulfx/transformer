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

## Held-out results

All seven additional cases missed both the original all-seed criterion
and the 99% all-split criterion. Two-hop lookup again passed both.
Multi-query recall met its original target. Both training modes had
the same exact-sequence counts on every split at every seed.

| Additional case | Exact on novel validation, seeds 0 / 1 / 2 | Exact on length-64 test, seeds 0 / 1 / 2 |
| --- | --- | --- |
| Alternating blocks | 18 / 11 / 9 of 28 | 4 / 5 / 13 of 64 |
| One-type brackets | 14 / 12 / 8 of 32 | 7 / 7 / 10 of 64 |
| Two-type brackets | 9 / 9 / 6 of 32 | 10 / 11 / 8 of 64 |
| Three-type brackets | 4 / 4 / 4 of 32 | 12 / 12 / 4 of 64 |
| Counting formula, depth 1 | 30 / 30 / 31 of 32 | 18 / 38 / 36 of 64 |
| Counting formula, depth 2 | 22 / 22 / 22 of 32 | 41 / 42 / 40 of 64 |
| Counting formula, depth 3 | 23 / 23 / 22 of 31 | 10 / 26 / 44 of 64 |

Depth 1 fit all 1,400 supervised training positions and all 128 training
sequences at every seed. At length 64 it scored 3,774, 3,877 and 3,903
correct objects out of 4,032. The other six new cases also made training
errors at every seed. The native data and metric checks preserve the
smaller novel subsets in the alternating-block and depth-3 cases.

The two independent builds of `3601fa91` sealed 108 fits and replayed
2,376 full raw score files. Both shared cases matched their earlier
models, complete training receipts and 528 original raw files, including
the initial outputs and all probes. Counting the overlapping cases once
gives **58 measured, nine original passes and eight strict passes** for
this model. Six original arithmetic sources are unavailable and 101
catalog cases are outside the two cohorts.
