# Final-answer modular chains with learned object radii

Can the same next-object model learn the hidden composition from final
answers alone, and predict a final object after twice as many composition
steps? The study covers all twelve original combinations of modulus
17, 97 or 193 and one, two, four or eight training steps. Each example
has one supervised target. Exact sequence accuracy and raw object
accuracy therefore agree on each split. The original criterion uses
the 1,024 validation inputs. The strict criterion also includes the
1,024 doubled-length inputs. [Verified source data](../chain_data/README.md)

Each seed starts a fresh CM117 model and trains all its weights for
1,024 updates. A minibatch contains 256 original chain examples. The
analytic and forward-only methods see the same original training
prefixes and final labels. The model receives a sequence of immutable
object IDs. The recurrence and intermediate residues stay in data
preparation. The source and optimizer checks verify every training
mask and the physical forward-loss query count on a complete original
minibatch. [Parent model and optimizer](../product_radii/README.md)

Modulo-97 addition is the thirteenth case. It checks that the fitted
weights, optimizer traces and 48 initial and final raw files from the
parent study reproduce. Both independent builds seal all 156 fits before
scoring. Full-vocabulary f64 logits require 1,140,971,712 bytes for all
912 raw files. Reconciliation counts the overlap once and retains the
six unavailable arithmetic sources. [Frozen protocol](protocol.json)

From the repository root, set `INPUTS` to the source snapshot prepared
by the modular-chain data audit, then run

```sh
base=/run/media/misha/storage/transformer-runs
study="$base/composed-retrieval"
out="$study/learned-radii-chains13-v1"
experiment=experiments/composed_retrieval
producer=$(git log --diff-filter=A --format=%H -- "$experiment/chains/protocol.json")
bash "$experiment/run.sh" "$PWD" /run/media/misha/storage "$producer" "$out" chain-cohort "$INPUTS"
"$out/original/target/release/reconcile-chains" "$PWD" /run/media/misha/storage \
  "$study/learned-radii64-v1" "$out"
```

When the generation study has also completed, passing its directory as
the final argument checks both overlaps and combines all three studies
under the same architecture and training protocol.

## Held-out results

All twelve modular chains missed the original 99% validation criterion
and the strict criterion. Each entry is the range of correct final answers
out of 1,024 across the three seeds and both training methods.

| Modulus | Training steps in the chain | Validation exact | Double-length exact |
| ---: | ---: | ---: | ---: |
| 17 | 1 | 89–108 | 77–84 |
| 17 | 2 | 72–76 | 56–92 |
| 17 | 4 | 66–77 | 63–82 |
| 17 | 8 | 61–69 | 68–77 |
| 97 | 1 | 7–12 | 9–15 |
| 97 | 2 | 6–12 | 7–13 |
| 97 | 4 | 3–9 | 8–15 |
| 97 | 8 | 7–14 | 8–11 |
| 193 | 1 | 5–6 | 5–7 |
| 193 | 2 | 4–7 | 4–8 |
| 193 | 4 | 1–8 | 4–8 |
| 193 | 8 | 3–8 | 4–12 |

The strongest validation fit answered 108/1,024 correctly on the
one-step modulo-17 chain. On four-step chains modulo 193, training
accuracy reached 999–1,057 of 2,048 while validation stayed at 1–8 of
1,024. The training gain gave little information about unseen answers.

Both builds of `9df38fb5` produced matching endpoints and replayed
all 912 raw files. The six modulo-97 addition fits reproduced the
parent's training receipts and all 48 original raw files. Counting that
control once brings CM117 to **76 of 165 cases measured, 10 original
passes and 8 strict passes**. Six original arithmetic sources remain
unavailable, and 83 other cases fall outside these two reconciled cohorts.
