# Training two reads with a log-odds depth gate

Does keeping the second read active early in training let ordinary
next-object cross-entropy learn both lookups? In the preceding experiment
the depth gate reached zero during the first 51 updates. At the final
weights, forcing the second read increased loss on the complete training
set in all six fits. [Training-only measurements](../depth/README.md)

| Protocol | Initial depth | Adam coordinate for depth |
| --- | ---: | --- |
| Matched-coordinate control | 0.5 | Physical gate projected to [0,1] |
| This experiment | 0.9 | Unconstrained log odds with physical gate equal to its sigmoid |

The model still scores a learned mixture of the same two reads. With
physical depth `h`, the ordinary CE derivative in log odds is its
physical depth derivative times `h(1−h)`. Forward-only training evaluates
two actual minibatch losses at perturbed log odds. Both modes visit every
coordinate at every update. The remaining weights, learning rates, data
batches and 1,024-step endpoint come from the frozen matched-visit protocol.
[Predeclared comparison](protocol.json)

Both builds fit all four original controls at seeds 0, 1 and 2. Each build
seals every endpoint before held-out scoring. The runner checks all weights,
training receipts and full raw logits and replays every original split.

```sh
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  <producer-commit> <fresh-output> logodds
cargo run --release --locked --offline --features logodds-depth \
  --manifest-path experiments/composed_retrieval/Cargo.toml --bin audit-raw -- \
  "$PWD" /run/media/misha/storage <fresh-output>
```

## Results on the four original controls

Both modes learned two-hop lookup at all three seeds. Every supervised
object and every complete sequence was correct on every original split.
The fit saw only the final answers for the two original queries per
sequence. Novel validation went from 4–8/32 exact in the control to 32/32.
The length-96 test went from 0/64, 3/64 and 0/64 to 64/64 at all three seeds.

| Case and split | Matched control, seeds 0 / 1 / 2 | Log-odds depth, seeds 0 / 1 / 2 |
| --- | --- | --- |
| Two-hop lookup, novel validation | 5 / 8 / 4 of 32 | 32 / 32 / 32 of 32 |
| Two-hop lookup, tests at lengths 24 / 48 / 96 | Length 96 scored 0 / 3 / 0 of 64 | 64/64 at each length and seed |
| Binding, novel validation and test | 512/512 throughout | 512/512 throughout |
| Modular addition, validation | 4,705/4,705 throughout | 4,705/4,705 throughout |
| Multi-query recall, novel validation | 32 / 32 / 31 of 32 | 31 / 32 / 30 of 32 |
| Multi-query recall, length-96 test | 64 / 64 / 63 of 64 | 64 / 62 / 63 of 64 |

Each count in the table holds for both standard and forward-only
minibatch training. The log-odds study passed three original cases and
three strict cases. Multi-query recall missed its original 95% exact
target at seed 2. The physical-depth control passed three original cases
and two strict cases. These are distinct training protocols.

All six lookup models ended with depth 0.9696–0.9730 and product gate zero.
Holding the other learned weights fixed, setting depth to zero gave 0/256
correct training targets in every run. Setting it to one gave 256/256.
At their trained depth the full-data physical CE slopes were −0.00291 to
−0.00279, and actual training CE was 0.000262–0.000276. For the earlier
control, the corresponding full-data slopes at zero were positive. The
intervention changed both the starting depth and the optimizer coordinate.

Two fresh builds of `4c2ea1fe` trained and sealed all 48 endpoints before
scoring, and independently reproduced and decoded all 672 raw-score
files. The run took 353.39 seconds. The complete per-seed comparison,
training slopes and held-out counts can be reproduced locally.

```sh
base=/run/media/misha/storage/transformer-runs/composed-retrieval
cargo run --release --locked --offline --features logodds-depth \
  --manifest-path experiments/composed_retrieval/Cargo.toml --bin compare-depth -- \
  "$PWD" /run/media/misha/storage "$base/matched-visits-v1" "$base/logodds-depth-v1"
```
