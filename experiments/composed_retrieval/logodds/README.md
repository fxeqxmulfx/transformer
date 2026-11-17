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
