# Forward-only training with matched coordinate visits

Does visiting every weight on every minibatch let forward-only training
learn the original modular-addition task? In the first four-case CM113
study, standard minibatch learned all 4,705 held-out sums. Forward-only
training got 79, 64 and 82 correct at the same three seeds. Its key, value
and rotation coordinates received only 21 or 22 updates. [Previous results](../README.md#results-at-the-frozen-endpoint)

| Training mode | Updates per physical coordinate | Source of the update |
| --- | ---: | --- |
| Standard minibatch | 1,024 | Derivative of full-vocabulary minibatch CE |
| Forward-only minibatch | 1,024 | Difference of two feasible full-vocabulary minibatch CE observations per coordinate |

Freeze this comparison on all four original cases, including binding, two-query
recall, and two-hop lookup. The model, initial angles, masks, splits, learning
rates, Adam settings, gate constraints and training batches come from the
pinned parent protocol. Every loss query uses only its original training
minibatch. Reused forward values are those the perturbed coordinate cannot
change. Coordinates with a zero slope still advance their optimizer clocks.

Each of two fresh builds trains all 24 endpoints before held-out scoring.
Compare the twelve standard minibatch models and their training receipts
to the earlier cohort. Report the original success criteria, 99% raw-object
and exact-sequence flags on every held-out split, and the unseen-answer
counts. The runner records fit time, CPU time and peak memory beside the
numeric receipts. [Frozen comparison](protocol.json)

```sh
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  <producer-commit> /run/media/misha/storage/transformer-runs/composed-retrieval/<fresh-run> dense
cargo run --release --locked --offline --features dense-forward \
  --manifest-path experiments/composed_retrieval/Cargo.toml --bin audit-raw -- \
  "$PWD" /run/media/misha/storage /run/media/misha/storage/transformer-runs/composed-retrieval/<fresh-run>
```
