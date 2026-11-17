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

## Results with matched visits

Producer `0c8436c1` learned all 4,705 held-out modular sums with forward-only
training at each seed. Both modes also got all 4,704 training pairs correct
and learned a product gate of one. The previous forward-only fit got 79,
64 and 82 validation pairs correct. Its angular coordinates received 21
or 22 visits at the same fixed endpoint.

| Original measurement | Standard minibatch, seeds 0 / 1 / 2 | Forward-only minibatch, seeds 0 / 1 / 2 |
| --- | --- | --- |
| Binding, novel validation and test | 512/512 on both at every seed | 512/512 on both at every seed |
| Addition, validation | 4,705/4,705 at every seed | 4,705/4,705 at every seed |
| Multi-query recall, novel validation | 32 / 32 / 31 exact of 32 | 32 / 32 / 31 exact of 32 |
| Multi-query recall, length-96 test | 64 / 64 / 63 exact of 64 | 64 / 64 / 63 exact of 64 |
| Two-hop lookup, novel validation | 5 / 8 / 4 exact of 32 | 5 / 8 / 4 exact of 32 |
| Two-hop lookup, length-24 test | 7 / 14 / 10 exact of 64 | 7 / 14 / 10 exact of 64 |
| Two-hop lookup, length-96 test | 0 / 3 / 0 exact of 64 | 0 / 3 / 0 exact of 64 |

Binding, addition and multi-query recall pass the original target at all
six endpoints. Binding and addition pass the separate 99% raw and exact
targets on every applicable held-out split, bringing this protocol to
two strict passes from four measured cases and 161 unmeasured. All six
binding fits got 65/65 previously unsupervised queried pairs and 5/5 pairs
absent from all training memories on test. The lookup depth gate again
ended at zero in every fit.

All twelve standard minibatch controls reproduced the previous weights,
optimizer traces and raw logits byte for byte. Comparing the two training
modes within this cohort found zero differences in 46,659 next-object
decisions over every original split at the three seeds, including original
split multiplicities. Their logits differed by up to 0.2433 on lookup. All
672 raw files from the independent builds passed full-vocabulary decoding.

| Case | Complete minibatch CE queries per forward-only fit | Forward-only elapsed seconds | Standard minibatch elapsed seconds |
| --- | ---: | ---: | ---: |
| Modular addition | 4,995,072 | 38.09–39.60 | 1.48–1.58 |
| Binding | 5,189,632 | 28.00–28.23 | 0.93–1.11 |
| Multi-query recall | 8,810,496 | 182.87–201.47 | 2.27–2.29 |
| Two-hop lookup | 5,533,696 | 119.73–120.78 | 1.88–1.96 |

Times include training-data preparation and receipt publication for each
process, with up to 18 concurrent fits on the shared CPU. The complete
original two-build run, through scoring and replay, took 339.92 seconds.
Peak RSS over fit processes was 13,860 KiB. Resource files are per process
and independent of the numerical seals.

The raw runs are in
`/run/media/misha/storage/transformer-runs/composed-retrieval/matched-visits-v1`.
Compare them with the sparse parent and recount every cross-mode decision
with the following command.

```sh
base=/run/media/misha/storage/transformer-runs/composed-retrieval
cargo run --release --locked --offline --features dense-forward \
  --manifest-path experiments/composed_retrieval/Cargo.toml --bin compare-visits -- \
  "$PWD" /run/media/misha/storage "$base/original-four-v1" "$base/matched-visits-v1"
```
