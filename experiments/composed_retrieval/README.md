# Composed learned retrieval on four original GPT stress tests

Can one trainable model predict an unseen binding, follow a second lookup and
learn a two-operand rule from immutable object sequences? The previous head
solved binding, missed two-hop lookup and lost the second arithmetic operand
at every parameter setting. CM113 proves that the new model can express both
a genuine two-hop change of answer and all 9,409 original modular sums.

| Original case | Training examples | Batch | Held-out criterion |
| --- | ---: | ---: | --- |
| `gpt-probe-key-value-binding` | 2,048 | 32 | 99% exact on novel validation, plus all original splits |
| `suite-mqar` | 128 | 32 | 95% exact on novel validation, with length-48 and length-96 probes |
| `suite-lookup` | 128 | 32 | 95% exact on novel validation, with length-48 and length-96 probes |
| `arithmetic-add` | 4,704 | 256 | 99% exact on the original 4,705 held-out pairs |

Each case trains a fresh instance of the same width-eight physical model.
Input IDs address learned tables. The model sees only the causal prefix. Both
reads attend to every visible memory position. The readout scores the entire
original vocabulary. Eight angular coordinates per object are learned for
each of the key, tied value/readout and rotation maps. Training also fits the
boundary state, predecessor and relative potentials, recency, key scale and
both gates. With vocabulary size V and maximum context L ≥ 2 this uses
3VD + D + (V+1) + (2L−3) + 4 real parameters. On a memory of n objects a
factored forward costs O(n²D + VD) and stores O(n² + VD) working values.

The prospectively fixed comparison uses data seed 1 and model seeds 0, 1 and
2. Each run takes 1,024 updates. Ordinary Mini differentiates full-vocabulary
cross-entropy. Forward-only Mini estimates coordinate updates from feasible
pairs of complete minibatch-loss evaluations. Both train on the original
training examples and supervised positions. Two builds must seal all their
training endpoints before scoring. The separate study target requires 99%
raw and exact on every applicable held-out split, in both modes at every
seed. Report per-answer training exposure and the original binding-pair
diagnostics. This protocol has no adaptive choice of seed, fit or stopping
point. The frozen settings are in [protocol.json](protocol.json).

The learned geometry, margin assumptions and indistinguishable training
completions are stated in [CM113](../../review/flat_spectrum.md#cm113-composed-learned-retrieval-and-a-two-operand-phase-interaction).

## Native reproduction

Use the commit containing the native producer and an unused directory on the
external storage volume. The runner archives tracked inputs, builds twice with
Cargo offline and trains each endpoint in its own process. Each score command
verifies the training seals of both complete arms and compares their weights
and optimizer receipts. The replay recomputes all logits and original metrics.
Generated data, raw scores and logs remain in the specified output directory.

```sh
bash experiments/composed_retrieval/run.sh "$PWD" /run/media/misha/storage \
  <producer-commit> /run/media/misha/storage/transformer-runs/composed-retrieval/<fresh-run>
```

The Rust minibatch implementation caches the V² pairwise key similarities and
one n² relative kernel per distinct memory length in a batch. Forward-only
loss queries reuse forward values unaffected by a changed weight. A changed
readout for an object absent from memory still contributes to the complete
class partition on every example. The regression compares each projected
coordinate query to recomputation from the literal nested path sums.

## Results at the frozen endpoint

Producer `0f0aadbd` ran all four cases at seeds 0, 1 and 2 in both modes.
Binding and multi-query recall met their original selection targets at every
seed in both modes. Binding alone met the separate 99% raw-object and
exact-sequence goals on every held-out split at all six endpoints. The 161
other catalog cases have no result for this model and protocol.

| Original held-out measurement | Standard minibatch, seeds 0 / 1 / 2 | Forward-only minibatch, seeds 0 / 1 / 2 |
| --- | --- | --- |
| Binding, novel validation and test | 512/512 on each split at every seed | 512/512 on each split at every seed |
| Modular addition, validation | 4,705 / 4,705 / 4,705 correct of 4,705 | 79 / 64 / 82 correct of 4,705 |
| Multi-query recall, novel validation | 32 / 32 / 31 exact of 32 | 32 / 32 / 32 exact of 32 |
| Multi-query recall, length-24 and length-48 tests | 64/64 exact at each length and seed | 64/64 exact at each length and seed |
| Multi-query recall, length-96 test | 64 / 64 / 63 exact of 64 | 52 / 57 / 56 exact of 64 |
| Two-hop lookup, novel validation | 5 / 8 / 4 exact of 32 | 6 / 2 / 4 exact of 32 |
| Two-hop lookup, length-24 test | 7 / 14 / 10 exact of 64 | 8 / 8 / 6 exact of 64 |
| Two-hop lookup, length-48 test | 5 / 5 / 6 exact of 64 | 5 / 0 / 0 exact of 64 |
| Two-hop lookup, length-96 test | 0 / 3 / 0 exact of 64 | 0 / 0 / 0 exact of 64 |

On addition, standard minibatch also got every one of the 4,704 training
pairs correct. Its learned product gate reached one at all three seeds.
At each seed the learned value and rotation tables implemented the correct
winner over all 9,409 ordered pairs from their independent random starts.
Forward-only Mini answered just 90, 74 and 99 training pairs correctly.
Its product gates reached 0.590–0.609. Each key, value and rotation angle
received 21 or 22 feasible loss-difference updates, while standard Mini
visited every coordinate 1,024 times. The comparison leaves open what
forward-only fitting learns with matched coordinate visits.

All six binding endpoints correctly answered the 71 previously unsupervised
queried pairs on novel validation and the 65 on test. Among those, nine and
five, respectively, were absent as stored pairs throughout training.
On multi-query recall, object 132 occurred 33 times in training inputs and
zero times as a positive target. Every endpoint answered its single length-24
test query correctly. On the length-96 test, the forward-only endpoints
answered both queries for that object correctly and missed 13, 8 and 8
of the other 126 queries. Standard minibatch seeds 0 and 1 answered every
query on every split, while seed 2 missed one of those two queries at
length 96 and one sequence on novel validation.

All six two-hop endpoints learned zero for both the depth and product
gates. Their predictions used exactly the first read at the frozen endpoint.
On the length-24 test, 118 of 128 answers had appeared as training targets.
Standard minibatch got 46, 55 and 51 of those 118 correct, and forward-only
got 45, 45 and 41. Missing positive labels therefore explain only a small
part of the failure. CM113 proves a second read is sufficient given learned
routing gaps, but the trained endpoints put zero weight on this route.

The two builds sealed 48 independently trained endpoints before evaluation.
Their model parameters, training traces, original metrics and raw logits
matched. Independent decoding checked all 672 raw files, including random
initializations. It recovered the original CE to within 1e−12, the raw counts, exact counts
and supervised-answer exposure groups, and reconciled all 165 catalog IDs.
The separate earlier CM111 study remains documented in
[learned retrieval transfer](../learned_retrieval_transfer/README.md).

The reviewed raw evidence is under
`/run/media/misha/storage/transformer-runs/composed-retrieval/original-four-v1`.
Rebuild the cohort with the preceding command at producer `0f0aadbd`.
The following command independently decodes its retained scores and prints the full case ledger.

```sh
cargo run --locked --offline --release --manifest-path experiments/composed_retrieval/Cargo.toml \
  --bin audit-raw -- "$PWD" /run/media/misha/storage \
  /run/media/misha/storage/transformer-runs/composed-retrieval/original-four-v1
```
