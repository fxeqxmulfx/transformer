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
