# Softmax against sparsemax attention on associative recall

Does a small RoPE transformer learn multi-query associative recall (MQAR)
with softmax attention, and does it learn it at more lengths and rates when
sparsemax replaces softmax and nothing else? These are the trained
transformers of the [convex MQAR comparison](../archive/convex_mqar): its
`full` profile with its attention ablation, and its `sanity` profile, in
the lab's language and on the lab's MQAR. Its convex construction and
certificate are not ported.

## Runs

A sequence of `length` tokens is the lab's `MQAR` at the comparison's
sizes: after BOS it binds `length / 4` distinct keys, out of 4,096 key
tokens, to values out of 4,096 value tokens, in adjacent pairs. Each key
recurs once later, at a position p drawn with weight p^-0.1, and the model
is scored on the value it predicts there; every other token is a random
value. A run trains on 100,000 sequences for 64 epochs and is observed on
3,000 validation sequences after every epoch; the best model, by the share
of validation sequences with every query answered, then balanced accuracy,
then loss, is evaluated once on 3,000 test sequences. A run records the
first observation at which 99% of the validation queries are answered, and
`report` compares the runs that differ in their rate alone: the best
observation, the first to reach 99%, and the first to reach it and hold it.
Every seed is 0.

The model is the comparison's `RopeTransformer`: two pre-norm layers of
width 64 with one head of interleaved RoPE (base 10,000), LayerNorm, a
tanh-GELU feed-forward layer of width 256 with biases, a tied readout and
GPT-2's initialization at 0.02. It trains under AdamW with betas
(0.9, 0.999), epsilon 1e-8 and decay 0.1 on matrices alone, the rate warmed
up linearly over the first tenth of the updates, counting the update being
made. A batch, and an evaluation chunk, holds 64 sequences, 16 at length
256 and 8 at length 512.

| Labels | Attention | Lengths | Rates |
| --- | --- | --- | --- |
| `softmax-n<length>-lr<rate>` | softmax | 64, 128, 256, 512 | 1e-4, 4.6e-4, 2.2e-3, 1e-2 |
| `sparsemax-n<length>-lr<rate>` | sparsemax | 64, 128, 256, 512 | 1e-4, 4.6e-4, 2.2e-3, 1e-2 |
| `sanity` | softmax | 8, out of 16 tokens | 3e-3 |

The rates are `np.logspace(-4, -2, 4)`, labeled to eight significant
digits: `lr0.0001`, `lr0.00046415888`, `lr0.0021544347` and `lr0.01`.
`sanity` is a model of width 32, trained on 2,048 sequences out of 8 key
and 8 value tokens and observed and tested on 512 each, for 128 epochs.
Every run trains in float32, replays from CUDA graphs, and keeps a
checkpoint after every epoch. At length 512 the training sequences alone
take 1.2 GB of device memory.

## Running

```sh
./make.py check experiments/mqar_sparsemax              # the runs, and what differs between them
./make.py run experiments/mqar_sparsemax [label ...]    # train every run, or the labeled ones
./make.py report experiments/mqar_sparsemax [label ...] # what the runs recorded, as JSON
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again.

## Archived runs

The comparison's records are in [`../archive/convex_mqar`](../archive/convex_mqar),
and its code at commit `9416d03`, under `experiments/convex_mqar/`. Its runs
(RTX 3050, bf16 autocast, fused AdamW, the loss compiled though some softmax
epochs ran eagerly) covered lengths 64, 128 and 256, none 512. The first
epoch at which each reached 99% validation accuracy
([`validation_milestones_all_lr.csv`](../archive/convex_mqar/reports/validation_milestones_all_lr.csv)):

| Length | Softmax, at rates 1e-4 / 4.6e-4 / 2.2e-3 / 1e-2 | Sparsemax, at the same rates |
| --- | --- | --- |
| 64 | never; at most 0.25 | 3 / 2 / 1 / 1 |
| 128 | never / never / never / 18 | 9 / 5 / 3 / 46 |
| 256 | never / never / 5 / never | 9 / 6 / 13 / never |

Softmax reached 99% in 2 of its 12 runs and held it to the last epoch in
neither; at length 128 it fell to 0.0065. Sparsemax reached it in 11 of 12
and held it in 8; under rate 1e-2 it fell to 0.02 at length 64 and to 0.004
at length 128. The sanity run reached validation and test accuracy 1.0
([`rope_sanity.json`](../archive/convex_mqar/reports/rope_sanity.json)).

Here a sequence opens with BOS and is drawn by the lab's MQAR, not by the
comparison's NumPy generator; the best observation ranks first by whole
sequences answered, not by queries; every run trains in float32 and replays
from CUDA graphs; and the initial model is observed too. So none is its
archived run bit for bit.
