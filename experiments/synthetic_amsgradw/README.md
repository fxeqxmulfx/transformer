# GPTMini on the synthetic suite under AMSGradW

Which algorithmic tasks does a small GPTMini learn from a few hundred
examples, and which does it only memorize? This is the softmax baseline of
the synthetic trainers, `baseline.py` with its recipes in
`baseline_recipes.py`, and its studies, in the lab's language: 61 runs.

## Runs

GPTMini of width 64, with 2 layers of 4 heads and the context of its task,
trains under AMSGradW at rate 1e-3 with betas (0.9, 0.999), epsilon 1e-8
and weight decay 0.1, on 32 rows per update, and is observed in chunks of
32 rows. Every run is a memorization study at curve tolerance 0.02 on data
seed 1, and draws its batch order from its model seed, as the historical
trainer drew it.

| Labels | Runs | What they train |
| --- | --- | --- |
| `suite-<task>` | 37 | each task of the suite and each named control of it, 1,000 updates on 128 training rows from model seed 0, observed every 100 updates on 32 validation rows and tested on 64 |
| `transitions-<task>-seed<seed>` | 9 | copy, and parity with and without a running scratchpad, 5,000 updates on disjoint pools of 64 training, 64 validation and 128 test rows, observed every 250 and tested at 16 and 32 |
| `capacity-width<width>-seed<seed>` | 12 | parity on the same pools with 20% of its training labels corrupted (noise seed 2), 1,000 updates observed every 100, at widths 16, 32, 64 and 128 |
| `control-seed<seed>` | 3 | the random control, on 8 training rows, 64 validation and 128 test, 1,000 updates, tested at 16 and 32 |

The suite runs at lengths 8 to 16 and is tested at 32 and 64: MQAR and
composed lookup at length 24, with 4 associations and 2 queries, tested at
48 and 96; Dyck and typed Dyck of 2 and 3 bracket kinds from length 12;
alternating blocks; the histogram, with and without BOS, and the double
histogram; the mode, without a scratchpad, with counts or itemized; the most
frequent symbols; copy, reverse and sort, of repeating or unique symbols;
counting; addition of 2 to 4 digits in 8 variants, tested at 8 and 16;
parity in 4; Boolean AND with the zero early or anywhere; and C-RASP
formulas of depths 1, 2 and 3. `./make.py check` lists the labels, and
`./make.py show` describes a run. The studies run from model seeds 0, 1
and 2. Every run replays from CUDA graphs.

## Running

```sh
./make.py check experiments/synthetic_amsgradw              # the runs, and what differs between them
./make.py run experiments/synthetic_amsgradw [label ...]    # train every run, or the labeled ones
./make.py report experiments/synthetic_amsgradw [label ...] # what the runs recorded, as JSON
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again.

## Archived runs

The archived runs (RTX 3050, 2026-10-01, issued eagerly) are
[`baselines/amsgradw_softmax_20261002`](../archive/synthetic_trainers/baselines/amsgradw_softmax_20261002)
of the [synthetic trainers](../archive/synthetic_trainers), whose code is at
commit `5d64147`, under `experiments/synthetic_trainers/`. They fit their
training splits, but for two parity runs of the suite, one run at width 128
under label noise, and the random controls, which fit 1 of their 8 training
rows. Few generalized in distribution:

| Runs | Test sequence accuracy of the last model |
| --- | --- |
| suite: Boolean AND, both | 1.0 |
| suite: Dyck; alternating blocks | 0.95; 0.92 |
| suite: C-RASP of depth 1, 2, 3 | 0.91, 0.91, 0.70 |
| suite: typed Dyck of 2 and 3 kinds | 0.64, 0.66 |
| suite: parity without a scratchpad | 0.56, near the chance of its one-bit answer |
| suite: mode without a scratchpad | 0.39 |
| suite: every other task, MQAR and lookup included | at most 0.19 |
| transitions: parity with a running scratchpad | 0.91 to 0.99 |
| transitions: parity without it | 0.35 to 0.44 |
| transitions: copy | at most 0.04 |
| capacity: every width | 0.35 to 0.56, without a trend in width |

The runs replay from CUDA graphs here, so none is its archived run bit for
bit.
