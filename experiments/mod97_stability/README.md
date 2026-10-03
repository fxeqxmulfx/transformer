# Stability of GPTMini on x / y mod 97

Once GPTMini generalizes x / y mod 97, does it stay generalized? The
stability protocols of the synthetic trainers asked it of AdamW and of raw
AMSGradW at one recipe, and of four changes to the AMSGradW arm. These are
the 2026-10-02 optimizer pair of the `adamw_stability` protocols and the
calibrations of the `amsgradw_stability` protocols, in the lab's language.

## Runs

The task and GPTMini are those of [`mod97_grokking`](../mod97_grokking),
here on 50% of the 9,312 equations. A run trains in batches of 512 for
150,000 updates from model and data seed 0, at rate 1e-3 warmed up over 10
updates with weight decay 0.1, and is evaluated on both splits every 250
updates. An epoch ends with a batch of the 48 equations left over, unless
the batches wrap across epochs.

| Label | Optimizer | Rate | Batches |
| --- | --- | --- | --- |
| `adamw` | AdamW, betas (0.9, 0.98) | 1e-3 | short last batch |
| `amsgradw` | raw AMSGradW, betas (0.9, 0.999) | 1e-3 | short last batch |
| `amsgradw-wrap` | raw AMSGradW | 1e-3 | 512 each, wrapping |
| `amsgradw-lr0003` | raw AMSGradW | 3e-4 | short last batch |
| `amsgradw-lr0002` | raw AMSGradW | 2e-4 | short last batch |
| `amsgradw-lr0001` | raw AMSGradW | 1e-4 | short last batch |

AdamW is PyTorch's, with bias correction. Raw AMSGradW keeps the maximum of
the second moment and corrects no bias. Both multiply the weights by
`1 - rate * decay` every update, so a lower rate also decays less, and the
calibrations do not separate the two. Every run samples per-tensor
diagnostics at each evaluation and at the updates beside it, and records
every gradient norm; the historical calibrations of the AMSGradW arm did not
record gradient norms, and no diagnostic changes a trajectory. Eager
execution issues the updates as the historical trainer did.

## Running

```sh
./make.py check experiments/mod97_stability              # the runs, and what differs between them
./make.py run experiments/mod97_stability [label ...]    # train every run, or the labeled ones
./make.py report experiments/mod97_stability [label ...] # what the runs recorded, as JSON
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again.

## Archived runs

[`STABILITY.md`](../archive/synthetic_trainers/STABILITY.md) of the
synthetic trainers sets the criterion. A run groks stably when it first
memorizes, at train accuracy at least 99% and held-out accuracy at most 10%
for five consecutive evaluations spanning at least 1,000 updates, before
held-out accuracy first reaches 99%; then confirms, with both at 99% for 20
consecutive evaluations; and then persists, with both at 99% at each of the
201 evaluations of its last 50,000 updates.

| Run | Memorized | Confirmed | Last 50,000 updates: failed, worst held-out | At the end, train / held-out |
| --- | --- | --- | --- | --- |
| [`adamw`](../archive/synthetic_trainers/protocols/adamw_stability_20261002/adamw-result.md) | no | 1,250–6,000 | 7, 43.90% | 100% / 100% |
| [`amsgradw`](../archive/synthetic_trainers/protocols/adamw_stability_20261002/raw-amsgradw-result.md) | no | 52,750–57,500 | 10, 58.14% | 100% / 99.98% |
| [`amsgradw-wrap`](../archive/synthetic_trainers/protocols/amsgradw_stability_20261002/wrap-lr001-result.md) | no | 24,250–29,000 | 26, 1.98% | 100% / 100% |
| [`amsgradw-lr0003`](../archive/synthetic_trainers/protocols/amsgradw_stability_20261002/short-lr0003-result.md) | no | 3,000–7,750 | 2, 40.55% | 100% / 100% |
| [`amsgradw-lr0002`](../archive/synthetic_trainers/protocols/amsgradw_stability_20261002/midpoint-result.md) | no | 95,750–100,500 | 150, 0.79% | 100% / 88.66% |
| [`amsgradw-lr0001`](../archive/synthetic_trainers/protocols/amsgradw_stability_20261002/short-lr0001-result.md) | 1,000–61,250 | never | 201, 17.53% | 31.27% / 17.53% |

None groks stably. AdamW and raw AMSGradW both generalize without
memorizing first, held-out accuracy reaching 99% at update 1,250 and 5,250,
and both fall below 99% inside their last 50,000 updates however they end.
Full batches fail more often, not less. At rate 3e-4 AMSGradW fails twice;
at 2e-4 it generalizes only at update 93,250 and fails at every evaluation
from 112,750 on; at 1e-4 it memorizes but never generalizes, its held-out
accuracy peaking at 67.16%, and its train accuracy falls from 100% to
31.27% over the last 250 updates.

Each label links the result note of its archived run, which links its
measurements in the [baselines](../archive/synthetic_trainers/baselines);
the [pair's comparison](../archive/synthetic_trainers/protocols/adamw_stability_20261002/optimizer-pair-result.md)
sets `adamw` beside `amsgradw`. The AMSGradW calibration
[`short-lr001`](../archive/synthetic_trainers/protocols/amsgradw_stability_20261002/short-lr001-result.md)
is `amsgradw` without the gradient trace, and failed at the same ten
evaluations. On 2026-10-02 the user chose AdamW for the continuing
benchmark, and the protocols moved to the harder
[`mod193_stability`](../mod193_stability).
