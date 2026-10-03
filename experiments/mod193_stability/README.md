# Stability of GPTMini on x / y mod 193

The harder task the stability protocols of the synthetic trainers turned to
after [`mod97_stability`](../mod97_stability): does GPTMini under AdamW
grok x / y mod 193 and stay generalized, and what do sparsemax attention
weights or an annealed rate change? These are the 2026-10-02 runs of the
`adamw_stability` protocols on mod 193, in the lab's language.

## Runs

An equation is `a / b = c` in the field of 193 elements, `b` nonzero: 37,056
of them, both `c` and the end token predicted. GPTMini is that of
`mod97_stability`, with 436,104 parameters for the larger vocabulary, under
PyTorch's AdamW with betas (0.9, 0.98), weight decay 0.1 and the rate warmed
up over 10 updates. A run trains in batches of 512 from model and data seed
0, an epoch ending with a batch of the equations left over, and is
evaluated on both splits every 250 updates.

| Label | Train | Rate | Updates | Attention weights | Rate after warmup |
| --- | --- | --- | --- | --- | --- |
| `base` | 25% | 3e-4 | 300,000 | softmax | constant |
| `sparsemax` | 25% | 3e-4 | 300,000 | sparsemax | constant |
| `cosine` | 25% | 3e-4 | 300,000 | softmax | annealed by a half cosine to a tenth over updates 150,000–250,000 |
| `lr001` | 25% | 1e-3 | 150,000 | softmax | constant |
| `fraction50-lr001` | 50% | 1e-3 | 150,000 | softmax | constant |

The last two are the earlier calibrations. `base` is the calibration at rate
3e-4, extended to 300,000 updates, and `sparsemax` and `cosine` each change
one thing in it. Every run samples per-tensor diagnostics at each evaluation
and at the updates beside it, and records every gradient norm. Eager
execution issues the updates as the historical trainer did.

## Running

```sh
./make.py check experiments/mod193_stability              # the runs, and what differs between them
./make.py run experiments/mod193_stability [label ...]    # train every run, or the labeled ones
./make.py report experiments/mod193_stability [label ...] # what the runs recorded, as JSON
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again.

## Archived runs

The criterion is that of [`mod97_stability`](../mod97_stability#archived-runs),
from [`STABILITY.md`](../archive/synthetic_trainers/STABILITY.md): memorized,
then confirmed for 20 evaluations, then at 99% at each of the 201
evaluations of the last 50,000 updates.

| Run | Memorized | Confirmed | Last 50,000 updates: failed, worst held-out | At the end, train / held-out |
| --- | --- | --- | --- | --- |
| [`base`](../archive/synthetic_trainers/protocols/adamw_stability_20261002/budget300k-result.md) | 4,500–27,250 | 95,750–100,500 | 2, 49.18% | 100% / 100% |
| [`sparsemax`](../archive/synthetic_trainers/protocols/adamw_stability_20261002/attention-pair-result.md) | 49,750–255,750 | never | 201, 2.51% | 100% / 34.06% |
| `cosine` | not run | | | |
| [`lr001`](../archive/synthetic_trainers/protocols/adamw_stability_20261002/fraction25-result.md) | 4,500–11,500 | 20,000–24,750 | 3, 48.92% | 100% / 100% |
| [`fraction50-lr001`](../archive/synthetic_trainers/protocols/adamw_stability_20261002/larger-modulus-result.md) | no | 52,250–57,000 | 7, 79.90% | 100% / 100% |

None groks stably. On 25% of the equations GPTMini memorizes first at
either rate, which on 50% at rate 1e-3 it did not; but at 1e-3 three
evaluations of the last 50,000 updates fail, and at 3e-4 two do, at updates
274,000 and 275,500. The first 150,000 updates of the archived `base` were
the calibration at 3e-4, and eight evaluations of their last 50,000 had
failed; after it confirmed, `base` failed in six episodes, and each
recovered within 1,250 updates. Sparsemax fits every training equation, but
memorizes for 206,000 updates and never generalizes: its held-out accuracy
peaks at 35.48%, at update 299,500. The project's
[experiment plan](../../EXPERIMENT_PLAN.md) takes up that gap.

Each label links the result note of its archived run, which links its
measurements in the [baselines](../archive/synthetic_trainers/baselines).
The archived `base` is the
[150,000-update calibration](../archive/synthetic_trainers/protocols/adamw_stability_20261002/lower-rate-result.md)
continued to 300,000 updates with its optimizer, random and sampling state
restored; the softmax control of the sparsemax pair, trained afresh, recorded
the same 1,201 evaluations but for their times. `cosine` never ran: the
schedule pair, `base` afresh and then `cosine`, was stopped on 2026-10-03
by the user's decision, with the constant case at update 288,000 and the
cosine case not started
([`EXPERIMENT_PLAN.md`](../../EXPERIMENT_PLAN.md#abandoned-schedule-pair)).
