# Stability of GPTMini on x / y mod 193

The harder task the stability protocols of the synthetic trainers turned to
after [`mod97_stability`](../mod97_stability): does GPTMini under AdamW
grok x / y mod 193 and stay generalized, and what do sparsemax attention
weights or an annealed rate change? The historical table records the
2026-10-02 `adamw_stability` protocols. Step 5 of the current
[experiment plan](../../EXPERIMENT_PLAN.md) adds fresh lab controls and the
best starting-scale attempt from the basis, QKNorm-one. Its question is
whether this attempt generalizes here while retaining sparse attention.

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
| `repair-qknorm-one` | 25% | 3e-4 | 300,000 | sparsemax; learned QKNorm scale starts at 1 | constant |
| `repair-qknorm-one-seed1` | 25% | 3e-4 | 300,000 | the same attempt, fresh model and data seeds 1 | constant |
| `cosine` | 25% | 3e-4 | 300,000 | softmax | annealed by a half cosine to a tenth over updates 150,000–250,000 |
| `lr001` | 25% | 1e-3 | 150,000 | softmax | constant |
| `fraction50-lr001` | 50% | 1e-3 | 150,000 | softmax | constant |

The last two are the earlier calibrations. `base` is the calibration at rate
3e-4, extended to 300,000 updates, and `sparsemax` and `cosine` each change
one thing in it. Every run samples per-tensor diagnostics at each evaluation
and at the updates beside it, and records every gradient norm. Eager
execution issues the updates as the historical trainer did. The two
controls and the QKNorm-one variants also observe attention on 256 fixed
held-out examples at initialization and every observation, including
diagnostic neighbors. The ordinary sparsemax backward, XSA, data fraction,
optimizer, batch size and schedule remain those of the sparsemax control.
The seed-1 repeat has model seed 1, data seed 1 and the default batch-order
seed 10,001; it is trained only if the first attempt meets confirmation.

## Fresh confirmation

All 60 basis repair trials have finished. QKNorm-one matches 17 of 22
required softmax passes, against 12 for ScaledDot, and rescues one original
persistent recall seed. Neither is a complete basis repair; the label
above tests the better partial attempt rather than asserting a repair.
See [the basis results](../basis_sparsemax/README.md#starting-scale-and-score-map-repairs).

The archived table below is not evidence that the lab has trained these
labels. Step 5 trains `base`, `sparsemax` and `repair-qknorm-one` afresh for
their complete 300,000-update budgets. Only the diagnostic observer is
added to the existing controls; the attempt changes their score block's
initial scale. The frozen abandoned schedule pair is not resumed.

Confirmation requires 20 consecutive canonical evaluations with both
train and held-out accuracies at least 99%, and a final accuracy of 100%
on both. Neighbor probes do not count. The report separately records the
pre-target memorization plateau, all 201 canonical checks from updates
250,000 through 300,000, the number of failed joint checks and the worst
held-out accuracy there. Strict final-window persistence is not required
by step 5's confirmation rule: the archived softmax reference itself fails
two such checks. A confirmed attempt is repeated from fresh model and data
seeds before a broader success claim.

Fresh results are pending. The current lab descriptions and actual-shape
CUDA observation checks pass: all three main labels preserve parameters,
buffers, gradients, optimizer, sampler, CPU/CUDA RNG, modes, hooks and
inputs exactly around observations at updates 0, 10 and 20. These
temporary short runs provide no generalization evidence. All seven
descriptions check; the legacy calibration descriptions are unchanged,
the controls add only the observer, and the attempt changes only the
initial QKNorm scale relative to `sparsemax`. The seed-1 repeat changes
only model/data seeds. Before/after descriptions, verification source and
output, and their scope are retained in [preparation.json](preparation.json).
Final per-layer/head support and visible-zero statistics will accompany
the complete canonical histories. The full setup gate passes all 166 lab
tests (936.095 seconds).

## Running

```sh
./make.py check experiments/mod193_stability              # the runs, and what differs between them
./make.py run experiments/mod193_stability [label ...]    # train every run, or the labeled ones
./make.py report experiments/mod193_stability [label ...] # what the runs recorded, as JSON

./make.py run experiments/mod193_stability base sparsemax repair-qknorm-one
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

The fixed-checkpoint CPU inspection has been independently checked and
archived in [sparsemax-final-review](../archive/synthetic_trainers/protocols/adamw_stability_20261002/sparsemax-final-review/README.md).
Its source, observations, independent program/output and input hashes are
retained there. All four fixed-weight forward swaps and the selected local
derivative probes match the original records exactly. That inspection
performs no optimizer updates and makes no claim about the cause of the
training failure; it does not replace the fresh confirmation above.
