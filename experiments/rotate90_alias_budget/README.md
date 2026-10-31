# Duration, decay and width after the alias interventions

The first 50 interventions reached 96--98% on two P193 seeds without crossing
99%. The user requested longer training and larger-model controls. This adaptive
follow-up records its [hypotheses](hypotheses.json) before its runs.

| Cases | Initialization | Updates | Decay | Comparison |
| --- | --- | ---: | --- | --- |
| 18 | Six original width-16 queries, fresh optimizer histories | 20000 | 1, .1, 0 | Duration at fixed checkpoint; decay at matched duration |
| 6 | Fresh width-64 models, same six task/seeds | 5000 | 1 | Original fresh width-16 runs, same update budget and rates |

Tasks are P97/K2 and P193/K3; seeds 11, 17, 23. Every model has one block.
Learning rate .003, betas .9/.98, batch 512, a 10-update inclusive warmup and the
same half-table split are retained. Only Schedule-Free AdamW is used, with
Python 3.15, Torch 2.15, CUDA and compile max-autotune. Long runs start from the
original checkpoints, preserving the prior 5000-update controls unchanged.
All 24 runs completed: 10 reach the inherited 99% final heldout threshold,
including all six width 64 controls and four width 16 runs. All accuracies below
use the final Schedule-Free evaluation average; best epochs are not substituted.

| Task | Seed | Fresh W16, 5k, decay 1 | W16 restart, 20k, decay 1 | W16 restart, 20k, decay .1 | W16 restart, 20k, decay 0 | Fresh W64, 5k, decay 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P97/K2 | 11 | 31.14% | 31.81% | 30.18% | 27.99% | 100% |
| P97/K2 | 17 | 31.59% | 30.63% | 55.58% | 31.89% | 99.98% |
| P97/K2 | 23 | 31.64% | 31.23% | 43.49% | 26.63% | 99.96% |
| P193/K3 | 11 | 11.79% | 11.79% | 85.87% | 92.12% | 100% |
| P193/K3 | 17 | 58.14% | 75.58% | 99.41% | 99.74% | 100% |
| P193/K3 | 23 | 45.09% | 75.04% | 99.97% | 99.99% | 100% |

Width 16 has hidden size 43; width 64 has hidden size 171, with one block in both.
Fresh width comparisons match the original experiment definitions in every
field except width. Duration comparisons match each prior restart definition
in every field except 5000 versus 20000 updates. All heldout fingerprints match.
These are matched-update comparisons, not matched-FLOP comparisons.

For the two P193 runs behind the user's 96--98% observation, zero-decay accuracy
increases from 95.81% to 99.74% and from 98.39% to 99.989% between updates 5000 and 20000
of the same long run. Their first measured 99% points are updates 10250 and 5500.
The initially poor P193 seed 11 also improves from 7.90% to 92.12% within the long
zero-decay run. Thus a poor 5000-update result was not a permanent trap.

Every width 64 run is already above 99% at the 750-update evaluation, and finishes
at 99.96--100% by 5000 updates. The P97 width 16 restarts still do not solve the task
within 20000 updates; lower decay can improve fitting without matching that
heldout result. The small model's mathematical capacity is proved separately,
so these observations concern optimization and generalization rather than a
proof that the function requires width 64.

The 20000-update restarts see 2000 training epochs at P97 or about 540.55 at P193,
in addition to the original checkpoints' training. The fresh 5000-update runs
see 500 epochs or about 135.14 respectively. The lab's epoch counts account for
its smaller final minibatch.

[Results](results.json), [summary](summary.csv), [curves](curves.csv) and
[verified control comparisons](comparison.json) preserve all outcomes.
Independent repeats of the first 5000 updates differ by at most 0.473 percentage
points; duration effects above use measurements within one run.
CPU checkpoint replay differs from recorded compiled-CUDA final accuracy by
at most one heldout pair (0.0054 percentage points), recorded separately.
These are adaptive, three-seed results at two tasks, not a global convergence
claim or a search for the smallest sufficient trainable model.

Pretraining definition SHA256:
`dbab0058fc8330a1c8556c1195659d263dd347c59c70bf41799e46ca8955c602`.
Hypothesis SHA256:
`ca4959d6ebc85fe7eda3b9ab2c02c2c1d1642c8e4716198378b1dfd950dfb2a1`.

```sh
./run.py check experiments/rotate90_alias_budget
./run.py run experiments/rotate90_alias_budget
./run.py report experiments/rotate90_alias_budget
# From python/:
uv run --locked python ../experiments/rotate90_alias_escape/analyze.py budget
# From the repository root:
python3 experiments/rotate90_alias_budget/compare.py
```
