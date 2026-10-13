# Grokking progress in ordinary softmax transformers

Can a measurement made during ordinary transformer training expose rule
formation before held-out accuracy rises, and distinguish it from loss
improvement that never leads to generalization?

All runs use softmax and native AdamW, with 150,000 updates, batch 512,
learning rate 0.001, betas (0.9, 0.98), ten-update warmup, exhaustive
evaluation every 250 updates and structural observation every 1,000.
The task is division modulo 97. Observer overhead is recorded separately
from training time. Measurements never supply model inputs or gradients.

| Run | Model | Training fraction | Weight decay | Model/data seeds | Role |
| --- | --- | ---: | ---: | --- | --- |
| `gptmini-seed1` | GPTMini, 128 wide, two layers, four heads | 0.5 | 0.1 | 1/1 | Archived delayed generalization |
| `gptmini-seed2` | Same | 0.5 | 0.1 | 2/1 | Archived immediate generalization control |
| `gptmini-seed3` | Same | 0.5 | 0.1 | 3/1 | Independent initialization |
| `reference-fraction20` | openai/grok decoder, same width/depth/heads | 0.2 | 1.0 | 0/0 | Archived memorization without generalization within this budget |

GPTMini runs use the local CUDA GPU; the reference control uses four CPU
threads so it can train concurrently. Hardware and floating-point paths
are recorded. The control is a new run, not assumed to repeat an archived
trajectory exactly.

Recipes repeat [mod97_grokking](../mod97_grokking/README.md). The negative
control changes architecture, data fraction and regularization together;
it tests detector specificity, not a causal architectural comparison.
The recipes and thresholds are selected using existing runs. Fresh runs
repeat known seeds to validate instrumentation; they are not independent
evidence for a fitted universal predictor.

## Measurement

For each current model, collect answer logits on every raw five-token
prompt, excluding the answer token. The symmetry
`(x, y) -> (u*x, u*y)`, for nonzero `u`, preserves `x/y` in the prime field.
Average logits over each orbit to obtain an orthogonal functional
projection. It retains all invariant functions, including incorrect ones.

The main energy fraction `S` uses **held-out logits only**. Remove
per-example scalar shifts and the global class bias, and decompose the
remaining logits into an orbit mean and a within-orbit residual:

`S = invariant_energy / (invariant_energy + residual_energy)`.

Zero quotient is excluded from the main energy, preventing its easy
shortcut from dominating the score; it remains in the loss summaries.
Constant logits give `S = null`, not one. At least two held-out examples
are required in every nonzero orbit. Pooling training logits into the
held-out projection would let memorization create spurious projected
success; separate full-domain projection losses are recorded for audit.

The held-out-only restricted loss checks whether the projected part gives
correct answers. Excluded training loss removes the full-domain orbit
component. This follows the restricted/excluded-logit idea in Nanda et al.,
[arXiv:2301.05217v1](https://arxiv.org/abs/2301.05217), section 5.1.
Deviation: division scaling replaces selected addition frequencies, with
no final-checkpoint selection and no claim to identify a particular
attention or FFN circuit. The local manuscript is in
`papers/arXiv-2301.05217v1/iclr2023_conference.tex`.

## Causal phase evidence

`./make.py report experiments/grokking_progress` includes `grokking_progress`.
Every reported point uses only observations available at that update.
The report stores all thresholds and distinguishes four events:

- `loss_only_alarm`: held-out loss falls after confirmed memorization.
- `structure_forming`: held-out `S` rises and restricted loss falls.
- `cleanup`: the invariant part dominates, restricted loss is low and
  residual energy falls.
- `observed_delayed_generalization`: both splits exceed 99% for five
  observations after a recorded memorization plateau.

These are study heuristics and current evidence, not a guarantee of future
grokking. Immediate generalization is not called grokking. Record false
alarms and lead times before judging the detector.

## Found

Twenty-three focused observer/report tests pass on CPU and the local CUDA GPU. Actual AdamW
model, optimizer and sampler checkpoints, canonical metrics and resumed
trajectories are unchanged by the observer. A constructed training-only
memorizer has full-domain restricted loss below 0.01 but held-out-only
restricted loss `log(5)` and no structural signal. A wrong invariant rule
has `S = 1` and zero answer accuracy, confirming why correctness is needed.
The complete `./make.py test` suite also passes all 233 tests.

Existing mod-193 histories already contain a counterexample to using a
held-out loss slope alone: the sparsemax run records falling loss but
never reaches 99% within its full 300,000-update budget. Those histories
have no intermediate-model orbit measurements. Fresh softmax runs are
needed to judge the new structural indicator.

[Archived evaluation](archived_loss_baseline.json) covers nine mod-97
AdamW histories and four mod-193 histories, with source hashes and full
budgets. The loss-only alarm at 75,500 on mod-193 sparsemax is the sole
alarm without observed delayed generalization. On delayed GPTMini
seed 1 it fires at 35,500; five-observation 99% generalization is confirmed
at 50,750. Seeds 2 and 3 generalize immediately and produce no grokking
alarm. This validates the causal accounting, not the new structure signal.

Internal probes read actual attention outputs, FFN outputs and residual
streams at the answer query, with held-out orbit energy measured separately
for every block. They also record each attention head's entropy and mean
weights over the five prompt positions. Temporary hooks are removed even
after a failed forward. Norm reports group actual weights, gradients and
updates by embedding, attention and FFN; QKNorm head gains are retained.
Norms depend on parameter scale and are not used as a success criterion.

The first `gptmini-seed1` run started with the initial output observer.
It completed with byte-identical source in `/tmp/transformer-grokking-study`;
its canonical output/norm records remain intact. `watch.py` reads its
atomic checkpoints and records the newly added internal measurements on
CPU, without updating training. These snapshots explicitly record both
the CPU observation device and CUDA training device, checkpoint hashes
and observer source hashes. Internal snapshots start at update 30,000;
earlier internal activations were not recorded. Future fresh runs collect
internals directly during their ordinary diagnostic forward.

Reproduce the archive table from `python/` with
`uv run --locked python ../experiments/grokking_progress/archived.py`.

## Fresh evidence and full-budget controls

Update: the primary seed 1 completed its full **150,000-update** budget
with **100% held-out answer accuracy**. The checkpoint-preserving repeat
and all six new offline measurements are in
[grokking_internals](../grokking_internals/README.md), with a reproduced
transition and raw-checkpoint/source fingerprints. The measurements below
describe the initial snapshot. The repeat matches all 601 canonical
observations of the primary exactly. The reference control completes
150,000 updates with 1.503% held-out answer accuracy. Seed 2 completes
150,000 at 100%, first exceeding 99% at 1,250; seed 3 first exceeds 99%
at 750 and completes 150,000 at 100%. These controls generalize early,
so they do not replicate the delayed transition of seed 1.

On the freshly trained ordinary GPTMini seed 1, the pinned structural
signal first appears at update **33,000**, with held-out answer accuracy
**14.39%**. The first 99% observation is at **35,500**, and five consecutive
99% observations confirm delayed generalization at **36,500**. The signal
leads the first 99% observation by 2,500 updates in this one run. The new
PyTorch/CUDA trajectory differs numerically from the archived seed run;
the old transition time is not substituted for this observation.

Two actual internal checkpoint observations explain why overall weight
norm alone would miss the large change:

| Measurement | Update 30,000 | Update 35,000 |
| --- | ---: | ---: |
| Overall parameter L2 norm | 44.64 | 44.73 |
| Embedding L2 norm | 39.01 | 37.94 |
| Attention parameter L2 norm | 10.30 | 12.24 |
| FFN parameter L2 norm | 19.10 | 20.28 |
| Block 1 attention held-out invariant fraction | 0.157 | 0.738 |
| Block 1 FFN held-out invariant fraction | 0.264 | 0.839 |
| Raw held-out answer CE | 9.082 | 0.0547 |
| Held-out-only projected answer CE | 3.361 | 0.000185 |

Block indices start at zero. At update 1,000, logit invariant fraction was
already 0.939 while held-out accuracy was only 47.66%, demonstrating why
symmetry alone is insufficient. The detector also checks correctness,
confirmed fit, previous memorization and trends in past observations.
In the first 5,000 updates of the fresh reference control, held-out
accuracy is 1.36%, invariant fraction 0.0155 and projected CE 4.99; no
structural alarm has fired at that snapshot. The final full-budget control
result above does not prove absence of a transition beyond this budget.

[The compact snapshot](fresh_results.json) names the actual observed
updates, budgets, causal events, source hashes and internal checkpoints.
[Internal figure](internal_progress.svg), [fresh trajectories](fresh_progress.svg)
and [archived false alarm](archived_loss_baseline.svg) are generated with
Matplotlib. The internal comparisons are observations, not feature
ablations or evidence that those particular modules cause grokking.
All four budgets complete 150,000 updates. The additional answer/EOS
gradient decomposition and its pinned
checkpoint results are in [grokking_internals](../grokking_internals/README.md).

From `python/`, use `uv run --locked python
../experiments/grokking_progress/summarize.py` to refresh the snapshot and
`uv run --locked --with matplotlib==3.10.8 python
../experiments/grokking_progress/plot.py` to regenerate figures.
