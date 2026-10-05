# Sparsemax on the basis

Where does sparsemax attention fail where softmax passes the calibrated
[basis](../basis/README.md), and is a failure removed by a neighboring rate?
This is steps 0 to 4 of [EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md).
The main arms change only the attention weights. Both keep QKNorm, XSA, the
basis's recipes, splits, seeds and thread counts. Hard parity is the easy
parity run, as in the basis.

| Labels | Runs | Difference |
| --- | ---: | --- |
| `softmax-<mode>-<model>-<task>-seed<seed>` | 30 | The basis repeated under softmax |
| `sparsemax-<mode>-<model>-<task>-seed<seed>` | 30 | Euclidean simplex projection instead of softmax |
| `time-<weights>-small-<task>` | 6 | 300 updates at the easy recipe, without early stopping, observed every 100 |
| `sparsemax-<mode>-<model>-<task>-seed<seed>-lr<rate>` | 60 | The two adjacent rates of `RATES`; only failures whose softmax control passes are trained |
| `probe-<weights>-hard-large-recall-seed<seed>` | 6 | Repeat the target cell with attention measurements on 256 fixed validation examples |
| `init-<weights>-<model>-seed<seed>-<scores>` | 24 | Initial hard-recall attention under QKNorm and ScaledDot, both sizes and weights; one zero-rate warmup update retains update-zero records |
| `forward-<weights>-backward-<weights>-hard-large-recall-seed<seed>` | 6 | Sparsemax forward with softmax score gradients, or softmax forward with sparsemax score gradients; other target settings unchanged |
| `repair-<candidate>-<mode>-<model>-<task>-seed<seed>` | 60 | Ordinary sparsemax with QKNorm initially at scale one, or with ScaledDot; unchanged recipes and fixed-row attention measurements |

`Diagnostics` samples every observed update: `loss` is the
last training batch's mean supervised cross-entropy before that update, and
`temperatures` holds each head's `log_alpha` and `inverse_temperature`
(e^alpha) after it. Update 0 has no training batch. Diagnostics also
records the existing per-tensor gradient, update and moment norms.

`AttentionDiagnostics` retains those measurements and adds `attention.jsonl`
at update zero and every observation, including requested diagnostic
neighbors. It uses the first 256
examples of the selected validation split, in their stored order; their
fingerprint is recorded. The forward is uncompiled and teacher-forced.
Every layer records actual and shadow-sparsemax statistics per head, for
all valid context rows, rows after BOS and supervised positions. Support
means weight strictly greater than zero; neither future nor padded
positions count as zeros. `mean_support_share` averages row shares,
whereas `exact_zero_pair_share` counts all visible pairs. Entropy is divided
by the log of the visible prefix length, with zero for length one.
Turnover counts changed memberships among the same visible pairs; the
initial value is null. Previous masks travel in checkpoints, and records
beyond a resumed checkpoint are rewound.

For recall, `queries` identifies each example, query position and latest
answer-value position, and marks its teacher-forced prediction correct or
wrong. Each layer's `answer_routes` supplies the weights at that column in
head order, their maximum and whether any head supports it; the sparsemax
shadow supplies the same fields. Repeated values at unrelated writes or
fillers do not count as the answer position. Self-only XSA routes also
count value norms below its normalization epsilon: the real-arithmetic
self-cancellation identity needs a norm at least that epsilon. Eager and
compiled or fused attention can round differently; fused softmax's
reported weights are its explicit unfused reference.

## Found

Preparation on 2026-10-04 used torch 2.14.1 on the basis's Xeon E5-2680 v4
CPU. Steps 0 and 1 use the lab sources of ea18d55; the run manifests keep their
hashes and this experiment's source. Sparsemax compiles in a full graph,
including its custom backward. Tests compare its weights and score
gradients at lengths 1, 17 and 64, and all logits and parameter gradients
of the small GPTMini at length 64, to eager evaluation within float32
rounding. A short one-thread compiled modular run under each normalizer
records identical observation metrics, model, optimizer and sampler state
with and without diagnostics, with explicit nonempty checkpoint checks;
the sparsemax engine also checks interruption and resumption against an
uninterrupted run. On the actual large depth and recall shapes at two and
four threads, tests check that every measurement leaves its inputs, model,
gradients, optimizer, sampler, global RNG and module training modes exactly
unchanged. The full lab suite also covers CUDA graphs and historical trainers.
The final step 1 check passes all 143 lab tests.

Each timing run trains 300 updates on one physical core at the easy recipe.
The table divides `training_seconds` by 300, excluding compilation,
evaluation and diagnostic time:

| Task | Softmax, ms/update | Sparsemax, ms/update | Sparsemax / softmax |
| --- | ---: | ---: | ---: |
| depth | 22.580 | 37.949 | 1.681 |
| recall | 84.349 | 144.095 | 1.708 |
| parity | 6.691 | 8.281 | 1.238 |

All are below step 0's factor-of-two ceiling. The timing controls'
validation records match bd63e50's archived basis exactly at every common
observation: updates 0 and 200 for depth and parity, and 0, 100, 200 and 300
for recall. The full benchmark will check the rest of each trajectory.
All 60 original benchmark runs and the 12 required neighboring-rate checks
are complete.

The large softmax controls preserve the archive's pass/fail set, but 14 of
their 15 full non-timing histories differ. This comparison therefore uses
the current softmax arm, as step 0 allows. Separate twenty-update large
depth/sparsemax runs without diagnostics also differ at two threads:
126 checkpoint fields differ, and validation loss at update 20 is
1.3659723997 versus 1.3660305738. A sampled repeat differs comparably
(1.3660308123). This establishes that the drift also occurs without
diagnostics; it does not identify its computational cause or guarantee
bit-identical resumption for large multi-threaded runs. The small controls'
exact archive repetition below remains a separate result. Descriptions,
checkpoint differences and record hashes of the three short runs are in
[diagnostic_repeatability.json](diagnostic_repeatability.json).

### Small model

All 30 original small-model runs have finished. The 15 softmax controls
repeat all 661 non-timing observations of the archived basis exactly;
their model, optimizer and sampler checkpoints are bit-identical too.
The update of each pass, or the best selection sequence accuracy of a
failure, is below. Batch losses are the last sampled training batch before
the final update, not an exhaustive training-set evaluation.

| Mode | Task | Seed | Softmax | Sparsemax | Last batch loss, softmax | Last batch loss, sparsemax |
| --- | --- | ---: | --- | --- | ---: | ---: |
| easy | depth | 0 | passes at 200 | passes at 400 | 0.019239 | 0.002907 |
| easy | depth | 1 | passes at 200 | passes at 200 | 0.026029 | 0.032107 |
| easy | depth | 2 | passes at 200 | passes at 200 | 0.025272 | 0.021076 |
| easy | recall | 0 | passes at 1,650 | passes at 550 | 0.010832 | 0.305198 |
| easy | recall | 1 | passes at 2,450 | passes at 350 | 0.011766 | 0.843603 |
| easy | recall | 2 | passes at 1,800 | fails, 41.21% | 0.008468 | 0.183331 |
| easy | parity | 0 | passes at 8,200 | fails, 87.11% | 0.009457 | 0.064472 |
| easy | parity | 1 | passes at 5,000 | fails, 96.68% | 0.076804 | 0.015308 |
| easy | parity | 2 | passes at 4,200 | passes at 14,200 | 0.065896 | 0.000511 |
| hard | depth | 0 | fails, 44.34% | fails, 32.62% | 0.000196 | 0.000460 |
| hard | depth | 1 | fails, 34.96% | fails, 44.34% | 0.024446 | 0.004452 |
| hard | depth | 2 | fails, 31.05% | fails, 33.59% | 0.000307 | 0.000205 |
| hard | recall | 0 | fails, 0.00% | fails, 9.57% | 3.361794 | 1.167205 |
| hard | recall | 1 | fails, 0.00% | fails, 0.00% | 2.503449 | 2.310641 |
| hard | recall | 2 | fails, 0.00% | fails, 0.00% | 2.549058 | 3.820318 |

Sparsemax learns the easy depth from every seed, and recall much earlier
than softmax from seeds 0 and 1; from seed 2 it fails recall within 4,800
updates. Parity takes longer and fails from two seeds at its recipe.
Neither normalizer passes the small model's hard depth or hard recall.

H5 removes all three small-model failures whose softmax control passes:
rate 1e-4 passes both parity seeds, and 3e-4 passes recall seed 2. The
larger parity neighbor 1e-3 passes seed 1, but not seed 0; the larger recall
neighbor 3e-3 falls short at 98.83%.

| Parity seed | Recipe, 3e-4 | Neighbor, 1e-4 | Neighbor, 1e-3 |
| ---: | --- | --- | --- |
| 0 | fails, 87.11% | passes at 13,400 | fails, 95.31% |
| 1 | fails, 96.68% | passes at 9,800 | passes at 16,600 |

| Recall seed | Recipe, 1e-3 | Neighbor, 3e-4 | Neighbor, 3e-3 |
| ---: | --- | --- | --- |
| 2 | fails, 41.21% | passes at 3,850 | fails, 98.83% |

The full-precision numbers, final head scales and hashes of the source
records are in [small_results.json](small_results.json).

### Large model

All 30 original large-model runs have finished. This table compares the
current arms; their shared pass threshold is selection sequence accuracy
0.99, including length 128 for hard depth. Losses again describe the final
sampled training batch.

| Mode | Task | Seed | Softmax | Sparsemax | Last batch loss, softmax | Last batch loss, sparsemax |
| --- | --- | ---: | --- | --- | ---: | ---: |
| easy | depth | 0 | passes at 200 | passes at 600 | 0.005176 | 0.000455 |
| easy | depth | 1 | passes at 200 | passes at 200 | 0.004812 | 0.005366 |
| easy | depth | 2 | passes at 200 | passes at 200 | 0.004596 | 0.005002 |
| easy | recall | 0 | fails, 97.27% | fails, 94.92% | 0.010500 | 0.036267 |
| easy | recall | 1 | fails, 96.68% | passes at 2,200 | 0.017379 | 0.004321 |
| easy | recall | 2 | passes at 1,500 | fails, 80.27% | 0.013770 | 0.100726 |
| easy | parity | 0 | passes at 6,600 | passes at 13,600 | 0.032085 | 0.014851 |
| easy | parity | 1 | passes at 7,600 | passes at 16,800 | 0.004917 | 0.028669 |
| easy | parity | 2 | passes at 7,000 | passes at 16,800 | 0.010212 | 0.000640 |
| hard | depth | 0 | passes at 3,600 | passes at 600 | 0.000014 | 0.000161 |
| hard | depth | 1 | passes at 4,400 | passes at 400 | 0.000005 | 0.000555 |
| hard | depth | 2 | passes at 1,800 | passes at 600 | 0.000150 | 0.000435 |
| hard | recall | 0 | passes at 3,600 | passes at 3,900 | 0.019992 | 0.041314 |
| hard | recall | 1 | passes at 3,200 | fails, 98.63% | 0.002921 | 0.027439 |
| hard | recall | 2 | passes at 2,650 | fails, 97.66% | 0.010402 | 0.007515 |

Sparsemax passes hard depth substantially earlier from every seed; its
large parity passes are later than softmax's. Easy recall seed 2 passes at
the lower neighbor 3e-4 (update 2,750) and the upper 3e-3 (update 2,800),
so H5 removes that failure too. Neither neighboring rate removes hard
recall's failures: both seeds have best sequence accuracy 0% at 1e-4 and
1e-3 over their 4,800-update budgets. Their higher-rate runs also leave
large training batch losses, 1.57 and 1.52, unlike the low final batch
losses at the original 3e-4. This does not establish that every training
example was fitted at the original rate.

| Hard recall seed | Neighbor, 1e-4 | Recipe, 3e-4 | Neighbor, 1e-3 |
| ---: | --- | --- | --- |
| 1 | fails, 0.00% | fails, 98.63% | fails, 0.00% |
| 2 | fails, 0.00% | fails, 97.66% | fails, 0.00% |

The full-precision large results, final head scales, source hashes and
archived-control comparison are in [large_results.json](large_results.json).
Only the cell `(hard, large, recall)`, from seeds 1 and 2, remains a
mechanism target after H5. Step 2 measures that cell from all three seeds
under both weights, including its passing seed 0. There is no small-model
target for step 2's small-model factorial ablation. Sparsemax's depth
failure prediction is not borne out here; parity is slower but its small
failures are removed by the recipe. These outcomes do not identify the
attention mechanism behind the remaining recall failures.

### Attention measurement

The observer is implemented and the focused checks pass: hand-made rows
check every statistic, masking and turnover; a rewritten table checks
latest-write routing; CPU and CUDA checks preserve live parameters,
buffers, gradients, optimizer, sampler, RNG states and module modes.
Temporary hooks survive failure cleanup and do not invalidate an existing
compiled graph. Short one-thread compiled softmax and sparsemax runs have
identical losses, training diagnostics, results and non-observer checkpoint
state with and without the probe; interrupted runs reproduce their entire
attention stream and checkpointed supports exactly. Step 1's qualification
of multi-threaded trajectory repetition still applies.
All 155 lab tests pass after the observer is added. The six QKNorm/ScaledDot
initialization pairs also share every non-score parameter exactly, for
both sizes and all three model seeds; score parameters alone differ.

All 24 initialization runs have finished. Their first-layer sparsemax
statistics at query positions are below, averaged over the four heads and
model seeds 0 to 2. Contexts have variable lengths 48 to 64, so these are
query-prefix measurements, not measurements of exactly 64 independent
scores. QKNorm and ScaledDot start from identical non-score parameters.

| Model | Scores | Mean support, positions | Mean support share | Mean score standard deviation | Singleton query share |
| --- | --- | ---: | ---: | ---: | ---: |
| small | QKNorm | 3.4334 | 7.76% | 0.9772 | 2.66% |
| small | ScaledDot | 35.4657 | 79.25% | 0.0241 | 0.00% |
| large | QKNorm | 3.3356 | 7.54% | 0.9703 | 4.13% |
| large | ScaledDot | 24.8788 | 55.87% | 0.0480 | 0.00% |

This supports H2's narrow-start prediction; it does not yet establish that
the starting scale caused a failure or that changing it repairs training.
Initial self-only query shares are 0.0122% (small) and 0.0407% (large)
under QKNorm, and zero under ScaledDot. H4 needs the failing trajectories,
not merely these initialization counts. Full initial and unchanged
zero-rate-update statistics, descriptions and hashes of the source
observations are in [attention_initial.json](attention_initial.json).

All six target repeats finished on 2026-10-04 UTC. Their descriptions differ
from step 1 only in diagnostics, and all six have identical update-zero
metrics to those originals. Their multi-threaded trajectories diverge at
the first or second later observation. Softmax again passes all three
seeds, but all three sparsemax repeats fail with best selection sequence
accuracy 0%, unlike step 1's pass and near-passes. Their last training batch
losses are 1.78--2.09, so these measured runs also fail to fit well.
This is a comparison of the current arms; it does not reproduce step 1's
particular fitted checkpoints or identify why separate threaded runs drift.
An additional direct check at the actual large recall shape, four threads
and 256 fixed examples preserves parameters, buffers, gradients, optimizer,
sampler, inputs, RNG and module flags exactly around all observations,
under both weights. The checks and record hashes are retained with the data.

| Seed | Softmax | Sparsemax | Last batch loss, softmax | Last batch loss, sparsemax |
| ---: | --- | --- | ---: | ---: |
| 0 | passes at 3,400 | fails, 0.00% | 0.002643 | 1.781534 |
| 1 | passes at 2,450 | fails, 0.00% | 0.018210 | 2.086703 |
| 2 | passes at 2,700 | fails, 0.00% | 0.004171 | 2.050959 |

The final sparsemax query statistics below average the 24 layer/head pairs
at the 2,048 fixed queries. Turnover compares updates 4,750 and 4,800;
answer support means any head in any layer, and its denominator is only
the wrong teacher-forced queries, not every query.

| Seed | Mean support, positions | Visible pairs exactly zero | Query turnover | Changed query memberships | Wrong queries with answer in any support | Self-only query share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.7887 | 91.62% | 1.96% | 43,576 | 1,455/1,609 (90.43%) | 0.0020% |
| 1 | 3.6178 | 92.00% | 2.18% | 48,415 | 1,811/1,863 (97.21%) | 0.0081% |
| 2 | 3.6853 | 91.85% | 2.30% | 51,085 | 1,794/1,853 (96.82%) | 0.1119% |

H1 is **refuted by its stated criterion in these measured failures**:
query supports change at every observation through the budget, and most
wrong queries have their answer position in a support. Membership is not
evidence that the head uses the answer successfully. The existing Lean
singleton-row theorem still holds; it does not imply locked training.
H2's narrow-start prediction is **supported** by the initialization pairs;
its causal and repair predictions remain open until the score-map or
starting-scale interventions. H3 is **refuted as a persistent basis failure
after H5**: sparsemax passes depth and parity wherever the matched softmax
does, with neighboring rates removing the small parity failures. Large
parity is slower, so this does not assert equal convergence speed.
H4 is **refuted by the rarity of self-only routes** in these measured
failures. After BOS is excluded, their share averaged over layer/head pairs
never exceeds 1.87%; the query share never exceeds 1.24%. Final individual
heads reach at most 6.10% of non-BOS rows and 2.05% of queries. All measured
self-only values exceed or equal XSA's epsilon. No small-model cell survives
H5, so the prescribed small factorial and its no-XSA intervention are empty;
these results make no claim about an unrun large no-XSA ablation.

Descriptions, all reduced observations, initial and final per-head actual
and shadow statistics, final scales and source hashes are in
[attention_targets.json](attention_targets.json). Raw per-query weights and
routes remain in each run's `attention.jsonl`. These uncompiled,
teacher-forced measurements describe the observed trajectories, not a
real-arithmetic theorem or a certificate of compiled routing. Step 3 still
tests score sensitivity independently by swapping the two backward maps.

### Forward and backward intervention

`SurrogateWeights(forward, backward)` declares both normalizers in the DSL.
An identical pair dispatches to the ordinary block, including fused softmax.
A mixed pair computes both maps at the same scores: its probabilities and
value gradients follow `forward`, and its score gradients follow `backward`.
The implementation adds the other map minus its detached copy (exactly zero)
to detached forward weights. It therefore preserves forward probabilities
exactly while autograd computes the declared surrogate. This mixed rule is
not the derivative of its forward loss. Mixed pairs use unfused softmax;
backward probabilities are cast to the forward dtype before the difference.

The six labels retain the target recipe, seeds, QKNorm, XSA and fixed-row
observer. Step 2 supplies the ordinary diagonal cases. New tests check exact
diagonal logits and every parameter gradient, mixed forward equality,
independent `torch.func.vjp` results including inactive and future scores,
and full-graph compiled logits and gradients. The sparsemax VJP reference
differentiates regular tensor operations for the paper's Algorithm 1 rather
than invoking the production custom backward. All six focused tests pass
(54.184 seconds), as do all 161 lab tests (933.852 seconds). The 156 previous
descriptions remain identical, and the six new targets differ from their
diagnostic controls only in the weights block. All six mixed trainings
finished their 4,800-update budgets on 2026-10-05 UTC. Each repeats the exact
update-zero validation metrics and fixed-row fingerprint of its forward
normalizer's diagonal control. The table compares them with step 2's
measured diagonals; its sparsemax controls are the failed repeats, not
step 1's fitted near-passes.

| Forward | Backward | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- | --- |
| softmax | softmax | passes at 3,400 | passes at 2,450 | passes at 2,700 |
| sparsemax | sparsemax | fails, 0.00% | fails, 0.00% | fails, 0.00% |
| sparsemax | softmax | fails, 0.00% | fails, 0.00% | fails, 0.00% |
| softmax | sparsemax | fails, 96.88% | fails, 96.29% | fails, 98.63% |

The last sampled training batch losses are 1.7742, 1.8029 and 1.8043 for
sparse forward with softmax backward, versus 0.0403, 0.0524 and 0.0086 for
the inverse mixture. These are batch measurements, not exhaustive training
accuracies. Sparse forward retains about 3.42 query positions per layer/head
on average, with 92.44--92.46% of visible query pairs exactly zero. Its final
query turnover is still 2.03--2.18%. The inverse mixture has dense forward
support throughout these final queries.

Softmax score gradients do not rescue sparse routing at this recipe, so
the backward-rescue criterion is not met and the conditional inactive-score
focus is not run. The forward remains binding in the tested intervention;
the inverse mixture's failures also leave an interaction or an unsuitable
surrogate. These outcomes do not attribute the failures specifically to
inactive zeros, counting ties or XSA self-erasure, and do not establish a
gradient-only repair. H2's starting-scale prediction is still open causally:
step 4 will test QKNorm starting at scale one and ScaledDot with the ordinary
sparsemax backward.

Full descriptions, reduced trajectories, final per-layer/head statistics,
initial-forward comparisons, training losses and source hashes are in
[surrogate_results.json](surrogate_results.json). Raw attention observations
and all per-query routes remain in the run directories. The conclusions
retain step 2's distinction between eager measurements, compiled selection
metrics and separate multi-threaded trajectories.

### Starting-scale and score-map repairs

H2's initialization prediction motivates two candidates. `qknorm-one`
sets `QKNorm(initial_scale=1.0)`; its per-head `log_alpha` starts at zero and
continues to learn. `scaleddot` replaces QKNorm by `ScaledDot()`. Both retain
the ordinary sparsemax backward, XSA, every task recipe, split and seed.
They measure fixed attention at every observation so a pass can be checked
for actual sparsity, per layer and head. Neither the failed surrogate nor
the refuted counting/self-erasure explanations justify an additional repair
candidate in this screen.

With `initial_scale=None`, the constructor keeps the original expression
`0.5 * log(head width)` and its float32 rounding. The new optional field is
compatible with descriptions saved before it existed: an absent field and
its default null compare equally for continuation. An explicit new start
is a different experiment and is rejected for continuation onto old state.
The five focused tests pass (0.691 seconds), including exact non-score
parameter equality for both model sizes and all three seeds, normalized-dot
scores and query/key gradients, and the learned scale's gradient.
All 166 lab tests pass (922.757 seconds). All 30 small-model repair runs
finished before the 30 large-model ones started. The small screen is
complete; the large screen is in progress, so no best-candidate or
generalization claim is made yet.

Each candidate matches 6 of the 9 small-model softmax passes. QKNorm-one
misses parity seeds 0 and 2 and easy recall seed 2; ScaledDot passes every
parity seed but misses every easy recall seed. Neither passes a hard run.
Every passing run retains exact zeros among visible pairs after BOS, in
all eight heads. These are genuine sparse passes, but the failures already
exclude both candidates from the complete basis repair criterion at the
unchanged recipes. The large hard-recall screen still tests whether either
intervention helps the original persistent failure.

The table gives the pass update or best selection sequence accuracy of a
failure. Hard depth selects the length-128 validation split. Last batch
losses are sampled before the final update. Full initial and final
statistics for each layer/head, every reduced attention observation, the
collector and helper sources, description checks and five raw-file hashes
per run are in [small_repair_results.json](small_repair_results.json).
All 30 small runs used identical lab source hashes at `aadc772` and changed
only the declared score block and `AttentionDiagnostics` relative to their
original sparsemax runs. Final sparsity below describes the last trained
weights, including for failures; their best accuracies may occur earlier.
The screen's record and description checks pass, as do all 166 lab tests
(1,206.052 seconds).

| Mode | Task | Seed | Softmax | QKNorm-one | ScaledDot | Last batch loss, QKNorm-one | Last batch loss, ScaledDot |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy | depth | 0 | 200 | 200 | 2000 | 0.020944 | 0.000625 |
| easy | depth | 1 | 200 | 200 | 1000 | 0.020880 | 0.000409 |
| easy | depth | 2 | 200 | 200 | 200 | 0.022573 | 0.037704 |
| easy | recall | 0 | 1650 | 2600 | fail (67.97%) | 0.006364 | 0.158280 |
| easy | recall | 1 | 2450 | 3850 | fail (96.29%) | 0.005444 | 0.008972 |
| easy | recall | 2 | 1800 | fail (68.36%) | fail (91.21%) | 0.091421 | 0.033928 |
| easy | parity | 0 | 8200 | fail (91.60%) | 3000 | 0.082636 | 0.040531 |
| easy | parity | 1 | 5000 | 6600 | 10200 | 0.060305 | 0.089290 |
| easy | parity | 2 | 4200 | fail (91.99%) | 14800 | 0.041831 | 0.039936 |
| hard | depth | 0 | fail (44.34%) | fail (42.58%) | fail (54.10%) | 0.007547 | 0.001628 |
| hard | depth | 1 | fail (34.96%) | fail (57.42%) | fail (37.50%) | 0.000041 | 0.000145 |
| hard | depth | 2 | fail (31.05%) | fail (41.99%) | fail (43.55%) | 0.000092 | 0.001248 |
| hard | recall | 0 | fail (0.00%) | fail (0.00%) | fail (0.59%) | 3.220756 | 1.661782 |
| hard | recall | 1 | fail (0.00%) | fail (0.00%) | fail (19.53%) | 2.971327 | 1.193219 |
| hard | recall | 2 | fail (0.00%) | fail (0.00%) | fail (1.37%) | 2.396232 | 1.779001 |

Mean support positions and the share of exact zeros among visible
pairs after BOS, averaged over layer/head pairs; no future or padded
position contributes a zero. The JSON retains every head separately.

| Mode | Task | Seed | Support, QKNorm-one | Zeros, QKNorm-one | Support, ScaledDot | Zeros, ScaledDot |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| easy | depth | 0 | 7.336 | 71.41% | 3.568 | 86.09% |
| easy | depth | 1 | 6.703 | 73.87% | 4.877 | 80.99% |
| easy | depth | 2 | 7.198 | 71.94% | 8.074 | 68.53% |
| easy | recall | 0 | 5.157 | 80.29% | 3.218 | 87.70% |
| easy | recall | 1 | 3.464 | 86.76% | 4.081 | 84.40% |
| easy | recall | 2 | 4.137 | 84.19% | 5.173 | 80.23% |
| easy | parity | 0 | 4.085 | 46.19% | 5.307 | 30.09% |
| easy | parity | 1 | 3.908 | 48.52% | 5.871 | 22.66% |
| easy | parity | 2 | 3.821 | 49.66% | 5.713 | 24.73% |
| hard | depth | 0 | 7.687 | 88.17% | 6.031 | 90.72% |
| hard | depth | 1 | 9.194 | 85.86% | 5.170 | 92.05% |
| hard | depth | 2 | 9.616 | 85.21% | 5.586 | 91.41% |
| hard | recall | 0 | 7.571 | 74.31% | 5.708 | 80.63% |
| hard | recall | 1 | 7.146 | 75.75% | 7.274 | 75.32% |
| hard | recall | 2 | 6.069 | 79.41% | 6.088 | 79.34% |

Run a pair with:

```sh
./make.py check experiments/basis_sparsemax
./make.py run experiments/basis_sparsemax softmax-easy-small-depth-seed0 sparsemax-easy-small-depth-seed0
./make.py report experiments/basis_sparsemax softmax-easy-small-depth-seed0 sparsemax-easy-small-depth-seed0
```
