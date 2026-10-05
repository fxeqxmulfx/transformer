# Sparsemax on the basis

Where does sparsemax attention fail where softmax passes the calibrated
[basis](../basis/README.md), and is a failure removed by a neighboring rate?
This is steps 0 to 2 of [EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md).
The arms change only the attention weights. Both keep QKNorm, XSA, the
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

Run a pair with:

```sh
./make.py check experiments/basis_sparsemax
./make.py run experiments/basis_sparsemax softmax-easy-small-depth-seed0 sparsemax-easy-small-depth-seed0
./make.py report experiments/basis_sparsemax softmax-easy-small-depth-seed0 sparsemax-easy-small-depth-seed0
```
