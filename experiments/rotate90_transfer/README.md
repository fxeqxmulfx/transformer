# Rotate90 learning transfer

Does an actual Schedule-Free update improve unseen powered-division equations,
and does its early effect predict later accuracy? Static rank-one certificates
do not distinguish the successful and failed models in the preceding studies.

| Task | Width | Initialization seeds | Shared settings |
|---|---|---|---|
| P97, K2 | 16, 64 | 31, 37, 41 | Depth 1; data/order seed 11 |
| P193, K3 | 16, 64 | 31, 37, 41 | Half table; 1500 updates |
| P17, K1 | 64 | 31, 37, 41 | Memorization control |

All 15 runs use Schedule-Free AdamW, lr .003, betas .9/.98, decay 1, ten-update
warmup, batch 512 with short epoch tails, and CUDA max-autotune compilation.
Evaluation uses the optimizer's averaged parameters. The split and batch order
are fixed across initialization seeds, unlike the earlier bundled-seed controls.
The initial state and every 50th checkpoint are retained for independent replay.

`hypotheses.json` freezes the update identity, next-step sign prediction, an early
loss extrapolation and a matched width comparison before training. Replays use
CPU float64 with the original float32 normalization epsilon. They differentiate
every parameter and use the actual next batch. Heldout gradients are diagnostic
only. They never enter training. A measured Taylor remainder is not a certified
curvature bound, and the extrapolation does not prove convergence.

Completed on 2026-10-31, using setup commit `d47112d5`. All 15 declared runs
finished all 1,500 updates; all 465 numbered states are retained locally in
`runs/`. No thresholds or optimizer settings were changed after training.

## Findings

All six width-64 P97/P193 runs exceed 99.8% heldout accuracy by the final update;
they first cross 99% at updates 600--650. The six width-16 runs remain at
1.96--17.93% within this shorter budget. All three P17 controls fit the training
split perfectly while heldout accuracy remains 7.35--9.56%.

| Task | Width | Model seed | Train % | Heldout % | Step-400 forecast of >=99% |
|---|---:|---:|---:|---:|---|

| P97, K2 | 16 | 31 | 18.26 | 9.7294 | No |
| P97, K2 | 16 | 37 | 25.64 | 17.9338 | No |
| P97, K2 | 16 | 41 | 20.38 | 15.9364 | No |
| P97, K2 | 64 | 31 | 100.00 | 99.8711 | No |
| P97, K2 | 64 | 37 | 100.00 | 99.9570 | No |
| P97, K2 | 64 | 41 | 100.00 | 99.8711 | Yes |
| P193, K3 | 16 | 31 | 5.07 | 2.8282 | No |
| P193, K3 | 16 | 37 | 5.70 | 2.9955 | No |
| P193, K3 | 16 | 41 | 4.32 | 1.9646 | No |
| P193, K3 | 64 | 31 | 100.00 | 99.9676 | No |
| P193, K3 | 64 | 37 | 100.00 | 99.9352 | No |
| P193, K3 | 64 | 41 | 100.00 | 99.8813 | No |
| P17, K1 | 64 | 31 | 100.00 | 9.5588 | No |
| P17, K1 | 64 | 37 | 100.00 | 7.3529 | No |
| P17, K1 | 64 | 41 | 100.00 | 9.5588 | No |

H1 passes: the independently derived evaluation displacement matches the source
optimizer at all 120 probes, with maximum absolute coordinate residual
2.042e-16. H2 also passes: its directional derivative predicts all 120 next-step
loss signs, including 98 improvements and 22 deteriorations. This is local
finite-step evidence, not a bound on every future segment's curvature.

H3 makes one successful >=99% prediction and misses five successful runs; there
are no false positives. Two different limitations matter. The mean-loss
threshold 0.01*log(2) is sufficient, not necessary, for 99% accuracy: **all six
successful runs finish above that loss threshold**. For example, P97 seed31's
forecast loss 0.009553 nearly matches its actual 0.009550, but still does not
certify its 99.8711% accuracy. The forecast also misses real changes in progress:
P193 seed31 predicts loss 1.977 but reaches 0.01189.

H4 fails: only three of six matched width comparisons favor width 64 in the
prespecified early relative-progress statistic (the threshold was four). All
three P193 comparisons favor width 16 early, despite its much worse final result.
For P193 width 64 seed31, relative first-order progress is about 0.000275 at
update 200, 0.00440 at 400 and 0.00990 at 750. The larger model's advantage emerges
later. A fixed early rate or early width ranking cannot explain these runs.

The Lean extension therefore keeps the complete time-varying transfer and
curvature remainder in its finite-time bound. A geometric rate requires a
condition at **every** intervening step. These experiments do not prove that
random initialization produces that condition. Explaining the later
acceleration and persistent generalization remains the next dynamics gap.

## Artifacts and replay

`results.json` contains every run and all 120 diagnostic probes, including
checkpoint/batch hashes, source hashes, optimizer terms, remainders and the
frozen forecast outcomes. `curves.csv` retains every recorded train/heldout
point. Reproduce the analysis from `python/` with:

```
uv run --locked python ../experiments/rotate90_transfer/analyze.py
```

All matched runs have identical split and next-batch fingerprints at every probe.
Before-update CPU float64 replays agree with recorded compiled CUDA float32
accuracy exactly; their maximum mean-loss difference is 4.040e-7. The CPU copies
preserve the original RMS epsilon 2^-23. Replay after update 1500 is a one-step
diagnostic continuation, not an extra trained endpoint in the reported results.

Runtime: Python 3.15.2, PyTorch 2.15.0+cu132, CUDA 13.2, RTX 3050 Laptop (4 GiB).
The requested compilation mode is max-autotune, fullgraph=True, dynamic=False;
Inductor reports too few SMs for its GEMM autotuning path on this GPU. The eager
source Schedule-Free optimizer is unchanged. Recorded training time totals
30.28 seconds, excluding compilation, evaluation, diagnostics and checkpoints;
it is not total wall time and is not a cross-width speed comparison.

Validation: all definitions pass `./run.py check`; the initial-state retention,
exact resume, independent update replay and all-parameter finite-difference
tests pass as part of 410 lab tests plus 104 optimizer tests. The accompanying
17 Lean theorems pass the full build and axiom audit, retaining the 36 existing
unfinished statements. No global curvature certificate or random-start success
probability is claimed from these fifteen runs.

## Optimizer memory follow-up

The next cycle tests whether the averaging term's local benefit survives when
new base updates stop. The seed31 exploration in `memory_exploration.json`
compares query, base and evaluation weights before declaring the interventions.
At P193 update 750 the evaluation model scores 99.94% but its base only 3.92%.
Thus the local averaging term is not simply approaching an already solved base.

`memory_hypotheses.json` pins all 15 source runs at steps 200/400/750. It freezes
the zero-rate horizons, 32-step actual-source identity check, transient-benefit
prediction and whole-parameter negation control. This reuses existing runs;
`memory.py` introduces no new fitting recipe. Path evaluation uses compiled CUDA
max-autotune. The exact zero-rate formula permits long-horizon evaluation without
simulating 50000 gradient steps. CPU float64 source steps independently verify
that the retained optimizer history, rather than newly learned weights, drives
this intervention. The declaration was recorded before counterfactual evaluation.

The setup was frozen in `daab04d6`, before executing any of the declared
counterfactuals. All 45 checkpoint controls completed; `memory_results.json`
retains every role, all 450 path points, all source checks, checkpoint hashes,
script/declaration hashes and the runtime. The source training setup remains
`d47112d5`. Reproduce from `python/`:

```
uv run --locked python ../experiments/rotate90_transfer/memory.py
```

All four declared hypotheses pass. H1's maximum coordinate discrepancy after
32 actual source updates is 6.661e-16; every base is bitwise unchanged, every
raw second moment changes, and the maximum total-clock discrepancy is 1.041e-17.
All 45 evaluation vectors move (H2). Negating every parameter gives exactly
identical heldout logits at all 45 checkpoints, and the zero midpoint has no
strict correct winner (H4). Ordinary argmax can still select a tied correct
class; the strict-winner metric deliberately counts ties as failures.

H3 passes each of its three conditions for all six wide P97/P193 checkpoints
at update 750. The first zero-rate step lowers heldout cross-entropy, but the
declared long continuation has higher loss than the initial point and accuracy
below 50%. All six cases are shown below; accuracies are percentages.

| Task / seed | Initial accuracy | Base accuracy | After 50000 averaging updates | Initial / final CE |
| --- | ---: | ---: | ---: | ---: |
| P97 K2 / 31 | 99.850 | 5.799 | 6.035 | 0.03916 / 6.02115 |
| P97 K2 / 37 | 99.764 | 4.639 | 4.790 | 0.06119 / 6.56472 |
| P97 K2 / 41 | 99.850 | 4.940 | 5.026 | 0.05304 / 6.51308 |
| P193 K3 / 31 | 99.935 | 3.924 | 4.059 | 0.14724 / 8.04662 |
| P193 K3 / 37 | 99.978 | 6.941 | 7.281 | 0.08361 / 6.75362 |
| P193 K3 / 41 | 99.984 | 6.018 | 6.347 | 0.06159 / 6.93682 |

The 50000-update points are direct evaluations of the proved closed form, not
50000 simulated gradients. The stored clock W/M² is 743.85 at these checkpoints.
Their lowest losses among the declared horizons occur at 100 or 200 additional
averaging updates; no horizon was tuned or added after inspecting outcomes.
At update 400, the P193 seed31 continuation improves from 28.15% to 48.47% at
the declared 100-update point without moving the base, then deteriorates.
The corresponding original positive-rate run later reaches above 99%.

The narrow and P17 controls remain in the report. Across every declared source
time and horizon, P97 width 16 reaches at most 15.38%, P193 width 16 at most
2.07%, and P17 at most 13.24%. The intervention does not repair those failures.
Stopping the rate removes both gradient and decay movement of the base, so
this test does not isolate those two contributions from each other.

The theory correction is now explicit: static spectral rank and local favorable
averaging are insufficient to predict continued learning. Actual optimizer
history determines a moving evaluation point, and the neural parameter path
can leave the region of correct predictions. The 27 new Lean theorems prove
the exact clock, its loss/classification limit and a nonconvex whole-table
solver region with genuine perfect powered-division endpoints. They do not
explain why continued positive-rate training creates a favorable path from
typical random starts. The original transfer study's failed early forecast
and width-ranking hypothesis remain unchanged.

Validation: full Lean build and axiom/source checks pass; no new unfinished
proof or extra axiom was introduced. All 412 lab tests (13 skipped) and 104
optimizer tests (14 skipped) pass. Runtime remains Python 3.15.2, Torch
2.15.0+cu132, CUDA 13.2, compiled max-autotune/fullgraph/static shapes.
