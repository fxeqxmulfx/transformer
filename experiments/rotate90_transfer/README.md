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
