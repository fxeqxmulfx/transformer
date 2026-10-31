# Rotate90 learning transfer

Does an actual Schedule-Free update improve unseen powered-division equations,
and does its early effect predict later accuracy? Static rank-one certificates
do not distinguish the successful and failed models in the preceding studies.

| Task | Width | Initialization seeds | Shared settings |
|---|---|---|---|
| P97, K2 | 16, 64 | 31, 37, 41 | Depth1; data/order seed11 |
| P193, K3 | 16, 64 | 31, 37, 41 | Half table; 1500 updates |
| P17, K1 | 64 | 31, 37, 41 | Memorization control |

All 15 runs use Schedule-Free AdamW, lr .003, betas .9/.98, decay1, ten-update
warmup, batch512 with short epoch tails, and CUDA max-autotune compilation.
Evaluation uses the optimizer's averaged parameters. The split and batch order
are fixed across initialization seeds, unlike the earlier bundled-seed controls.
The initial state and every 50th checkpoint are retained for independent replay.

`hypotheses.json` freezes the update identity, next-step sign prediction, an early
loss extrapolation and a matched width comparison before training. Replays use
CPU float64 with the original float32 normalization epsilon. They differentiate
every parameter and use the actual next batch. Heldout gradients are diagnostic
only. They never enter training. A measured Taylor remainder is not a certified
curvature bound, and the extrapolation does not prove convergence.

Status: declared before training; results pending.
