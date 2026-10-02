# Complete optimizer pair and launch of the harder arithmetic task

Both frozen mod-97 budgets are complete: 300,000 updates, 1,202 canonical
observations, 2,400 exhaustive neighbor probes, 3,600 tensor diagnostics and
300,000 dense gradient records. Neither primary AdamW nor raw AMSGradW
passes the required memorization phase or final-tail persistence criterion.
The independent-confirmation gate remains closed.

| Outcome | Primary AdamW | Raw AMSGradW control |
| --- | ---: | ---: |
| Final train / held-out accuracy | 100% / 100% | 100% / 99.9785% |
| First canonical held-out 99%, update | 1,250 | 5,250 |
| First long joint target interval | 1,250–6,000 | 52,750–57,500 |
| Observed long confirmation, training / wall seconds | 181.81 / 194.91 | 1,595.81 / 1,692.87 |
| Failing final-window observations / 201 | 7 | 10 |
| Worst final-window held-out accuracy | 43.9003% | 58.1400% |
| Required memorization plateau | Absent | Absent |
| Eligible persistent-target timing support | 0 | 0 |
| Complete training / wall seconds | 4,022.04 / 4,284.18 | 4,256.08 / 4,511.30 |
| Diagnostic seconds | 44.04 | 41.62 |
| Peak allocated CUDA bytes | 117,470,208 | 119,166,464 |

The [complete comparison](../../baselines/adamw_stability_optimizer_pair_lr001_20261002/REPORT.md)
includes byte-identical copies of both individual archives and every scheduled
observation. The [primary result](adamw-result.md),
[raw control result](raw-amsgradw-result.md), and
[pair validation](optimizer-pair-result-validation.json) retain failures,
event costs and empty eligible timing support. Verification passed with
system Python without importing PyTorch. Original plans/logs and local
checkpoint fingerprints agree; all 31 unique frozen source/manuscript
fingerprints and the launcher remain unchanged.

The original comparison PNG/PDF remains immutable. A reviewed standalone
[readable figure](optimizer-pair-curves/optimizer-pair.png) and
[PDF](optimizer-pair-curves/optimizer-pair.pdf) plot the same complete histories
with update labels in thousands. Provenance checks both 601-point input
histories, script SHA256 and artifact hashes. No smoothing, row filtering
or criterion change is applied; the loss display floor is 1e-12 nats.

Only `optimizer` differs between the configurations: identical task, model,
split, initialization, 150,000-update exposure and instrumentation. AdamW
uses betas (0.9, 0.98), bias correction and no maximum; raw AMSGradW uses
(0.9, 0.999), no bias correction and a maximum. This compares complete
optimizer choices, not individual update components. One initialization and
one split do not support mean/SD or a universal speedup. CPU archiving and
verification tests overlapped portions of the control's training; observed
wall and training costs cannot be attributed solely to the optimizer.

The user requested a harder task. The already prospectively frozen
[mod-193 adaptation](larger-modulus-protocol.md) started at
2026-10-02T15:50:59.797959+00:00 after both budgets and the complete verified
comparison. Trainer PID 119293 executes the unchanged launcher through the
verified CSV runtime adapter; serial worker PID 117203/session 46645 will
preserve and verify its complete archives. The optimizer-pair trainer and
archive worker finished with exit code 0; do not start another GPU trainer.

Mod-193 changes only the prime in the primary recipe. Its larger vocabulary,
436,104 parameters, 18,528-example splits, 96-example tails and different
exposure/initial weights are explicit derived differences, not an isolated
arithmetic-difficulty effect. Keep the full 150,000-update budget and all
prospective gates. Six new confirmations, the paired architecture change
and scientific complementary mechanics remain outstanding; the verified
CPU complementary probes establish implementation only.
