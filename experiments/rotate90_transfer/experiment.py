"""Prospective transfer study: vary initialization and width at fixed data/order.

Source: EXPERIMENT_PLAN.md, Rotate90 learning transfer. Training uses only
Schedule-Free AdamW, with compiled max-autotune forward/backward/evaluation.
CheckpointHistory retains independent replay inputs without changing updates.
"""

from lab.dsl import *

base = Experiment(
    model=PeriRotate90(16, 1), benchmark=PowerDivision(97, 2, .5),
    optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=1., warmup=0),
    schedule=Schedule(warmup=10, inclusive=True), budget=Budget(updates=1500, batch=512),
    seeds=Seeds(31, 11, 11), evaluate=Evaluate(every=50, batch=1024),
    checkpoint=CheckpointHistory(every=50), execution=MaxAutotune(device="cuda"),
)

experiments = {}
for prime, exponent, widths in [(97, 2, (16, 64)), (193, 3, (16, 64)), (17, 1, (64,))]:
    for width in widths:
        for seed in (31, 37, 41):
            variant = swap(base, "benchmark", PowerDivision(prime, exponent, .5))
            variant = swap(variant, "model.width", width)
            experiments[f"p{prime}k{exponent}-w{width}-s{seed}"] = swap(variant, "seeds.model", seed)
