"""SwiGLU step controls with nearly equal cumulative lr * weight_decay.

Source: rotate90_analytic at c59d4426, outcomes at 73ae6c26, and
landscape_grokking_plan.md, M9.5. The source browser's other Fast settings
are retained. Extended budgets distinguish smaller steps from shorter time.
"""

from lab.dsl import *

base = Experiment(
    model=Rotate90(width=128, branch=SwiGLU(hidden=43)),
    benchmark=LabDivision(prime=97, train_fraction=0.5),
    optimizer=AdamW(lr=0.01, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=10_000, batch=512),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=10, batch=512),
    execution=Eager(device="cuda"),
    diagnostics=Diagnostics(every=10, gradients=True),
    checkpoint=Checkpoint(every=1000),
)

experiments = grid(
    {"lr01": base,
     "lr003": swap(swap(base, "optimizer.lr", 0.003), "budget.updates", 33_330),
     "lr001": swap(swap(base, "optimizer.lr", 0.001), "budget.updates", 99_960)},
    {"seeds": {f"s{seed}": Seeds(model=seed, data=seed, batches=seed) for seed in (11, 17, 23)}},
)
