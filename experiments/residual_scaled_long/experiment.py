"""Task154: unchanged P97 scaled trajectories, extended from update 23,000.

The cohort runner stops at the user's first validation crossing of 99%.
The original benchmark stability settings remain in the description so that
require_continuation can verify the unchanged recipe, apart from the budget.
This first-crossing endpoint is not the original three-observation 99.9% result.
"""

from lab.dsl import *

base = Experiment(
    model=ResidualTransformer(width=128, depth=4, heads=4, context=33, mixing="scaled"),
    benchmark=ResidualPairs(prime=97, operation="division", target_accuracy=.999),
    optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=.1, warmup=0),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=100000, batch=256),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=1000, batch=512),
    checkpoint=Checkpoint(every=1000),
    diagnostics=Diagnostics(gradients=True),
    execution=ResidualMaxAutotune(device="cuda", attention_backend="math"),
)

experiments = grid({"scaled": base}, {"seeds.model": {"s11": 11, "s17": 17}})
