"""Task145: matched full-transformer generalization, two fresh GPU hours maximum.

Initialization seeds vary; data and batch order remain paired. Every variant
uses centered guarded XSA, Q/K norms, Peri norms, exact RoPE and tied readout.
The reserved test is evaluated only after this cohort closes.
"""

from lab.dsl import *

base = Experiment(
    model=ResidualTransformer(width=128, depth=4, heads=4, context=33),
    benchmark=ResidualPairs(prime=97, operation="division"),
    optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=.1, warmup=0),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=100000, batch=256),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=1000, batch=512),
    checkpoint=Checkpoint(every=1000),
    diagnostics=Diagnostics(gradients=True),
    execution=MaxAutotune(device="cuda"),
)

# The primary comparison includes the raw Q/K control for the repair itself.
experiments = grid({"division": base}, {
    "seeds.model": {"s11": 11, "s17": 17},
    "benchmark.prime": {"p97": 97, "p193": 193},
    "model.mixing": {"add": "add", "qk": "qk", "conditioned": "conditioned_qk"},
})
# Paper mechanism and fixed branch scale: separate secondary P97 controls.
experiments.update(grid({"mechanism": swap(base, "budget.updates", 30000)}, {
    "seeds.model": {"s11": 11, "s17": 17},
    "model.mixing": {"scaled": "scaled", "attnres": "attnres"},
}))
