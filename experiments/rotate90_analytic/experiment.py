"""The published Rotate90 classifier with ReLU, SiLU and near-budget SwiGLU.

Protocol: grokking_abmod f49969393a7482490d45483d8a4fd3046b1adede,
web/own-channel/README.md, W128 Fast. See README.md for deliberate differences.
"""

from lab.dsl import *

base = Experiment(
    model=Rotate90(width=128, branch=DenseBranch(ReLU())),
    benchmark=LabDivision(prime=97, train_fraction=0.5),
    optimizer=AdamW(lr=0.01, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=10_000, batch=512),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=10, batch=512),
    execution=Eager(device="cuda"),
    diagnostics=Diagnostics(gradients=True),
    checkpoint=Checkpoint(every=1000),
)

experiments = grid(
    {"relu": base,
     "silu": swap(base, "model.branch", DenseBranch(SiLU())),
     "swiglu": swap(base, "model.branch", SwiGLU(hidden=43))},
    {"seeds": {f"s{seed}": Seeds(model=seed, data=seed, batches=seed) for seed in (11, 17, 23)}},
)
