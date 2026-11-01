"""Task155: separate finite-epsilon RMS gradients from coordinate conditioning.

The two model blocks have identical forward values and finite known optima.
Only the local RMS VJP differs. Gates after RMS are an unaffected control.
"""

from lab.dsl import *

base = Experiment(
    model=BranchGates(), benchmark=GradientTargets(),
    optimizer=ScheduleFreeAdamW(lr=.01, betas=(.9, .98), weight_decay=0),
    schedule=Schedule(), budget=Budget(updates=10000, batch=64),
    seeds=Seeds(model=0, data=11, batches=11),
    evaluate=Evaluate(every=250, batch=256),
    checkpoint=Checkpoint(every=250), diagnostics=Diagnostics(gradients=True),
    execution=MaxAutotune(device="cpu", threads=1),
)

experiments = grid({"rms": base}, {
    "model": {"native": BranchGates(), "accurate": AccurateBranchGates()},
    "model.depth": {"d1": 1, "d8": 8},
    "model.scale": {"a0.001": .001, "a1": 1., "a1000": 1000.},
    "model.placement": {"before": "before", "after": "after"},
    "seeds": {"s11": Seeds(model=0, data=11, batches=11),
              "s17": Seeds(model=0, data=17, batches=17)},
})
