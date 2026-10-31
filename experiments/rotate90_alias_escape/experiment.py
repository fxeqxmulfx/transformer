"""Prospective Schedule-Free interventions on the six narrow-model checkpoints.

Source: review/flat_spectrum.md, Alias symmetry and training. Every restart
resets optimizer histories and batch order; controls share their initial query
weights, task split, rates and new update budget. Original runs are untouched.
"""

from lab.dsl import *

sources = [
    (97, 2, 3, 11, "sf-w16d1-p97-k2-lr003", "13fb693f29e2988ce289c6a694d1a3893d3648e719e72062eef6a89c828f9efa"),
    (97, 2, 3, 17, "sf-confirm-w16d1-p97-k2-s17", "eef4398bd09fb2dcb70798e70ef429fd91cc502e23d80c24d0ab700aeb8a5c5e"),
    (97, 2, 3, 23, "sf-confirm-w16d1-p97-k2-s23", "3a3de4f95b25e490d8b4f111c56e589a54e8cd5a7c63c1fa13274949b850e6b8"),
    (193, 3, 8, 11, "sf-w16d1-p193-k3-lr003", "9827d06cdb1c326f1b42ce92037290de907a11927eccdc657e94264595938356"),
    (193, 3, 8, 17, "sf-confirm-w16d1-p193-k3-s17", "8a63cfaefd327109871b48cd7508867c90da8d59c1e6104359e9c47ad8eb5cf3"),
    (193, 3, 8, 23, "sf-confirm-w16d1-p193-k3-s23", "8b79b28210fc349454ae6e41a04996f712b86fb3e91eeb9c3a38afb843fe7f48"),
]

experiments = {}
for prime, exponent, alias, seed, source, digest in sources:
    base = Experiment(
        model=PeriRotate90Restart(PeriRotate90(16, 1),
            f"experiments/rotate90_power_learning/runs/{source}/checkpoint.pt", digest),
        benchmark=PowerDivision(prime, exponent, .5),
        optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=1., warmup=0),
        schedule=Schedule(warmup=10, inclusive=True),
        budget=Budget(updates=5000, batch=512),
        seeds=Seeds(seed, seed, seed),
        evaluate=Evaluate(every=250, batch=1024),
        checkpoint=Checkpoint(every=1000),
        execution=MaxAutotune(device="cuda"),
    )
    variants = {"restart": base,
                "wd01": swap(base, "optimizer.weight_decay", .1),
                "wd0": swap(base, "optimizer.weight_decay", 0.),
                "frozen-e": swap(base, "model.train_embedding", False),
                "frozen-branch": swap(base, "model.train_branch", False),
                "noise003": swap(base, "model.noise", .03),
                "noise03": swap(base, "model.noise", .3)}
    experiments.update({f"{name}-p{prime}k{exponent}-s{seed}": value for name, value in variants.items()})
    if seed == 11:
        full = swap(swap(base, "benchmark.train_fraction", 1.), "budget", Budget(200, prime * (prime - 1)))
        full = swap(swap(full, "evaluate.every", 50), "checkpoint.every", 200)
        projected = swap(full, "model.alias_size", alias)
        symmetry = {"full-original": full, "full-alias": projected,
                    "full-alias-noise": swap(projected, "model.noise", .03),
                    "mini-alias": swap(projected, "budget.batch", 512)}
        experiments.update({f"{name}-p{prime}k{exponent}": value for name, value in symmetry.items()})
