"""Current Rotate90 learning of powered division, requested 2026-10-31.

All runs use Python 3.15, PyTorch 2.15, max-autotune compilation, the source
initializer, a fixed half split, shuffled epochs and complete-split reports.
Model and optimizer screens vary one factor at a time from the base.
"""

from lab.dsl import *

base = Experiment(
    model=PeriRotate90(width=64, depth=1),
    benchmark=PowerDivision(prime=97, exponent=1, train_fraction=.5),
    optimizer=AdamW(lr=.003, betas=(.9, .98), weight_decay=1., eps=1e-8),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=5000, batch=512),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=250, batch=1024),
    checkpoint=Checkpoint(every=1000),
    execution=MaxAutotune(device="cuda"),
)

tasks = {"benchmark.prime": {f"p{p}": p for p in (17, 97, 193)},
         "benchmark.exponent": {f"k{k}": k for k in (1, 2, 3)}}
architectures = {"w64d1": base,
                 "w16d1": swap(base, "model.width", 16),
                 "w128d1": swap(base, "model.width", 128),
                 "w64d2": swap(base, "model.depth", 2),
                 "w64d4": swap(base, "model.depth", 4)}

experiments = grid(architectures, tasks)

# Optimizer screen on a common width/depth across all three primes and powers.
recipes = {"lr001": swap(base, "optimizer.lr", .001),
           "lr01": swap(base, "optimizer.lr", .01),
           "wd0": swap(base, "optimizer.weight_decay", 0.),
           "wd01": swap(base, "optimizer.weight_decay", .1),
           "beta999": swap(base, "optimizer.betas", (.9, .999)),
           "beta05": swap(base, "optimizer.betas", (.5, .98))}
experiments.update(grid(recipes, tasks))

# Full-table controls isolate fitting at K=3 from generalization.
full = swap(swap(base, "benchmark.train_fraction", 1.), "benchmark.exponent", 3)
experiments.update(grid({"full": full}, {"benchmark.prime": {f"p{p}": p for p in (17, 97, 193)}}))

# A short run warms the exact compiler path before the numerical screen.
experiments["timing"] = swap(swap(base, "budget.updates", 100), "evaluate.every", 100)

# Follow up the narrow model's K=1 underfitting where a Lean witness exists.
narrow = swap(swap(base, "model.width", 16), "benchmark.prime", 193)
experiments.update(grid({"narrow": narrow}, {
    "optimizer.lr": {"lr001": .001, "lr003": .003, "lr01": .01},
    "optimizer.weight_decay": {"wd0": 0., "wd01": .1, "wd1": 1.}}))

# K=5 is coprime to P-1 at every chosen prime, preserving all output classes.
experiments.update(grid({"power5": swap(base, "benchmark.exponent", 5)}, {
    "benchmark.prime": {f"p{p}": p for p in (17, 97, 193)}}))

# Theory follow-up: cases chosen before inspecting seeds 17 and 23.
cases = {name: experiments[name] for name in (
    "w16d1-p97-k2", "w64d4-p193-k2", "w16d1-p193-k3",
    "w64d1-p97-k2", "w64d1-p193-k3", "w64d1-p17-k1")}
experiments.update(grid({f"confirm-{name}": case for name, case in cases.items()}, {
    "seeds": {f"s{seed}": Seeds(seed, seed, seed) for seed in (17, 23)}}))

# Schedule-Free is a targeted control for the three observed aliasing failures.
# Keep the external 10-step warmup; built-in warmup is zero, with no decay schedule.
free = {f"sf-{name}": swap(case, "optimizer", ScheduleFreeAdamW(
    lr=.003, betas=(.9, .98), weight_decay=1., warmup=0))
        for name, case in cases.items() if name in (
            "w16d1-p97-k2", "w64d4-p193-k2", "w16d1-p193-k3")}
experiments.update(grid(free, {"optimizer.lr": {"lr001": .001, "lr003": .003, "lr01": .01}}))

# User-requested larger-model stability check, now exclusively Schedule-Free AdamW.
large_base = swap(base, "optimizer", ScheduleFreeAdamW(
    lr=.003, betas=(.9, .98), weight_decay=1., warmup=0))
large_shapes = {"w256d1": PeriRotate90(256, 1), "w512d1": PeriRotate90(512, 1),
                "w256d4": PeriRotate90(256, 4)}
large_tasks = {"p97k2": PowerDivision(97, 2), "p193k2": PowerDivision(193, 2),
               "p193k3": PowerDivision(193, 3)}
experiments.update(grid({"large": large_base}, {
    "model": large_shapes, "benchmark": large_tasks,
    "seeds": {f"s{seed}": Seeds(seed, seed, seed) for seed in (11, 17, 23)}}))
experiments.update(grid({"large-small-data": swap(swap(large_base, "model", PeriRotate90(512, 1)),
                                                "benchmark", PowerDivision(17, 1))}, {
    "seeds": {f"s{seed}": Seeds(seed, seed, seed) for seed in (11, 17, 23)}}))
# The seed-11 small controls already exist above at lr=.003. Add the other seeds.
experiments.update(grid({f"sf-confirm-{name}": swap(case, "optimizer", large_base.optimizer)
                        for name, case in cases.items() if name in (
                            "w16d1-p97-k2", "w64d4-p193-k2", "w16d1-p193-k3")}, {
    "seeds": {f"s{seed}": Seeds(seed, seed, seed) for seed in (17, 23)}}))
experiments.update(grid({"sf-small-data": swap(large_base, "benchmark", PowerDivision(17, 1))}, {
    "seeds": {f"s{seed}": Seeds(seed, seed, seed) for seed in (11, 17, 23)}}))
