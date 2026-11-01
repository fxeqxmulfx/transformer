"""Frozen first mixed-arithmetic screen, user request 2026-11-01.

All cases use the normalized source RoPE transformer, final-answer CE,
Schedule-Free AdamW and CUDA max-autotune. Seeds 17 change initialization
alone. The two principal shapes approximately match parameter counts;
measured runtime is reported separately, not claimed to be matched.
"""

from lab.dsl import *

base = Experiment(
    model=SourceTransformer(width=256, depth=1, heads=4, context=33),
    benchmark=ModularPairs(prime=97),
    optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=.1, warmup=0),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=2000, batch=256),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=250, batch=512),
    checkpoint=Checkpoint(every=500),
    execution=MaxAutotune(device="cuda"),
)

shapes = {"w256d1": base.model, "w128d4": SourceTransformer(128, 4, 4, 33)}
primes = {f"p{prime}": prime for prime in (17, 97, 193)}
experiments = grid({"pairs": base}, {
    "benchmark.operation": {"mixed": "mixed", "division": "division"},
    "benchmark.prime": primes, "model": shapes,
})
experiments.update(grid({"confirm-mixed": swap(base, "seeds.model", 17)}, {
    "benchmark.prime": {"p97": 97, "p193": 193}, "model": shapes,
}))
experiments.update(grid({"full-mixed": swap(base, "benchmark.train_fraction", 1.)}, {
    "benchmark.prime": primes,
}))

# A smaller fixed sample permits the P17 one-step domain (17^3 = 4913).
chain = swap(base, "benchmark", ModularChain(train_examples=2048, heldout_examples=1024,
                                            longer_examples=1024))
experiments.update(grid({"chain": chain}, {
    "benchmark.prime": primes,
    "benchmark.steps": {f"l{length}": length for length in (1, 2, 4, 8)},
    "model": shapes,
}))

# Predeclared optimizer sensitivity controls, independent of screen outcomes.
recipes = {"mixed-wd0": swap(base, "optimizer.weight_decay", 0.),
           "mixed-lr001": swap(base, "optimizer.lr", .001)}
experiments.update(grid(recipes, {"seeds.model": {"s11": 11, "s17": 17}}))
experiments["timing"] = swap(swap(base, "budget.updates", 100), "evaluate.every", 100)
