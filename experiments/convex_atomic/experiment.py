from lab.dsl import *


ORDER = Experiment(
    model=AtomicMatching(context=2, width=1, heads=3, cap=1.0, channels=1,
                         initial_value=0.75, precision="float64"),
    benchmark=MatchingOrders(), optimizer=AtomicColumns(pricing=OrderPricing()),
    schedule=Schedule(), budget=Budget(updates=1_000, batch=2),
    seeds=Seeds(), evaluate=Evaluate(every=1, batch=2), execution=Eager(device="cpu"),
    diagnostics=Diagnostics(every=1), checkpoint=Checkpoint(every=1),
)


def raw_control(run):
    """Train the same initial actual head with ordinary raw-coordinate AdamW."""
    run = swap(run, "optimizer", AdamW(lr=0.01, betas=(0.9, 0.999), weight_decay=0.0))
    return swap(run, "model.heads", 1)


def columns_basis(run):
    """128 outer updates, 4x65 physical pricing points per update, no route labels."""
    run = swap(run, "execution", Eager(device="cpu", threads=4))
    run = swap(run, "schedule", Schedule())
    run = swap(run, "optimizer", AtomicColumns(pricing=SearchPricing(restarts=4, steps=64)))
    run = swap(run, "budget.updates", 128)
    run = swap(run, "evaluate", Evaluate(every=16, batch=32))
    run = swap(run, "diagnostics", Diagnostics(every=16))
    return swap(run, "checkpoint", Checkpoint(every=16))


SATURATED = swap(swap(ORDER, "model.initial_query", 1.0), "model.initial_key_spread", 1.0)


def search_order(run):
    """Use Basis's numerical pricing on the formal task, without its analytic oracle."""
    run = swap(run, "optimizer.pricing", SearchPricing(restarts=4, steps=64))
    run = swap(run, "model.heads", 129)
    run = swap(run, "budget.updates", 128)
    return swap(run, "evaluate.every", 16)


experiments = {
    **grid({"order-columns": ORDER, "order-adamw": raw_control(ORDER),
            "saturated-columns": SATURATED, "saturated-adamw": raw_control(SATURATED)},
           {"seeds.model": {f"seed{seed}": seed for seed in range(8)}}),
    "outside-columns": swap(ORDER, "benchmark.targets", (2.0, -2.0)),
    "order-search-seed0": search_order(ORDER),
    "saturated-search-seed0": search_order(SATURATED),
    **{f"basis-columns-{task}-seed0": columns_basis(run)
       for task, run in basis(AtomicMatching(context=64, width=4, heads=129, cap=4.0), "easy",
                              execution=Eager(device="cpu")).items()},
}
