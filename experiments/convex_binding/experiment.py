from lab.dsl import *

"""Learn key/value binding with factorized neighboring-token keys and answer loss.

Source: pairedHeadOutput, pairedMixture_criterion_convex and pairedBinding_fit.
Only the encoder differs between the matched Basis arms. No routes are labeled.
"""

BINDING = Experiment(
    model=PairedMatching(context=6, width=4, heads=3, cap=1.0, channels=1,
                         initial_value=0.5, precision="float64"),
    benchmark=MatchingBindings(), optimizer=AtomicColumns(pricing=BindingPricing()),
    schedule=Schedule(), budget=Budget(updates=1_000, batch=2), seeds=Seeds(),
    evaluate=Evaluate(every=10, batch=2), execution=Eager(device="cpu"),
    diagnostics=Diagnostics(every=10), checkpoint=Checkpoint(every=10),
)


def numerical(run):
    """Search all Q/K coordinates with exact conditional original values."""
    run = swap(run, "optimizer.pricing", SearchPricing(restarts=4, steps=64))
    run = swap(run, "model.heads", 129)
    run = swap(run, "budget.updates", 128)
    return swap(run, "evaluate.every", 16)


def content_only(run):
    """Use the original content-only head at the same total Q/K width."""
    return swap(run, "model", AtomicMatching(context=6, width=4, heads=run.model.heads,
                                            cap=1, channels=1, initial_value=0.5,
                                            precision="float64"))


def raw_control(run):
    run = swap(run, "optimizer", AdamW(lr=0.01, betas=(0.9, 0.999), weight_decay=0))
    return swap(run, "model.heads", 1)


def recall_columns(model):
    """512 updates, 266,240 pricing forwards, unchanged easy MQAR data/loss."""
    run = basis(model, "easy", execution=Eager(device="cuda", threads=4))["recall"]
    run = swap(run, "schedule", Schedule())
    run = swap(run, "optimizer", AtomicColumns(pricing=SearchPricing(restarts=4, steps=64)))
    run = swap(run, "budget.updates", 512)
    run = swap(run, "evaluate", Evaluate(every=16, batch=32))
    run = swap(run, "diagnostics", Diagnostics(every=16))
    return swap(run, "checkpoint", Checkpoint(every=16))


experiments = {
    **grid({"paired-exact": BINDING, "paired-search": numerical(BINDING),
            "content-search": content_only(numerical(BINDING)),
            "content-adamw": content_only(raw_control(BINDING))},
           {"seeds.model": {f"seed{seed}": seed for seed in range(4)}}),
    "basis-content-recall-seed0": recall_columns(
        AtomicMatching(context=64, width=8, heads=513, cap=4)),
    "basis-paired-recall-seed0": recall_columns(
        PairedMatching(context=64, width=8, heads=513, cap=4)),
}
