"""GPTMini on the synthetic suite under AMSGradW: the softmax baseline and its studies.

`README.md` beside this file describes the runs and what their archived
runs found.
"""

from lab.dsl import *

gptmini = Transformer(
    width=64, depth=2, context=64,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

study = Memorization(tolerance=0.02)


def run(benchmark, updates=1000, every=100):
    """GPTMini on `benchmark`, through a context of the benchmark's own size."""
    return Experiment(
        model=swap(gptmini, "context", benchmark.context),
        benchmark=benchmark,
        optimizer=AMSGradW(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1),
        schedule=Schedule(),
        budget=Budget(updates=updates, batch=32),
        seeds=Seeds(model=0, data=1, batches=0),
        evaluate=Evaluate(every=every, batch=32),
        execution=CudaGraph())


def suite(task, length=16, min_length=8, ood=(32, 64)):
    return run(Synthetic(task=task, length=length, min_length=min_length, train=128, validation=32, test=64,
                         ood=ood, study=study))


recall = {"symbols": 64, "pairs": 4, "queries": 2}
numbers = {"symbols": 64, "number_limit": 128}
tasks = {
    "mqar": suite(MQAR(**recall), length=24, min_length=24, ood=(48, 96)),
    "lookup": suite(Lookup(**recall), length=24, min_length=24, ood=(48, 96)),
    "dyck": suite(Dyck(), min_length=12),
    "blocks": suite(AlternatingBlocks()),
    "histogram-bos": suite(Histogram(**numbers)),
    "histogram-no-bos": suite(Histogram(bos=False, **numbers)),
    "histogram2": suite(DoubleHistogram(**numbers)),
    "mode-none": suite(Mode(**numbers)),
    "mode-counts": suite(Mode(scratchpad="counts", **numbers)),
    "mode-itemized": suite(Mode(scratchpad="itemized", **numbers)),
    "most_freq": suite(MostFrequent(symbols=64)),
    **{f"{name}-{kind}": suite(task(symbols=64, unique=unique))
       for name, task in (("copy", Copy), ("reverse", Reverse), ("sort", Sort))
       for kind, unique in (("repeat", False), ("unique", True))},
    "dyck-2": suite(TypedDyck(), min_length=12),
    "dyck-3": suite(TypedDyck(types=3), min_length=12),
    "count": suite(Count(**numbers)),
    **{f"addition-{order}-{plain}-{carries}": suite(Addition(order=order, hints=hints, carries=carries, **numbers),
                                                    length=4, min_length=2, ood=(8, 16))
       for order in ("forward", "reverse") for plain, hints in (("plain", False), ("hints", True))
       for carries in ("standard", "balanced")},
    "parity-none": suite(Parity(**numbers)),
    "parity-running": suite(Parity(scratchpad="running", **numbers)),
    "parity-running-hints": suite(Parity(scratchpad="running", hints=True, **numbers)),
    "parity-ones-hints": suite(Parity(scratchpad="ones", hints=True, **numbers)),
    "boolean-and-shift": suite(BooleanAnd(symbols=64)),
    "boolean-and-random": suite(BooleanAnd(shift=False, symbols=64)),
    **{f"crasp-depth-{depth}": suite(CRASP(depth=depth)) for depth in (1, 2, 3)},
}

# Disjoint pools of 64 training rows.
pools = {"train": 64, "validation": 64, "test": 128, "ood": (16, 32)}
disjoint = swap(study, "disjoint", True)
parity = Parity(symbols=8, number_limit=64)
transitions = {
    "copy-repeat": run(Synthetic(task=Copy(symbols=8), length=8, min_length=4, study=disjoint, **pools), 5000, 250),
    "parity-none": run(Synthetic(task=parity, length=8, study=disjoint, **pools), 5000, 250),
    "parity-running": run(Synthetic(task=swap(parity, "scratchpad", "running"), length=8, study=disjoint, **pools),
                          5000, 250),
}

noisy = Memorization(disjoint=True, noise=0.2, noise_seed=2, tolerance=0.02)
capacity = grid(run(Synthetic(task=parity, length=8, study=noisy, **pools)),
                {"model.width": {f"width{width}": width for width in (16, 32, 64, 128)}})

control = run(Synthetic(task=RandomLM(symbols=4, number_limit=64), length=8, train=8, validation=64, test=128,
                        ood=(16, 32), study=study))

seeds = {f"seed{seed}": Seeds(model=seed, data=1, batches=seed) for seed in (0, 1, 2)}
experiments = {
    **{f"suite-{name}": experiment for name, experiment in tasks.items()},
    **grid({f"transitions-{name}": experiment for name, experiment in transitions.items()}, {"seeds": seeds}),
    **grid({f"capacity-{name}": experiment for name, experiment in capacity.items()}, {"seeds": seeds}),
    **grid({"control": control}, {"seeds": seeds}),
}
