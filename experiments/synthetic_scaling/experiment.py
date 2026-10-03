"""GPTMini on parity under label noise across sample sizes and widths, and on binary copy across model sizes.

`README.md` beside this file describes the runs and what their archived
runs found.
"""

from lab.dsl import *

gptmini = Transformer(
    width=64, depth=2, context=64,
    block=Block(
        attention=Attention(heads=8, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))


def run(benchmark, lr, updates, every):
    """GPTMini on `benchmark`, through a context of the benchmark's own size."""
    return Experiment(
        model=swap(gptmini, "context", benchmark.context),
        benchmark=benchmark,
        optimizer=AMSGradW(lr=lr, betas=(0.9, 0.999), weight_decay=0.1),
        schedule=Schedule(),
        budget=Budget(updates=updates, batch=64),
        seeds=Seeds(model=0, data=1, batches=0),
        evaluate=Evaluate(every=every, batch=64),
        execution=CudaGraph())


study = Memorization(noise=0.2, noise_seed=2, tolerance=0.02)
noisy = run(Synthetic(task=Parity(symbols=2), length=16, train=512, validation=128, test=256, ood=(32, 64),
                      study=study), 3e-4, 3000, 500)
calibration = grid(noisy, {"benchmark.train": {f"samples{size}": size for size in (64, 256, 1024, 4096, 16384)}})
widths = grid(noisy, {"model.width": {f"width{width}": width for width in (16, 32, 64, 128, 256, 512)},
                      "seeds": {f"seed{seed}": Seeds(model=seed, data=1, batches=seed) for seed in (0, 1, 2)}})

copy = run(Synthetic(task=Copy(symbols=2), length=32, min_length=1, train=16384, validation=64, test=128,
                     ood=(64, 128), study=swap(study, "noise", 0.0)), 1e-4, 5000, 1000)
sizes = grid(copy, {"model.width": {"width64": 64, "width512": 512}, "model.depth": {"depth2": 2, "depth6": 6}})

experiments = {
    **{f"calibration-{name}": experiment for name, experiment in calibration.items()},
    **{f"widths-{name}": experiment for name, experiment in widths.items()},
    **{f"copy-{name}": experiment for name, experiment in sizes.items()},
}
