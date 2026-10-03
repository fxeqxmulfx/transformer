"""GPTMini on parity under label noise across sample sizes and widths, and on binary copy across model sizes.

`scaling.py` of `experiments/synthetic_trainers` (its phases dd-calibration,
dd-widths and copy-scaling), in the experiment language: 27 runs of GPTMini
with 8 heads under AMSGradW with betas (0.9, 0.999), epsilon 1e-8 and decay
0.1, on 64 rows per update, observed in chunks of 64, each under the
double-descent study with curve tolerance 0.02 on data seed 1. A run draws
its batch order from its model seed, as `train_run` drew it.

- calibration: parity of 16 bits with 20% of its training labels corrupted
  (noise seed 2), 3000 updates at rate 3e-4 observed every 500, on 64, 256,
  1024, 4096 and 16384 training rows, each split a prefix of the next, with
  128 validation and 256 test rows, tested at 32 and 64 bits; width 64 and
  2 layers, from model seed 0.
- widths: the same on 512 training rows, at widths 16, 32, 64, 128, 256 and
  512, from model seeds 0, 1 and 2. 512 is the geometric midpoint of the
  bracket the archived calibration found, between the largest training split
  it fit (256) and the smallest larger one it did not (1024):
  `critical_sample_choice` of `scaling_calculations.py`.
- copy: binary strings of 1 to 32 symbols, 5000 updates at rate 1e-4
  observed every 1000, on 16384 training rows, 64 validation and 128 test,
  tested at 64 and 128 symbols, at widths 64 and 512 and depths 2 and 6,
  from model seed 0. 16384 is the power of two above the 12118 rows a union
  bound requires for every motif of 4 symbols to occur at every position of
  every length with probability 0.95 (`motif_bound`): a coverage condition
  weaker than the diversity condition of the RASP-Generalization Conjecture
  (arXiv:2310.16028v1, Section 2), and no guarantee about training. Its
  source cites Section 3 for that condition.

The archived runs (RTX 3050, 2026-10-01, issued eagerly;
`experiments/synthetic_trainers/baselines/amsgradw_softmax_scaling_20261002`)
found no double descent in width: the mean test loss and error of the last
models over width descend twice nowhere, and the error of two seeds alone
does, by at most 0.07 around chance. The calibration fit its noisy labels on
64 and 256 rows and not on 1024 or more. At 512 rows width 16 fit in no
run, every larger width fit in every run but one at width 512, and every
width tested at sequence accuracy 0.48 to 0.57, the chance of parity. Copy
tested at 0.96 at width 64 and depth 2 and 1.0 at the three larger sizes,
and 0 at 64 and 128 symbols at every size.

The runs replay from CUDA graphs here, so none is its archived run bit for
bit.
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
