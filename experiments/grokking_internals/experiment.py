"""Repeat the delayed GPTMini recipe while retaining intermediate weights.

Source: experiments/grokking_progress at 43d4d66, itself the mod97_grokking
ordinary softmax/AdamW recipe. Only checkpoint cadence changes to 1,000
updates; every optimizer, model and data setting and the 150,000 budget
are retained. Offline probes never feed updates into this model.
"""

from lab.dsl import *

model = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

base = Experiment(
    model=model,
    benchmark=ModularDivision(prime=97, train_fraction=.5),
    optimizer=AdamW(lr=1e-3, betas=(.9, .98), weight_decay=.1),
    schedule=Schedule(warmup=10), budget=Budget(updates=150_000, batch=512),
    seeds=Seeds(model=1, data=1), evaluate=Evaluate(every=250),
    diagnostics=GrokkingDiagnostics(every=1000, orbit_every=1000, batch=512),
    execution=Eager(device="cuda", threads=1))

experiments = {"gptmini-seed1": swap(base, "checkpoint", Checkpoint(every=1000))}
