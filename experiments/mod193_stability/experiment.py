"""Stability of GPTMini on x / y mod 193 under native AdamW.

The 2026-10-02 adamw_stability protocols, in the experiment language. The
base is the calibration at rate 3e-4 on 25% train, extended to 300,000
updates; two pairs compare it with sparsemax attention weights and with a
cosine anneal of the rate to a tenth over updates 150,000-250,000. The two
earlier calibrations at rate 1e-3, on 25% and 50% train, ran 150,000 updates.

Every run samples per-tensor diagnostics at each evaluation and at the updates
beside it, and records every gradient norm. Eager execution issues the updates
as the historical trainer did.
"""

from lab.dsl import *

gptmini = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

base = Experiment(
    model=gptmini,
    benchmark=ModularDivision(prime=193, train_fraction=0.25),
    optimizer=AdamW(lr=3e-4, betas=(0.9, 0.98), weight_decay=0.1),
    schedule=Schedule(warmup=10),
    budget=Budget(updates=300_000, batch=512),
    seeds=Seeds(model=0, data=0),
    evaluate=Evaluate(every=250),
    diagnostics=Diagnostics(every=250, neighbors=True, gradients=True),
    execution=Eager())

faster = swap(swap(base, "optimizer.lr", 1e-3), "budget.updates", 150_000)

experiments = {
    "base": base,
    "sparsemax": substitute(base, Softmax, Sparsemax()),
    "cosine": swap(base, "schedule.anneal", Cosine(start=150_000, end=250_000, final=0.1)),
    "lr001": faster,
    "fraction50-lr001": swap(faster, "benchmark.train_fraction", 0.5),
}
