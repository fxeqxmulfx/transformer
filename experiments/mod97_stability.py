"""Stability of GPTMini on x / y mod 97 under AdamW and raw AMSGradW.

The 2026-10-02 optimizer pair of the adamw_stability protocols and the
amsgradw_stability calibrations, in the experiment language. The pair trains
GPTMini on 50% of the equations for 150,000 updates at rate 1e-3 with weight
decay 0.1, under AdamW with betas (0.9, 0.98) and under raw AMSGradW with
betas (0.9, 0.999); nothing else differs. Each AMSGradW calibration changes
one mechanism of its arm: the batch tail wraps across epochs, or the rate is
3e-4, 2e-4 or 1e-4.

Every run samples per-tensor diagnostics at each evaluation and at the updates
beside it, and records every gradient norm; the historical calibrations did
not record gradient norms, and no diagnostic changes a trajectory. Eager
execution issues the updates as the historical trainer did.
"""

from lab.dsl import *

gptmini = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

adamw = Experiment(
    model=gptmini,
    benchmark=ModularDivision(prime=97, train_fraction=0.5),
    optimizer=AdamW(lr=1e-3, betas=(0.9, 0.98), weight_decay=0.1),
    schedule=Schedule(warmup=10),
    budget=Budget(updates=150_000, batch=512),
    seeds=Seeds(model=0, data=0),
    evaluate=Evaluate(every=250),
    diagnostics=Diagnostics(every=250, neighbors=True, gradients=True),
    execution=Eager())

amsgradw = swap(adamw, "optimizer", AMSGradW(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1))

experiments = {
    "adamw": adamw,
    "amsgradw": amsgradw,
    "amsgradw-wrap": swap(amsgradw, "benchmark.tail", "wrap"),
    **grid({"amsgradw": amsgradw}, {"optimizer.lr": {"lr0003": 3e-4, "lr0002": 2e-4, "lr0001": 1e-4}}),
}
