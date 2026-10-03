"""Stability of GPTMini on x / y mod 97 under AdamW and raw AMSGradW.

`README.md` beside this file describes the runs and what their archived
runs found.
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
