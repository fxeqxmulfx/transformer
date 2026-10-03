"""GPTMini on Tiny Shakespeare under AMSGradW: softmax against sparsemax attention.

`README.md` beside this file describes the runs and what their archived
runs found.
"""

from lab.dsl import *

gptmini = Transformer(
    width=128, depth=2, context=64,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

softmax = Experiment(
    model=gptmini,
    benchmark=TinyShakespeare(window=64),
    optimizer=AMSGradW(lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01),
    schedule=Schedule(),
    budget=Budget(updates=20_000, batch=32),
    seeds=Seeds(model=0, data=0),
    evaluate=Evaluate(every=250, batch=32),
    stopping=EarlyStopping(patience=8, min_delta=1e-4, after=1000, divergence=0.1, divergence_patience=3),
    execution=CudaGraph())

experiments = grid({"softmax": softmax, "sparsemax": substitute(softmax, Softmax, Sparsemax())},
                   {"seeds.model": {"seed0": 0, "seed1": 1, "seed2": 2}})
