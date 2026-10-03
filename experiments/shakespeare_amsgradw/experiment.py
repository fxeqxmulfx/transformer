"""GPTMini on Tiny Shakespeare under AMSGradW: softmax against sparsemax attention.

The amsgradw arm of the AMSGrad extensions benchmark (RTX 3050), in the
experiment language. GPTMini of width 128, with 2 layers of 4 heads, reads 32
windows of 64 characters per update under AMSGradW at its validation-selected
rate 3e-4, with betas (0.9, 0.999), epsilon 1e-8 and decay 0.01, from model
seeds 0, 1 and 2. Validation is evaluated every 250 updates in chunks of 32
windows; a run stops after 8 observations without an improvement of 1e-4, or
after 3 consecutive observations 0.1 above its best, both from update 1000
on, and at the latest after 20,000 updates. The best validation model is
restored and evaluated once on test.

Every historical run stopped on patience, with mean test cross entropy 1.6254
under softmax and 1.6389 under sparsemax. They are not reproduced bit for
bit: the historical step was compiled by inductor, and it moved a parameter
as x - lr m / (eps + sqrt(vmax)) - (lr decay) x, which rounds otherwise than
x - lr (m / (sqrt(vmax) + eps) + decay x).
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
