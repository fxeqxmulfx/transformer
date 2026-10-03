"""Grokking x / y mod 97 with the openai/grok reference transformer and GPTMini.

The 2026-10-02 runs of `paper_reproduction.grokking`, in the experiment
language. Three calibrations of the reference on seeds 0/0 vary the train
fraction and the weight decay. The confirmation repeats the calibration that
passed its launch condition (50% train, weight decay 0.1) in three arms: the
reference and GPTMini under AdamW, and GPTMini under raw AMSGradW with betas
(0.9, 0.999), on data seed 1 with initialization seeds 1, 2 and 3.

Eager execution issues the updates as the historical trainer did.
"""

from lab.dsl import *

reference = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=PerHeadQKV(), scores=ScaledDot(),
                            weights=Softmax(), exclusive=None),
        ffn=FFN(activation=ReLU()), norm=LayerNorm(), residual=PostNorm()),
    positions=Sinusoidal(), readout=Untied(), final_norm=None, init=TorchDefault())

gptmini = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

calibration = Experiment(
    model=reference,
    benchmark=ModularDivision(prime=97, train_fraction=0.2),
    optimizer=AdamW(lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0),
    schedule=Schedule(warmup=10),
    budget=Budget(updates=150_000, batch=512),
    seeds=Seeds(model=0, data=0),
    evaluate=Evaluate(every=250),
    execution=Eager())

half = swap(calibration, "benchmark.train_fraction", 0.5)
passed = swap(half, "optimizer.weight_decay", 0.1)
confirmation = swap(passed, "seeds.data", 1)

experiments = {
    "fraction20-wd1": calibration,
    "fraction50-wd1": half,
    "fraction50-wd01": passed,
    **grid({
        "reference-adamw": confirmation,
        "gptmini-adamw": swap(confirmation, "model", gptmini),
        "gptmini-amsgradw": swap(swap(confirmation, "model", gptmini), "optimizer",
                                 AMSGradW(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.1)),
    }, {"seeds.model": {"seed1": 1, "seed2": 2, "seed3": 3}}),
}
