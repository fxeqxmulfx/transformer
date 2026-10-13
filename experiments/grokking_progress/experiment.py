"""Observe grokking in unchanged ordinary softmax transformers with AdamW.

Recipes: experiments/mod97_grokking at 5bc161d, porting openai/grok and
GPTMini. New measurements adapt Nanda et al., arXiv:2301.05217v1, section
5.1 to fixed division-scaling orbits, without selecting future frequencies.
The optimizer, data protocol and full 150,000-update budgets are retained.
"""

from lab.dsl import *

gptmini = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

reference = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=PerHeadQKV(), scores=ScaledDot(),
                            weights=Softmax(), exclusive=None),
        ffn=FFN(activation=ReLU()), norm=LayerNorm(), residual=PostNorm()),
    positions=Sinusoidal(), readout=Untied(), final_norm=None, init=TorchDefault())

base = Experiment(
    model=gptmini,
    benchmark=ModularDivision(prime=97, train_fraction=.5),
    optimizer=AdamW(lr=1e-3, betas=(.9, .98), weight_decay=.1),
    schedule=Schedule(warmup=10),
    budget=Budget(updates=150_000, batch=512),
    seeds=Seeds(model=1, data=1),
    evaluate=Evaluate(every=250),
    diagnostics=GrokkingDiagnostics(every=1000, orbit_every=1000, batch=512),
    execution=Eager(device="cuda", threads=1))

negative = swap(swap(swap(base, "model", reference), "benchmark.train_fraction", .2),
                "optimizer.weight_decay", 1.)
negative = swap(negative, "seeds", Seeds(model=0, data=0))
negative = swap(negative, "execution", Eager(device="cpu", threads=4))

experiments = {
    **grid({"gptmini": base}, {"seeds.model": {"seed1": 1, "seed2": 2, "seed3": 3}}),
    "reference-fraction20": negative,
}
