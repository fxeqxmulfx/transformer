"""The two historical models and a modular-division run, written in the DSL."""

from lab.dsl import (FFN, XSA, AdamW, Attention, Block, Budget, Eager, Evaluate, Experiment, FusedQKV,
                     LayerNorm, ModularDivision, Normal, PerHeadQKV, PostNorm, PreNorm, QKNorm, ReLU,
                     ReLU2, RMSNorm, RoPE, ScaledDot, Schedule, Seeds, Sinusoidal, Softmax, Tied,
                     TorchDefault, Transformer, Untied)


def gptmini(width=128, depth=2, heads=4):
    """`experiments/gpt_mini.py` as `paper_reproduction.grokking` built it."""
    attention = Attention(heads=heads, projections=FusedQKV(), scores=QKNorm(), weights=Softmax(),
                          exclusive=XSA())
    block = Block(attention=attention, ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm())
    return Transformer(width=width, depth=depth, block=block, positions=RoPE(), readout=Tied(),
                       final_norm=RMSNorm(), init=Normal(0.02), context=50)


def reference(width=128, depth=2, heads=4):
    """The openai/grok decoder as `paper_reproduction.reference_transformer` defines it."""
    attention = Attention(heads=heads, projections=PerHeadQKV(), scores=ScaledDot(), weights=Softmax(),
                          exclusive=None)
    block = Block(attention=attention, ffn=FFN(activation=ReLU()), norm=LayerNorm(), residual=PostNorm())
    return Transformer(width=width, depth=depth, block=block, positions=Sinusoidal(), readout=Untied(),
                       final_norm=None, init=TorchDefault(), context=50)


def modular(model, *, prime=97, updates=150_000, batch=512, every=250, lr=1e-3, decay=1.0,
            execution=Eager(device="cpu")):
    return Experiment(model=model, benchmark=ModularDivision(prime=prime, train_fraction=0.2),
                      optimizer=AdamW(lr=lr, betas=(0.9, 0.98), weight_decay=decay),
                      schedule=Schedule(warmup=10), budget=Budget(updates=updates, batch=batch),
                      seeds=Seeds(model=0, data=0), evaluate=Evaluate(every=every), execution=execution)
