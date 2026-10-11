"""Dense GPTMini size search for modular division within 10,000 full epochs."""

from lab.dsl import *

model = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

# 4,656 training rows: nine batches of 512 and one of 48 per epoch.
base = Experiment(
    model=model,
    benchmark=ModularDivision(prime=97, train_fraction=0.5, tail="short"),
    optimizer=AdamW(lr=1e-3, betas=(0.9, 0.98), weight_decay=0.1),
    schedule=Schedule(warmup=10),
    budget=Budget(updates=100_000, batch=512),
    seeds=Seeds(model=1, data=1),
    evaluate=Evaluate(every=250, batch=1024),
    execution=CudaGraph(),
    checkpoint=Checkpoint(every=5000))

experiments = {
    "graph-w144": swap(base, "model.width", 144),
    "graph-w128": base,
    "graph-w96": swap(base, "model.width", 96),
    "graph-w64": swap(base, "model.width", 64),
    "graph-w48": swap(base, "model.width", 48),
    "graph-w32": swap(base, "model.width", 32),
    "graph-w16": swap(base, "model.width", 16),
    "graph-w8": swap(base, "model.width", 8),
    "graph-w32-wd02": swap(swap(base, "model.width", 32), "optimizer.weight_decay", 0.2),
    "graph-w24-wd02": swap(swap(base, "model.width", 24), "optimizer.weight_decay", 0.2),
    "graph-w16-wd02": swap(swap(base, "model.width", 16), "optimizer.weight_decay", 0.2),
    "native-w128": swap(base, "execution", Eager()),
}
