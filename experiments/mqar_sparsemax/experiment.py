"""A RoPE transformer on multi-query associative recall, its softmax attention against sparsemax.

`README.md` beside this file describes the runs and what their archived
runs found.
"""

import math

from lab.dsl import *


def rope(width, context):
    """`RopeTransformer` of `convex_mqar/rope.py`: two layers, one head, mlp ratio 4, RoPE base 1e4."""
    return Transformer(
        width=width, depth=2, context=context,
        block=Block(
            attention=Attention(heads=1, projections=FusedQKV(), scores=ScaledDot(), weights=Softmax(fused=True),
                                exclusive=None),
            ffn=FFN(activation=GELU(tanh=True), bias=True), norm=LayerNorm(), residual=PreNorm()),
        positions=RoPE(interleaved=True), readout=Tied(), final_norm=LayerNorm(), init=ScaledResidual(0.02))


def run(length, width, lr, vocab=8192, train=100_000, held_out=3000, epochs=64):
    """`train_run` at one length, width and rate: an observation and a checkpoint after every epoch."""
    batch = 8 if max(length, width) >= 512 else 16 if max(length, width) >= 256 else 64
    epoch = math.ceil(train / batch)
    updates = epochs * epoch
    return Experiment(
        model=rope(width, length),
        benchmark=AssociativeRecall(length=length, vocab=vocab, alpha=0.1, train=train, validation=held_out,
                                    test=held_out),
        optimizer=AdamW(lr=lr, betas=(0.9, 0.999), weight_decay=0.1, decay="matrices"),
        schedule=Schedule(warmup=max(1, int(0.1 * updates)), inclusive=True),
        budget=Budget(updates=updates, batch=batch),
        seeds=Seeds(model=0, data=0, batches=0),
        evaluate=Evaluate(every=epoch, batch=batch),
        checkpoint=Checkpoint(every=epoch),
        execution=CudaGraph())


rates = {f"lr{rate:.8g}": rate for rate in (0.0001, 0.00046415888336127773, 0.002154434690031882, 0.01)}
softmax = grid({f"softmax-n{length}": run(length, 64, 0.0001) for length in (64, 128, 256, 512)},
               {"optimizer.lr": rates})

experiments = {
    **softmax,
    **{label.replace("softmax", "sparsemax", 1): substitute(experiment, Softmax, Sparsemax())
       for label, experiment in softmax.items()},
    "sanity": run(8, 32, 0.003, vocab=16, train=2048, held_out=512, epochs=128),
}
