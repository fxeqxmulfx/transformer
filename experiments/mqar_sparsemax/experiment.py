"""A RoPE transformer on multi-query associative recall, its softmax attention against sparsemax.

`experiments/convex_mqar` (the `full` profile, its `attention_ablation`, and
the `sanity` profile), in the experiment language: the two-layer pre-norm
transformer of width 64 with one head of interleaved RoPE, LayerNorm, a
tanh-GELU feed-forward layer of width 256 with biases, a tied readout and
GPT-2's initialization at 0.02, under AdamW with betas (0.9, 0.999), epsilon
1e-8 and decay 0.1 on matrices alone, the rate warmed up linearly over the
first tenth of the updates (counting the update being made), for 64 epochs
of 100,000 training sequences. Validation (3000 sequences) is observed after
every epoch and selects the best model by accuracy, then loss; the test
(3000 sequences) is evaluated once, on it. Every seed is 0. A batch, and an
evaluation chunk, holds 64 sequences, 16 from length 256 and 8 from 512
(`Config.batch_size`).

- softmax-n{length}-lr{rate} and sparsemax-n{length}-lr{rate}: lengths 64,
  128, 256 and 512, a query for every fourth token, over 8192 tokens with
  query positions weighted p^-0.1, at the rates `np.logspace(-4, -2, 4)`.
  Sparsemax replaces softmax and nothing else.
- sanity: softmax at length 8 over 16 tokens, width 32, rate 3e-3, on 2048
  training sequences and 512 validation and test, for 128 epochs.

The archived runs (RTX 3050, bf16 autocast, fused AdamW, the loss compiled
though some softmax epochs ran eagerly;
`experiments/archive/convex_mqar/reports/validation_milestones.json`) covered
lengths 64, 128 and 256; the archive holds no run at length 512. Softmax
reached 99% validation accuracy in 2 of its 12 runs: at length 128 under
rate 1e-2 in epoch 18 (and fell to 0.0065 by the last epoch), and at length
256 under 2.2e-3 in epoch 5; at length 64 it peaked at 0.25. Sparsemax
reached it in 11 of 12, all but length 256 under 1e-2, and at length 64
within three epochs at every rate. The sanity run reached validation and
test accuracy 1.0 (`reports/rope_sanity.json`).

Here every run trains in float32 and replays from CUDA graphs, and the
initial model is observed too, so none is its archived run bit for bit.
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
