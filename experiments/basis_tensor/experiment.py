"""Measured original-softmax reference for the verified convex tensor cycle.

Candidate arms enter only after the corresponding first-success FLOP
budget has been measured. Source: plan.md stage 3 and experiments/basis.
"""

from lab.dsl import *


def gptmini(width, depth):
    attention = Attention(heads=4, projections=FusedQKV(), scores=QKNorm(), weights=Softmax(), exclusive=XSA())
    block = Block(attention=attention, ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm())
    return Transformer(width=width, depth=depth, block=block, positions=RoPE(), readout=Tied(),
                       final_norm=RMSNorm(), init=Normal(.02), context=64)


THREADS = {("hard", "recall"): 4, ("hard", "depth"): 2, ("hard", "parity"): 2}
MODELS = {"easy": gptmini(64, 2), "hard": gptmini(128, 6)}

experiments = {
    f"softmax-{mode}-{task}-seed{seed}": swap(
        swap(run, "execution", Measured(device="cpu", threads=THREADS.get((mode, task), 1))),
        "checkpoint", Checkpoint(every=run.evaluate.every))
    for mode, model in MODELS.items() for seed in (0, 1, 2)
    for task, run in basis(model, mode, seed).items()
}
