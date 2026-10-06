"""Does ANSR's Cohesive-like copy-task behavior transfer to ordinary softmax GPTMini on Basis?"""

from lab.dsl import *


def gptmini():
    """The unchanged small basis GPTMini at 5d91bc4: width 64, two layers, four softmax heads."""
    attention = Attention(heads=4, projections=FusedQKV(), scores=QKNorm(), weights=Softmax(), exclusive=XSA())
    block = Block(attention=attention, ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm())
    return Transformer(width=64, depth=2, block=block, positions=RoPE(), readout=Tied(),
                       final_norm=RMSNorm(), init=Normal(0.02), context=64)


def population(run, p_self):
    """100,000 generations; refresh attractors on the same sampled batch as their challengers."""
    run = swap(run, "schedule", Schedule())
    run = swap(run, "execution", Eager(device="cuda"))
    run = swap(run, "optimizer", ANSR(p_self=p_self))
    run = swap(run, "budget.updates", 100_000)
    run = swap(run, "evaluate.every", 32)
    run = swap(run, "diagnostics", Diagnostics(every=32))
    return swap(run, "checkpoint", Checkpoint(every=32))


BASE = basis(gptmini(), "easy", 0)
experiments = {
    **{f"adamw-easy-small-{task}-seed0": swap(run, "execution", Eager(device="cuda"))
       for task, run in BASE.items()},
    **{f"ansr-{arm}-easy-small-{task}-seed0": population(run, p_self)
       for arm, p_self in (("low", 0.05), ("high", 0.95)) for task, run in BASE.items()},
}
