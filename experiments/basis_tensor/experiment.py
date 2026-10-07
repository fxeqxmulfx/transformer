"""Measured original-softmax reference for the verified convex tensor cycle.

Candidate arms enter only after the corresponding first-success FLOP
budget has been measured. Source: plan.md stage 3 and experiments/basis.
The pinned ceilings below come from first successful history observations
under aten-reference-arithmetic-v1, lab revision 0700c49.
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

# Preserve the unsuccessful original 10,000-step attempt and continue
# without changing any trajectory setting except the update ceiling.
experiments["softmax-hard-depth-seed0"] = swap(experiments["softmax-hard-depth-seed0"], "budget.updates", 20_000)

# Keep failed/unfinished references out of this table. Candidate updates
# spend the full measured ceiling with the unchanged AdamW recipe.
REFERENCE_FLOPS = {
    ("easy", "depth", 0): 148_978_162_800,
    ("easy", "depth", 1): 148_978_162_800,
    ("easy", "depth", 2): 148_978_162_800,
    ("easy", "recall", 0): 4_989_090_525_500,
    ("easy", "recall", 1): 7_408_685_409_020,
    ("easy", "recall", 2): 5_443_331_961_200,
    ("easy", "parity", 0): 1_609_311_910_000,
    ("easy", "parity", 1): 981_287_750_000,
    ("easy", "parity", 2): 824_281_710_000,
    ("hard", "depth", 1): 11_225_096_867_600,
    ("hard", "depth", 2): 43_296_802_203_600,
    ("hard", "recall", 0): 107_443_940_040_500,
    ("hard", "recall", 1): 84_991_530_151_660,
    ("hard", "recall", 2): 76_977_398_143_040,
    ("hard", "parity", 0): 9_475_216_186_800,
    ("hard", "parity", 1): 16_243_227_748_800,
    ("hard", "parity", 2): 19_852_833_915_200,
}

for (mode, task, seed), ceiling in REFERENCE_FLOPS.items():
    reference = experiments[f"softmax-{mode}-{task}-seed{seed}"]
    model = TensorStack(width=reference.model.width, depth=reference.model.depth,
                        context=reference.model.context)
    candidate = swap(reference, "model", model)
    candidate = swap(candidate, "budget", FlopBudget(updates=100_000, batch=reference.budget.batch,
                                                     flops=ceiling))
    experiments[f"tensor-{mode}-{task}-seed{seed}"] = swap(candidate, "stopping", None)
