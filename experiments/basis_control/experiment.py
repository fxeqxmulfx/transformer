"""Task162: historical GPTMini under the same Schedule-Free hard-recall recipe."""

from lab.dsl import *

# Same model definition as experiments/basis/experiment.py:gptmini at bd63e509.
attention = Attention(heads=4, projections=FusedQKV(), scores=QKNorm(), weights=Softmax(), exclusive=XSA())
block = Block(attention=attention, ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm())
model = Transformer(width=128, depth=6, block=block, positions=RoPE(), readout=Tied(),
                    final_norm=RMSNorm(), init=Normal(.02), context=64)
recipe = basis(model, "hard")["recall"]
base = Experiment(model=recipe.model, benchmark=BasisControl(**vars(recipe.benchmark)),
                  optimizer=ScheduleFreeAdamW(lr=recipe.optimizer.lr, betas=(.9, .98), weight_decay=.1),
                  schedule=recipe.schedule, budget=swap(recipe.budget, "updates", 100000), seeds=recipe.seeds,
                  evaluate=swap(recipe.evaluate, "batch", 64), checkpoint=Checkpoint(recipe.evaluate.every),
                  diagnostics=Diagnostics(every=recipe.evaluate.every, gradients=True),
                  execution=ResidualMaxAutotune(device="cuda", attention_backend="math"))
experiments = grid({"gptmini-hard-recall": base}, {"seeds.model": {"s0": 0, "s1": 1}})
