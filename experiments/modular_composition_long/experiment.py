"""Frozen full-state follow-up; target100000 absolute updates, two-hour cohort cap.

Adaptive selection after the early screen; optimizer/model/task settings are
unchanged. Evaluation cadence is retained; only checkpoint retention differs.
"""

from lab.dsl import *

base = Experiment(
    model=SourceTransformer(width=256, depth=1, heads=4, context=33),
    benchmark=ModularPairs(prime=97),
    optimizer=ScheduleFreeAdamW(lr=.003, betas=(.9, .98), weight_decay=.1, warmup=0),
    schedule=Schedule(warmup=10, inclusive=True),
    budget=Budget(updates=100000, batch=256),
    seeds=Seeds(model=11, data=11, batches=11),
    evaluate=Evaluate(every=250, batch=512),
    execution=MaxAutotune(device="cuda"),
)
recipes = grid({"pairs": base}, {
    "benchmark.operation": {"mixed": "mixed", "division": "division"},
    "benchmark.prime": {"p97": 97, "p193": 193},
    "model": {"w256d1": base.model, "w128d4": SourceTransformer(128, 4, 4, 33)},
})
recipes["full-mixed-p193"] = swap(swap(base, "benchmark.prime", 193), "benchmark.train_fraction", 1.)
recipes["chain-p97-l1-w128d4"] = swap(swap(base, "model", SourceTransformer(128, 4, 4, 33)),
    "benchmark", ModularChain(97, 1, 2048, 1024, 1024))

sources = [
    ('pairs-division-p97-w256d1', 'b3bc378f98151191722298d367ace038af1822f17f3bbf4eb94dedfbf020612c'),
    ('pairs-mixed-p97-w256d1', '99ca9e511113cdfd21548a70eb40798a595ac6e467d0a15705fcb9aa520a40e1'),
    ('pairs-division-p193-w128d4', 'bdb2a7f38d151a64941fe1944af6acee32cd0b2c4b01ead383ebf14e9c41fcb4'),
    ('pairs-mixed-p193-w128d4', '4b337464af09d28d625882f03e2ce3a21ddb9bad3f781f7720df75f38770ca28'),
    ('pairs-division-p193-w256d1', 'c91258e735e761a5dbe8739e0e0840861483eead964f0a66acf412afa6761e63'),
    ('pairs-mixed-p193-w256d1', 'e383989ed9bb63a71b91deb279b24f3427d7a99101583241ff5db49d593e6471'),
    ('pairs-division-p97-w128d4', 'cab2073fd2d658ab49583523c8c8a14a44f9fdac91c9a0b5cb1748009831d430'),
    ('pairs-mixed-p97-w128d4', '3a349dc098b70235f0f737c1e65b5762b08a6ccebf9e5ba3315fecc1987098d7'),
    ('full-mixed-p193', '79b1207c10cdb81bc48de003814c9b76bd71ca45b1cb4418201617a659aadc89'),
    ('chain-p97-l1-w128d4', '353ebf457502e79ec313a9757ec3f2c330ed42f888d2e419dd8f1d597f1cf5a3'),
]
experiments = {label: swap(recipes[label], "checkpoint", ScheduleFreeContinuation(
    10000, f"experiments/modular_composition/runs/{label}/checkpoint.pt", digest))
    for label, digest in sources}
