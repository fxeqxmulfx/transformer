"""The basis: depth, recall and parity, three tasks a transformer solves by different means, in two modes.

Each task is a synthetic benchmark (`Synthetic`) passed when its selection
split reaches sequence accuracy 0.99; a run stops there (`Solved`) or at
its budget. The easy mode is set for a transformer of width 64 and two
layers, the hard mode for one of width 128 and six layers: the smaller
fails its depth and recall. Parity is the same in both modes;
`experiments/basis` measures all of it.

depth
    E_k (`AlternatingBlocks`) of arXiv:2506.16055v3, Appendix F,
    `thm:tlclpos_depth_hierarchy`: definable at depth k of TL[◁#]^pos and
    not below (`Transformer.CRASP.definablePos_altPlusNeutral`), and
    recognized by no fixed-precision transformer of k - 1 layers of one
    head whose RoPE angles are rational multiples of π
    (`Transformer.CRASP.not_recognizes_altPlusNeutral_rope`), hence by none
    of fewer layers, which is one of k - 1 whose last layers pass their
    input on. The bound holds at every length, and at bounded lengths the
    language is finite, so a run trains at lengths 32 to 64, and the hard
    mode judges E_4 at 128 (`select`). The models trained here have four
    heads a layer, compute in floating point, and turn by `RoPE` angles no
    rational multiple of π, so the bound predicts the outcome without
    covering it. The easy mode judges E_2 at the trained lengths and only
    tests it at 128: the smaller model learns E_2 at the trained lengths
    from every seed, and carries it to 128 from about half of them.
recall
    Multi-query associative recall (`MQAR`) of Zoology (arXiv:2312.04927v1,
    Appendix E.1) over 256 keys and 256 values, more tokens than either
    width: over c tokens, once each key and its value share a position,
    recall is counting at depth 1 of TL[◁#], c² counts in all, with no
    attention from a query to a key
    (`Transformer.CRASP.exists_rtfr_answers`). Attention that ignores the
    query needs, over n + 1 >= c rows, at k layers of width d and p bits,
    c - 1 <= k (2d + 1) log2(2^p (n + 1) + 1)
    (`Transformer.CRASP.two_pow_le_of_queryFree`), a width growing with c,
    while one layer of width 5 whose attention compares the query with each
    key recalls at p = O(log c + log n) bits
    (`Transformer.CRASP.matcher_answers`), so the two separate at every
    depth and width (`Transformer.CRASP.exists_matcher_not_queryFree`). The
    runs bind 8 or 16 keys, far fewer rows than tokens, where the bound says
    nothing, and that more tokens than width rule counting out there is not
    proved. Easy binds 8 keys. Hard writes 16 times, 8 of them
    rebinding a key, and answers a query with its key's latest value
    (`Transformer.ALM.latest_wins`). Both train on 20,000 rows, not
    Zoology's 100,000 (Appendix E.2), on which the smaller model passes
    later, or from one seed of three not within twice the budget.
parity
    Whether 1 to 16 bits hold an odd number of ones (`Parity`), without a
    scratchpad: RASP-L has no program for it (arXiv:2310.16028v1, §5.2),
    and without one the paper's transformer fits not even its training set
    (Appendix C.1). No formula of TL[◁#] defines it, so no future-masked
    rounded transformer recognizes it at every length
    (`Transformer.CRASP.not_recognizes_parity`); up to 16 bits it is a
    finite language, which a transformer can fit, and no run judges a
    longer string. The larger model learns parity of 16 bits in more
    updates than the smaller, and of 20 bits, from seed 0, not within
    12,000, where the smaller does; of 24 bits neither passes reliably
    within 16,000, so a longer parity would make the hard mode a draw
    rather than a test of size.

Every split is drawn from data seed 1: 20,000 training rows, 512
validation and 512 test. A run trains under AdamW with betas (0.9, 0.98)
and weight decay 0.1 after a linear warmup over 50 updates, at the batch,
rate and budget its task and mode calibrated on both models
(`experiments/basis/README.md`), and is observed after every 3,200
examples, in chunks of 256 rows, replaying CUDA graphs.
"""

from typing import NamedTuple

from .experiment import Experiment
from .generative import Parity
from .optimizers import AdamW
from .spec import require, swap
from .stopping import Solved
from .synthetic import Synthetic
from .tasks import MQAR, AlternatingBlocks
from .training import Budget, CudaGraph, Evaluate, Schedule, Seeds

MODES = ("easy", "hard")
TASKS = ("depth", "recall", "parity")
OBSERVED = 3_200
WARMUP = 50


class Recipe(NamedTuple):
    """Rows per update, rate and updates of a task's budget."""
    batch: int
    lr: float
    updates: int


def synthetic(task, length, min_length, ood=(), select=None):
    return Synthetic(task=task, length=length, min_length=min_length, train=20_000, validation=512, test=512,
                     ood=ood, target=0.99, select=select)


def depth(blocks, select=None):
    return synthetic(AlternatingBlocks(blocks=blocks), 64, 32, (128,), select)


def recall(min_length, **writes):
    return synthetic(MQAR(symbols=256, queries=8, **writes), 64, min_length)


PARITY = (synthetic(Parity(), 16, 1), Recipe(16, 3e-4, 20_000))

BASIS = {
    "easy": {"depth": (depth(2), Recipe(16, 1e-3, 2_000)),
             "recall": (recall(32, pairs=8), Recipe(64, 1e-3, 4_800)),
             "parity": PARITY},
    "hard": {"depth": (depth(4, "length-128"), Recipe(16, 1e-3, 10_000)),
             "recall": (recall(48, pairs=16, overwrites=8), Recipe(64, 3e-4, 4_800)),
             "parity": PARITY},
}


def basis(model, mode, seed=0):
    """The run of `model` on each task of `mode`, by task, from model seed `seed`."""
    require(mode in MODES, f"The mode is one of {', '.join(MODES)}")
    return {task: Experiment(
                model=swap(model, "context", benchmark.context), benchmark=benchmark,
                optimizer=AdamW(lr=recipe.lr, betas=(0.9, 0.98), weight_decay=0.1),
                schedule=Schedule(warmup=WARMUP), budget=Budget(updates=recipe.updates, batch=recipe.batch),
                seeds=Seeds(model=seed, data=1), evaluate=Evaluate(every=OBSERVED // recipe.batch, batch=256),
                execution=CudaGraph(), stopping=Solved())
            for task, (benchmark, recipe) in BASIS[mode].items()}
