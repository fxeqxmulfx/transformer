"""Does sparsemax need query/key normalization, once its learned head gain is retained?"""

from lab.dsl import *


def gptmini(width, depth):
    """The unchanged four-head basis GPTMini at bd63e50."""
    attention = Attention(heads=4, projections=FusedQKV(), scores=QKNorm(), weights=Softmax(), exclusive=XSA())
    block = Block(attention=attention, ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm())
    return Transformer(width=width, depth=depth, block=block, positions=RoPE(), readout=Tied(),
                       final_norm=RMSNorm(), init=Normal(0.02), context=64)


MODELS = {'small': gptmini(64, 2), 'large': gptmini(128, 6)}
THREADS = {('large', 'recall'): 4, ('large', 'depth'): 2, ('large', 'parity'): 2}


def matched_gain(model):
    """Initialization-only estimate, not fitted to validation labels or outcomes.

    With unit-RMS inputs and Normal(std) projections, a query component has
    variance approximately width * std^2. ScaledDot scores therefore have
    that standard deviation, whereas unit-cosine scores have about
    1 / sqrt(head width). This gain matches those estimates, not each row.
    """
    return 1 / (model.width * model.init.std ** 2 * model.head_width ** 0.5)


def variants(model):
    """Three score controls and two raw-query/key learned-scale candidates."""
    sparse = substitute(model, Softmax, Sparsemax())
    return {
        'softmax': model,
        'qknorm-one': substitute(sparse, QKNorm, QKNorm(initial_scale=1.0)),
        'scaleddot': substitute(sparse, QKNorm, ScaledDot()),
        'learned-dot-one': substitute(sparse, QKNorm, LearnedScaledDot(initial_scale=1.0)),
        'learned-dot-matched': substitute(sparse, QKNorm, LearnedScaledDot(initial_scale=matched_gain(model))),
    }


def observed(run, size, task):
    """The original basis recipes, thread counts and fixed 256-row attention observer."""
    run = swap(run, 'execution.threads', THREADS.get((size, task), 1))
    return swap(run, 'diagnostics', AttentionDiagnostics(every=run.evaluate.every, examples=256))


experiments = {
    f'{arm}-{mode}-{size}-{task}-seed{seed}': observed(run, size, task)
    for size, model in MODELS.items() for arm, variant in variants(model).items()
    for mode in ('easy', 'hard') for seed in (0, 1, 2)
    for task, run in basis(variant, mode, seed).items()
    if (mode, task) != ('hard', 'parity')
}
