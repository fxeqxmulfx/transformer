"""GPTMini on Tiny Shakespeare under the historical optimizer zoo, each recipe at its selected rate.

`README.md` beside this file describes the runs and what their archived
runs found.
"""

from lab.dsl import *

gptmini = Transformer(
    width=128, depth=2, context=64,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

softmax = Experiment(
    model=gptmini,
    benchmark=TinyShakespeare(window=64),
    optimizer=SGD(lr=0.1),
    schedule=Schedule(),
    budget=Budget(updates=20_000, batch=32),
    seeds=Seeds(model=0, data=0),
    evaluate=Evaluate(every=250, batch=32),
    stopping=EarlyStopping(patience=8, min_delta=1e-4, after=1000, divergence=0.1, divergence_patience=3),
    execution=CudaGraph())

betas = (0.9, 0.999)

# Each recipe, and the rates its validation selected under softmax and under sparsemax.
zoo = {
    "sgd": (lambda lr: SGD(lr=lr), 0.1, 0.3),
    "adagrad": (lambda lr: AdaGrad(lr=lr), 0.03, 0.03),
    "adam": (lambda lr: Adam(lr=lr), 3e-4, 3e-4),
    "adamw": (lambda lr: AdamW(lr=lr, betas=betas, weight_decay=0.01), 1e-3, 1e-3),
    "amsgrad": (lambda lr: AMSGradW(lr=lr, betas=betas, weight_decay=0.0), 3e-4, 3e-4),
    "amsgrad_inverse": (lambda lr: AMSGradW(lr=lr, betas=betas, weight_decay=0.0,
                                            beta1_decay=Inverse(), lr_decay=InverseSqrt()), 3e-3, 3e-3),
    "amsgrad_geometric": (lambda lr: AMSGradW(lr=lr, betas=betas, weight_decay=0.0,
                                              beta1_decay=Geometric(0.99), lr_decay=InverseSqrt()), 3e-3, 3e-3),
    "adamx": (lambda lr: AdamX(lr=lr), 3e-3, 3e-3),
    "adamnc": (lambda lr: AdamNC(lr=lr), 0.01, 0.01),
    "muon": (lambda lr: Muon(lr=lr), 0.03, 0.03),
    "muon_guarded": (lambda lr: Guarded(Muon(lr=lr)), 0.1, 0.3),
    "dash_evd": (lambda lr: Dash(lr=lr, solver=EVD()), 1e-3, 1e-3),
    "dash_ndb": (lambda lr: Dash(lr=lr, solver=NewtonDB()), 1e-3, 1e-3),
    "dash_cn": (lambda lr: Dash(lr=lr, solver=CoupledNewton()), 1e-3, 1e-3),
    "dash_chebyshev": (lambda lr: Dash(lr=lr, solver=Chebyshev()), 1e-3, 1e-3),
    "dash_ndb_guarded": (lambda lr: Guarded(Dash(lr=lr, solver=NewtonDB())), 0.1, 0.3),
    "adafisher": (lambda lr: AdaFisher(lr=lr), 1e-3, 1e-3),
    "adafisherw": (lambda lr: AdaFisher(lr=lr, weight_decay=0.01), 1e-3, 1e-3),
    "rmsprop": (lambda lr: RMSProp(lr=lr), 1e-4, 3e-4),
    "magma_rmsprop": (lambda lr: Magma(RMSProp(lr=lr)), 3e-4, 3e-4),
    "magma_adam": (lambda lr: Magma(Adam(lr=lr)), 9e-4, 9e-4),
    "magma_adamw": (lambda lr: Magma(AdamW(lr=lr, betas=betas, weight_decay=0.01)), 3e-3, 3e-3),
    "magma_muon": (lambda lr: Magma(Muon(lr=lr)), 0.03, 0.03),
    "magma_sgd": (lambda lr: Magma(SGD(lr=lr)), 0.3, 0.3),
    "amsgradw": (lambda lr: AMSGradW(lr=lr, betas=betas, weight_decay=0.01), 3e-4, 3e-4),
    "amsgradmd": (lambda lr: AMSGradMD(lr=lr), 3e-4, 3e-4),
    "amsgradmd_guarded": (lambda lr: Guarded(AMSGradMD(lr=lr, direction_rate=3e-4), sigma=0.25), 0.1, 0.1),
}

# torch cannot capture an eigendecomposition.
eager = {"dash_evd"}

attentions = {"softmax": softmax, "sparsemax": substitute(softmax, Softmax, Sparsemax())}
arms = {}
for index, (attention, experiment) in enumerate(attentions.items()):
    for name, (recipe, *rates) in zoo.items():
        execution = Eager() if name in eager else experiment.execution
        arms[f"{attention}-{name}"] = swap(swap(experiment, "execution", execution), "optimizer", recipe(rates[index]))

experiments = grid(arms, {"seeds.model": {"seed0": 0, "seed1": 1, "seed2": 2}})
