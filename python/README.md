# lab

Transformer experiments composed from swappable blocks. An experiment is a
file under `experiments/` that writes, in the language of `lab.dsl`, a model
and the conditions of its benchmark run. Variants are new values derived from
a base by replacing blocks; nothing is configured on the command line.

```python
from lab.dsl import *

model = Transformer(
    width=128, depth=2, context=50,
    block=Block(
        attention=Attention(heads=4, projections=FusedQKV(), scores=QKNorm(),
                            weights=Softmax(), exclusive=XSA()),
        ffn=FFN(activation=ReLU2()), norm=RMSNorm(), residual=PreNorm()),
    positions=RoPE(), readout=Tied(), final_norm=RMSNorm(), init=Normal(0.02))

base = Experiment(
    model=model,
    benchmark=ModularDivision(prime=97, train_fraction=0.2),
    optimizer=AdamW(lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0),
    schedule=Schedule(warmup=10),
    budget=Budget(updates=150_000, batch=512),
    seeds=Seeds(model=0, data=0),
    evaluate=Evaluate(every=250),
    execution=CudaGraph())

experiments = {
    "softmax": base,
    "sparsemax": substitute(base, Softmax, Sparsemax()),
}
```

`swap(base, "model.block.attention.weights", Sparsemax())` replaces one field,
`substitute` every occurrence of a block, and `grid` crosses labeled
alternatives on several paths. `lab blocks` lists the whole language.

## Commands

Run from the repository root through `make.py`, or from here with `uv run`:

```bash
uv sync --locked
uv run --locked lab blocks
uv run --locked lab check ../experiments/<name>.py
uv run --locked lab show ../experiments/<name>.py <label>
uv run --locked python -m unittest discover -s tests
```

## Layout

`src/lab/` is layered by dependency; inner layers never import outer ones.

```
domain/          the language (spec, model, training, benchmarks, experiment)
                 and pure rules (cadence, analysis); standard library only
application/     use cases over ports; no PyTorch
infrastructure/  PyTorch builders and engines, data, storage
interfaces/      the command line
dsl.py           the words an experiment file imports
```
