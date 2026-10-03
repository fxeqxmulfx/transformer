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
uv run --locked lab run ../experiments/<name>.py [label ...]
uv run --locked lab report ../experiments/<name>.py [label ...]
uv run --locked python -m unittest discover -s tests
```

`lab run` trains every experiment of the file, or the labeled ones, into
`runs/<file stem>/<label>/` beside the file:

```
experiment.json    the description, and one segment per training session
experiment.py      the experiment file as of the latest session
history.jsonl      canonical observations; probes.jsonl, their neighbors
diagnostics.jsonl  sampled per-tensor measurements
gradients.jsonl    the gradient norm of every update
checkpoint.pt      model, optimizer and sampler state at the last checkpoint
result.json        the summary, written when the budget is reached
```

Running the file again continues each unfinished run from its checkpoint and
skips finished ones. A run continues only the same experiment (raising
`budget.updates` extends it) under the same engine: lab sources, PyTorch
version and device, as recorded in each segment.

`lab report` prints, as JSON, each chosen run's state (not started,
unfinished, finished or stopped), its latest observation and its result. A
modular division run also gets the analyses of the historical stability
protocols, read from its records: its phases, whether its success
persisted, its failures and recoveries after generalizing, the probes and
diagnostics around each held-out failure, and its largest gradients.
Experiments that differ in `optimizer.lr` alone form a group; once each of
its runs has trained the budget its file sets, the report names the run
each policy of the convex MQAR comparison selects on the selection split:
the best observation (`best`), the earliest observation at 99% accuracy
(`first99`), and the earliest one that held to the end (`stable99`).

`Eager()` issues every update kernel by kernel with the native optimizer and
reproduces the historical modular trainer record for record
(`tests/test_engine.py`). `CudaGraph()` captures one update per recurring
batch size and one evaluation per split, and replays them; sampled updates
run the same operations eagerly. The optimizer runs in its capturable form,
which reads the rate from a device tensor and rounds differently from the
native form in the last bits, so a CudaGraph run equals its own operations
issued eagerly (`tests/test_graphs.py`), not an Eager run. On the GTX 1050 it trains the mod-97 models 1.4-2.2 times faster.

## Layout

`src/lab/` is layered by dependency; inner layers never import outer ones.

```
domain/          the language (spec, model, optimizers, training, benchmarks,
                 stopping, experiment)
                 and pure rules (cadence, analysis); standard library only
application/     use cases over ports; no PyTorch
infrastructure/  PyTorch builders and engines, data, storage
interfaces/      the command line
dsl.py           the words an experiment file imports
```

## Ports

The lab replaces three earlier codebases, removed from the tree once what
the experiments need was ported and checked against their recorded outputs
(`tests/fixtures/legacy_*.json`). Docstrings cite them by module or path;
read a cited source with `git show <commit>:<path>`:

| Cited as | Path | Last in |
| --- | --- | --- |
| `gpt_mini.*` | `legacy/gpt_mini/src/` | `2aec5b9` |
| `optimizer_benchmark.*`, `full_compile_benchmark.*`, `magma_benchmark.*`, `amsgrad_extensions_benchmark.*` | `legacy/gpt_mini/src/infrastructure/benchmark/` (as measured: its `archive/experiments/`) | `2aec5b9` |
| the synthetic trainers' modules, `paper_reproduction.*` | `experiments/synthetic_trainers/` | `5d64147` |
| the convex MQAR comparison's modules | `experiments/convex_mqar/src/convex_mqar/` | `9416d03` |

Their records stay under `experiments/`: the plans and results of the
TinyShakespeare benchmarks (`*_benchmark/`), the plans, protocols and
baselines of the synthetic trainers, and the convex MQAR reports.
`experiments/gpt_mini.py` stays as well; the Lean formalization and the lab
cite it as the reference GPTMini.

Not ported: the complementary-attention study of the synthetic trainers,
and the reports, plots, layouts and integrity checks of their protocols;
the sweep summaries; the convex construction of the MQAR comparison and its
certificate, its Zoology models and diagnostics, and its BF16 and fused
AdamW training; TorchInductor compilation, wherever it was used (the lab
accelerates with CUDA graphs).
