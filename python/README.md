# lab

Transformer experiments composed from swappable blocks. An experiment is a
folder under `experiments/`: its `experiment.py` writes, in the language of
`lab.dsl`, a model and the conditions of its benchmark runs, and its
`README.md` says what it asks and what its runs found. Variants are new values
derived from a base by replacing blocks; nothing is configured on the command
line.

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

## Context extension with a derived temperature

```python
# Copy training length, RoPE base and pair layout from an existing model spec.
extended = swap(base, "model", extend_context(base.model, 16_384))
```

`extend_context` installs `YaRN` and retains the original training context
across subsequent extensions. Only the new target length is supplied.
`YaRN(original_context=4096)` is also available when that metadata comes
from an external pretrained configuration. Model parameters and state-dict
keys retain their names, so the original model weights load with `strict=True`.
The lab's run-continuation checks still require the original experiment spec;
context extension is a new model configuration, not a continuation bypass.

This is a **derived variant** of YaRN: the frequency ramp follows
arXiv:2309.00071v3, Section 3.2, Eqs. (10)-(13), with the paper's LLaMA
band defaults `alpha=1`, `beta=32` and fixed configured-context scaling.
It replaces the paper's empirical temperature fit with the formula proved
in `Transformer.RoPE.YaRN.Calibration`, commit `4f4bbc61`:

```text
N = original_training_context - 1
M = number_of_visible_keys - 1
Delta = largest_visible_logit - second_largest_visible_logit
g = 1 + log(M / N) / Delta            # when M > N and the gap is resolved
adjusted_logits = g * (logits - largest_visible_logit)
```

The measured gap includes rotation and ordinary score normalization, so it
also includes QKNorm's or LearnedScaledDot's learned head scale. Each head,
sample and causal query has its own gain. There is no manual gap, temperature
or extension factor. Multiplying logits by `g` has the same softmax effect
as multiplying both Q and K by `sqrt(g)`; it is applied only once.

For one winner and equal losing scores this preserves its reference
probability `1/(1+N*exp(-Delta))`. For arbitrary losing scores it preserves
that lower bound, not an actual old model's probability. The reference uses
the current measured gap; a change in gap caused by rescaling or earlier
layers is not inferred from the old model. The current winner need not be
the desired token, so this is not a retrieval or perplexity guarantee.

Implementation choices beyond the real-arithmetic proof: rows no longer
than the training context use gain one. Ties and gaps no larger than
`machine_epsilon * max(1, visible_score_span)` also use gain one, without a
retention certificate. Other rows differentiate through the measured gap.
Half and bfloat16 logits are calibrated in float32, with probabilities cast
to the value dtype for the weighted sum. Future keys never enter calibration.
The original frequency tables and attention path are recovered exactly
when the configured target is no longer than the original context.

The variant requires `Softmax`. Extended models materialize scores and use
explicit attention even with `Softmax(fused=True)`, so their memory and runtime
differ from fused attention. It supports both projection layouts, both rotary
pair layouts, and ScaledDot, LearnedScaledDot and QKNorm scores.

## Commands

Run from the repository root through `run.py`, or from here with `uv run`:

```bash
uv sync --locked
uv run --locked lab blocks
uv run --locked lab check ../experiments/<name>
uv run --locked lab show ../experiments/<name> <label>
uv run --locked lab run ../experiments/<name> [label ...]
uv run --locked lab report ../experiments/<name> [label ...]
uv run --locked python -m unittest discover -s tests
```

`lab run` trains every run of the experiment, or the labeled ones, into
`runs/<label>/` in its folder:

```
experiment.json    the description, and one segment per training session
experiment.py      the experiment file as of the latest session
history.jsonl      canonical observations; probes.jsonl, their neighbors
diagnostics.jsonl  sampled per-tensor measurements
gradients.jsonl    the gradient norm of every update
checkpoint.pt      model, optimizer and sampler state at the last checkpoint
result.json        the summary, written when the budget is reached
```

Running the experiment again continues each unfinished run from its checkpoint
and skips finished ones. A run continues only the same experiment (raising
`budget.updates` extends it) under the same engine: lab sources, PyTorch
version and device, as recorded in each segment.

`lab report` prints, as JSON, each chosen run's state (not started,
unfinished, finished or stopped), its latest observation and its result. A
modular division run also gets the analyses of the historical stability
protocols, read from its records: its phases, whether its success
persisted, its failures and recoveries after generalizing, the probes and
diagnostics around each held-out failure, and its largest gradients.
Experiments that differ in `optimizer.lr` alone form a group; once each of
its runs has trained the budget its experiment sets, the report names the run
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
The optimizer specifications remain in the domain. Their PyTorch arithmetic
lives in the independent [`lean-optim`](../optimizers/README.md) package;
`infrastructure/optim/` translates specifications and matching model/task
protocols into its public API. Population search remains local to the lab.
Trajectory provenance includes the optimizer sources as well as lab sources.

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
| `optimizer_benchmark.*`, `full_compile_benchmark.*`, `magma_benchmark.*`, `amsgrad_extensions_benchmark.*` | `legacy/gpt_mini/src/infrastructure/benchmark/` (as measured: its subfolder `archive/experiments/`) | `2aec5b9` |
| the synthetic trainers' modules, `paper_reproduction.*` | `experiments/synthetic_trainers/` | `5d64147` |
| the convex MQAR comparison's modules | `experiments/convex_mqar/src/convex_mqar/` | `9416d03` |

Regression tests use the exact historical source snapshots checked in under
`tests/references/sources/`, with pinned revisions and SHA256 hashes. They do
not need those Git objects or another local repository. The optimizer
benchmark's measured source files are also retained, byte for byte, under
`scripts/fixtures/optimizer_benchmark/`. Old absolute paths in archived
metadata identify those files; checks resolve them to these local snapshots.
`../run.py references` verifies both sets; see
[`tests/fixtures/README.md`](tests/fixtures/README.md) for replay and maintenance.

Their records are kept in [`experiments/archive/`](../experiments/archive),
a folder per codebase: the reference `gpt_mini.py`, which the Lean
formalization and the lab cite as GPTMini, with the plans and results of its
Tiny Shakespeare benchmarks; the plans, protocols and baselines of the
synthetic trainers; and the convex MQAR reports.

Not ported: the complementary-attention study of the synthetic trainers,
and the reports, plots, layouts and integrity checks of their protocols;
the sweep summaries; the convex construction of the MQAR comparison and its
certificate, its Zoology models and diagnostics, and its BF16 and fused
AdamW training; TorchInductor compilation, wherever it was used (the lab
accelerates with CUDA graphs).
