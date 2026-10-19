# lean-optim

Standalone PyTorch optimizers extracted from the Transformer formalization
lab. This directory builds as its own wheel and source distribution. Runtime
dependencies: PyTorch and Python's standard library. No Lean installation,
`lab`, neighboring repository, paper download or experiment files are needed.

```bash
pip install ./optimizers
# Or copy this directory anywhere, then: pip install .
```

```python
import torch
from lean_optim import AMSGradW, Guarded, Inverse, InverseSqrt, Muon

model = torch.nn.Linear(8, 4)
optimizer = AMSGradW(
    model.parameters(), lr=1e-3, weight_decay=0.01, beta1_decay=Inverse(), lr_decay=InverseSqrt()
)
optimizer.zero_grad()
model(torch.randn(2, 8)).square().mean().backward()
optimizer.step()
```

Available rules: SGD, native AdamW, AdamWDirection (for stages), raw-moment
Adam, AMSGrad, AMSGradW, AdamX, AdaGrad, AdamNC, RMSProp, Muon, Dash (blocked
Shampoo), AdaFisher, AMSGradMD, AtomicColumns, ScheduleFreeSGD,
ScheduleFreeAdamW and ClippedOptimistic. Guarded, Magma and Clipped compose
with the rules described below. The separate
[correspondence audit](src/lean_optim/LEAN_CORRESPONDENCE.md) maps each rule to Lean and
records numerical extensions and missing formal coverage. This is a manual
recurrence audit with numerical tests, not a formal verification of PyTorch
or IEEE floating-point execution.

`Adam` deliberately has **no bias correction**. `AdamW` is PyTorch's native
bias-corrected implementation. AdaGrad uses the mean squared gradient and
the configured step decay; its epsilon placement differs from native Adagrad.

Choose native PyTorch behavior through `standard`, which is the actual
`torch.optim` module, including every optimizer available in the installed
Torch version. These are identical class objects, with their original
constructors, defaults, state dictionaries and kernels:

```python
from lean_optim import standard, PaperAdam, PaperAMSGrad, PaperAdaGrad

native = standard.Adam(model.parameters(), lr=1e-3)
native_amsgrad = standard.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
paper_adam = PaperAdam(model.parameters(), lr=1e-3)  # raw moments, no debiasing
```

`PaperAdam`, `PaperAMSGrad` and `PaperAdaGrad` are explicit aliases for the
paper conventions (`Adam`, `AMSGrad`, `AdaGrad`). Guarded and the checked
AMSGradMD variant add the stated descent/fallback corrections. The latter
requires `guard_sigma <= 0.5`, including at a halved fallback. Muon, Dash,
AdaFisher, Magma and the Schedule-Free rules implement their cited methods;
their presence here is not a claim that this package invented them. The
audit records the AMSGradW schedule correction as well.

Joint-direction methods take a module or one iterable of tensors. They
support one parameter group, closures, learning-rate schedulers and ordinary
`state_dict()` checkpoints. Native AdamW also supports multiple groups.
`rate=device_tensor` selects the historical capturable arithmetic for
direction rules; warm up all buffers before capture. Dash with EVD cannot be
captured. Schedule-Free uses eager execution. Moment hyperparameters
are stored in the parameter group; schedule/solver settings use plain dicts
in state dictionaries. Recreate the same matrix selection and stages before
loading a checkpoint. Clipping/stage settings must also match construction.

```python
# Select matrices explicitly to exclude embeddings and readouts.
optimizer = Muon(model.parameters(), lr=0.02, matrix_params=[model.weight])
optimizer = Guarded(optimizer, sigma=0.5)

from lean_optim import AdamWDirection, Clipped, Magma

optimizer = Clipped(
    Magma(AdamWDirection(model, lr=1e-3), updates=1000, seed=7, matrix_params=[model.weight]),
    norm=1.0,
)
```

Muon and AMSGradMD apply their matrix rule to all 2D tensors by default;
other parameters use the documented auxiliary rule. AdaFisher takes a module
with bias-free Linear layers so it can collect input/output-gradient
factors; call `close()` when replacing it. AtomicColumns uses an explicit matching-family model, task
and head-factory protocol; [its module](src/lean_optim/atomic.py) documents
the required physical head operations. Its numerical solver and pricing
logic live here; the lab only supplies its matching model and data adapter.

For Schedule-Free methods, `optimizer.eval()` installs the averaged
evaluation weights; `optimizer.train()` restores the query weights before
the next training forward. Save the model and optimizer in the same mode.
ScheduleFreeSGD includes the initial point in its average. ScheduleFreeAdamW
implements the paper's Algorithm 1, with raw-second-moment epsilon placement,
decay at the query and squared effective-rate averaging. ClippedOptimistic
requires zero-initialized tensors and uses one joint radius ball; it chooses
its own learning rate and takes no external scheduler.

Run all package and lab checks from the repository root with `./run.py test`.
`./run.py optimizer-parity` checks the rational cases in Lean's kernel.
`./run.py package` builds an isolated copy, installs the wheel away from the
checkout and runs its tests with lab imports blocked.
After installing this directory on its own, its standard-library test suite
runs with `python -m unittest discover -s tests`. The tests include frozen
Lean-checked vectors when a rational specialization is available, direct
independent recurrence calculations and checkpoint continuation checks.

Development checks use Ruff 0.17.1 with `select = ["ALL"]` and `preview = false`.
Run `./run.py optimizer-lint` from the repository root. Both lint and formatting
are mandatory in `./run.py verify`. The only global exceptions select compatible
docstring conventions and let the formatter own trailing commas. Unittest
assertions and a few established public signatures have documented, narrow
exceptions; annotation, correctness, security and complexity rules remain enabled.

Updates reuse work tensors after warmup. Scratch is separate from `state_dict`
and discarded on checkpoint load; replacing gradients does not retain their
old storage. Matrix roots share work by shape, and their internal return buffers
are consumed before reuse. Fisher keeps a small independent input-factor vector
for each retained forward, preserving gradient accumulation and repeated backward.

`./run.py optimizer-profile --device cuda` records raw allocator events after
warmup. `--revision 0a70e33d` measures the original implementation without changing
the checkout. `--scalene` adds CPU/native memory sampling. Reports and Chrome
traces go to the ignored `optimizers/profiles/` directory.

On the measured float32 CUDA workloads, 22 custom variants make **zero new
device allocations per warmed step**, including coordinate rules, iterative
DASH, Muon, guarded/masked variants and Schedule-Free methods. This concerns
optimizer steps with fixed parameter shapes. EVD, native clipping, ordinary
training activations/backward and AtomicColumns still allocate. Persistent
scratch also consumes memory: see the measured sizes, backend limits and full
before/after data in [the allocation audit](benchmarks/README.md).
