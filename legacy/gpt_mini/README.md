# GPTMini and TinyShakespeare optimizer benchmarks

An independent uv package with the reference GPTMini model, corrected
AMSGradW, and all 27 optimizer recipes from the completed comparisons.
Python 3.14 and PyTorch 2.14 match the measured experiment.

## Architecture

```text
src/
  domain/          model settings, paired job planning, stopping, ranking
  application/    benchmark orchestration and execution/storage ports
  infrastructure/ PyTorch model, optimizers, CUDA kernels, dataset, evidence, I/O
  interfaces/      CLI and composition of the concrete adapters
tests/             numerical, application, architecture, and migration contracts
```

The domain and application import neither PyTorch nor filesystem services.
The application owns planning, preflight, resume, execution, and validation;
execution and persistence are injected through `application/ports.py`.
The CLI connects `CudaTraining` and `FilesystemResults` to `RunBenchmark`.
Architecture tests check the dependency direction and import the inner layers
without loading PyTorch or NumPy.

## Install and test

Run from this directory, `legacy/gpt_mini/`:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
OPTIMIZER_BENCH_CUDA=1 uv run --locked python -m unittest discover -s tests -v
```

The first suite checks CPU behavior; the second includes actual NVIDIA GPU
execution. Contracts compare compiled and eager losses, gradients, weights,
optimizer histories, Sparsemax projection/Jacobians, real CUDA Graph capture,
stable tracing, stopping, and checkpoint restoration. Application tests use a
fake execution port to check resume and corruption handling independently.
Additional GPU tests exercise both execution backends on real TinyShakespeare.

Six complete fresh AMSGradW runs after migration exactly reproduced the archived
training/validation curves, stopping steps, selected weights, and test losses.
The mean test CE over seeds 0, 1, 2 is 1.625374713347326 for Softmax and
1.6389354029880645 for Sparsemax. See the
[reproduction report](reports/amsgradw_reproduction_20261002/REPRODUCTION.md)
for the recorded protocol, raw runs, preflight tests, and comparison checks.

## Run a comparison

```bash
uv run --locked gpt-mini-benchmark --help
uv run --locked gpt-mini-benchmark --tests-only
uv run --locked gpt-mini-benchmark
```

The default comparison runs corrected AMSGradW on Softmax/Sparsemax and seeds
0, 1, 2. Use `--all` to compare all 27 recipes or `--only` to select optimizers.
It retains the archived validation-selected learning rates and the
reference model: two layers, four heads, width 128, FFN 512, context 64, batch 32.
Validation runs every 250 updates; patience 8, min_delta 0.0001, min_steps 1000,
max_steps 20000, and divergence delta 0.1/patience 3 match the previous protocol.
The exact best validation checkpoint is restored before test evaluation.

```bash
uv run --locked gpt-mini-benchmark --all
uv run --locked gpt-mini-benchmark --only amsgradw amsgradmd amsgradmd_guarded
uv run --locked gpt-mini-benchmark --output results/another_comparison
uv run --locked gpt-mini-benchmark --output results/another_comparison --validate-only
```

Every training launch passes CPU/CUDA numerical contracts first. Runs resume
from matching recorded protocols. Changed code/settings and duplicate or
unpaired run records are rejected. Usable results enter the report with their
completion status; optimizer proof status does not filter empirical results.

Outputs default to `python/results/combined/` in a checkout. For an installed
wheel they default to `./gpt-mini-benchmark/results/combined/`.
`GPT_MINI_BENCHMARK_HOME` overrides the writable root. Metadata, raw runs,
summary, report, preflight, checkpoint, and validation artifacts stay together.

## Data and historical evidence

The complete dataset is `src/infrastructure/benchmark/data/tinyshakespeare.txt`.
Its SHA256 is `53493bf304b639aba6a47400da75c58ce32372b378fd2cd9d16ee987aeb12e4e`.
The original character vocabulary and 90/5/5 splits remain unchanged.

All six historical benchmark families, their reports, protocols, preflight
logs, and measured source snapshots are preserved under
`src/infrastructure/benchmark/archive/`. `migration.json` records the hashes
before the transfer. Original source/report bytes and protocol fingerprints
are preserved; active code records a separate source fingerprint. Original
`experiments/` paths remain compatibility links to the transferred files.
Previous saved models and curves are now in `python/runs/`; these large run
artifacts are ignored by git and excluded from distributions.

The numerical kernels and original stopping rules are checked against the
archived syntax trees. Frozen float64 AMSGradW outputs additionally check each
update independently of importing the historical implementation.

The deterministic Lean convergence results have additional hypotheses. These
are not certified for stochastic nonconvex GPTMini; the benchmark is empirical.

## Public model and optimizer API

```python
from gpt_mini import AMSGradW, Config, GPTMini

model = GPTMini(Config(vocab_size=65, n_layers=2, n_heads=4,
                       d_model=128, d_ff=512, max_seq_len=64))
optimizer = AMSGradW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

AMSGradW retains raw moments, the running maximum of second moments, and
decoupled weight decay. Its defaults match the completed comparison's winner.

`uv build` creates a wheel and source distribution containing the model,
optimizers, benchmark adapters, test suites, dataset, and preserved evidence.
