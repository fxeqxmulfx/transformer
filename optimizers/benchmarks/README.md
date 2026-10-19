# Optimizer lint and allocation audit

Measured on 2026-10-19 with Python 3.14.8, PyTorch 2.14.1+cu126,
CUDA 12.6 and an NVIDIA GeForce RTX 3050 Laptop GPU. CPU workloads use one
thread. The baseline is commit `0a70e33d`. The exact source digests and all
case measurements are in [allocation_results.json](allocation_results.json).

## Stable Ruff

The standalone package pins Ruff 0.17.1 and enables `ALL`, with preview
explicitly disabled at the user's request. `./run.py optimizer-lint` checks
source, tests, benchmark drivers and formatting; `./run.py verify` includes it.
The package's development dependency does not enter wheel runtime requirements.

Global exclusions are D203/D213, in favor of D211/D212, and COM812, whose
trailing commas are handled by the formatter. Tests retain standard-library
unittest assertions (PT009/PT027). Function-local compatibility exceptions
preserve established optimizer/recurrence signatures, capitalized composition
constructors and required interface arguments. No annotation or security rule
family is disabled. Public APIs, internal numerical helpers and tests have
explicit annotations and docstrings.

## What changed

Coordinate denominators, scheduled coefficients, bias corrections, direction
vectors, guard reductions and MAGMA masks reuse buffers. Plain SGD reads its
gradient directly. Schedule-Free mixes and optimistic projections write into
existing tensors. Muon, DASH and AMSGradMD reuse buffers with disjoint lifetimes;
matrix products never overwrite an input they still need. DASH root workspaces
are shared by shape after the previous result has been consumed.

Decoded schedule objects are cached until the serialized configuration changes,
including in-place changes. Scratch remains outside mathematical optimizer state
and checkpoints. Loading state clears it. Cache keys never retain replaced
gradients. Fisher reuses full activation/derivative square buffers, keeps each
forward's small input-factor vector independent, and preserves dtype promotion
when accumulated forwards cross autocast contexts.

Arithmetic order is retained, including reciprocal-then-multiply for Python
scalar reverse division. The Chebyshev historical training fixture caught this
distinction; its exact trajectory passes after correction. Tests also compare
buffered roots to the original float32/64 recurrences bit for bit.

## Steady-step measurements

Most fixed-gradient cases use a 64×96 matrix and a 64-vector: 24,832 bytes of
float32 parameters. Fisher uses a bias-free 96→64 linear layer. Atomic cases use
the physical two-token matching family in float64. Five warmup steps precede
five measured steps. Inputs and gradients are allocated outside measurement.

Counts come from positive `Bytes` in raw PyTorch Profiler `[memory]` events.
Summed operator memory can conceal an allocation freed inside the same operator;
changes in CUDA reserved memory conceal caching-allocator traffic. Neither is
used as a substitute for allocator events. CUDA rounds these allocations, so
device byte counts differ from logical tensor sizes.

| CUDA case | Allocated bytes/step before | After | Persistent scratch bytes after |
| --- | ---: | ---: | ---: |
| SGD | 25,088 | 0 | 0 |
| Adam | 75,264 | 0 | 24,832 |
| AMSGradW | 125,440 | 0 | 49,664 |
| AdamX | 139,776 | 0 | 49,760 |
| Muon | 909,824 | 0 | 107,053 |
| DASH NewtonDB | 2,380,288 | 0 | 399,328 |
| DASH CoupledNewton | 1,981,952 | 0 | 432,104 |
| DASH Chebyshev | 2,167,808 | 0 | 383,832 |
| Guarded AMSGradMD | 865,792 | 0 | 85,125 |
| ScheduleFreeSGD | 75,264 | 0 | 24,832 |
| ScheduleFreeAdamW | 225,792 | 0 | 49,664 |
| ClippedOptimistic | 318,976 | 0 | 74,568 |
| AdaFisher optimizer step | 109,056 | 0 | 50,482 |

In total, 22 custom fixed-gradient CUDA variants have zero positive allocation
events after warmup. Parameter gradients are left intact. CUDA graph replay
tests cover live gradients/rates and evolving state in coordinate methods,
Muon, all iterative DASH solvers, AMSGradMD, guards and MAGMA.

Persistent scratch is a real memory cost. For example, the original NDB case
had 329,216 bytes of peak transient CUDA storage; the reused workspace has
399,328 logical bytes and zero transient peak. The measured CN and Chebyshev
workspaces also exceed their old transient peaks. This optimization removes
allocator churn and does **not** promise a lower total VRAM peak for every
method. Source shapes/dtypes or Fisher batch shapes changing can allocate new
buffers. The table excludes parameters, gradients and mathematical state;
the JSON records mathematical state separately. CUDA allocator padding also
adds to the logical workspace size.

## Remaining allocations

- CPU eager execution still allocates tiny scalar wrappers inside PyTorch:
  Adam uses 72 bytes/step, NDB 960 bytes/step in this workload. Python dispatch
  also creates temporary objects. Zero CUDA tensor traffic is not a claim of
  allocation-free Python execution.
- EVD still allocates inside LAPACK/cuSOLVER despite preallocated eigenvalue
  and eigenvector outputs. CUDA traffic falls from 3,175,936 to 2,625,024
  bytes/step; CPU traffic falls from 619,276 to 86,492 bytes/step. The remaining
  backend work is visible in raw traces. Iterative solvers avoid this traffic
  but retain their documented finite-iteration approximation semantics.
- The Clipped wrapper calls native `torch.nn.utils.clip_grad_norm_`, retaining
  its norm, epsilon and nonfinite behavior. Clipped Adam still makes nine small
  CUDA allocations, 4,608 bytes/step. Native torch optimizers remain unchanged;
  the measured native AdamW allocates 25,088 CUDA bytes/step.
- A full Fisher forward/backward/update cycle falls from 204,288 to 75,264
  CUDA bytes/step. The otherwise identical SGD training control uses 74,752.
  The remaining 512-byte difference is the input-factor vector whose lifetime
  must cover its forward's backward, including retained/repeated graphs.
  Ordinary activations and autograd allocations remain necessary.
- AtomicColumns performs dynamic pricing, physical head construction,
  autograd search and simplex correction. Its audit is included; these
  allocations are unchanged. The scalar CPU order case uses 10,132 bytes/step;
  the small CPU search averages 12,820.6, CUDA search 365,260.8. These toy
  measurements do not predict the cost of larger matching families.

Scalene 2.3.0 ran with explicit `--memory`, a 65,536-byte sampling window and
GPU sampling disabled. It attributes the old hot allocations to root iterations,
grafting, magnitude proposals and masking; the final profile highlights EVD
and AtomicColumns. Both runs include initialization and a fixed-duration loop,
so sampled totals are not used as equal-step allocation counts. In particular,
the large first optimizer construction includes PyTorch's lazy imports.

## Reproduction and checks

```sh
./run.py optimizer-lint
./run.py optimizer-profile --revision 0a70e33d --output optimizers/profiles/before-cpu.json
./run.py optimizer-profile --output optimizers/profiles/after-cpu.json
./run.py optimizer-profile --device cuda --revision 0a70e33d --output optimizers/profiles/before-cuda.json
./run.py optimizer-profile --device cuda --output optimizers/profiles/after-cuda.json
./run.py optimizer-profile --scalene --revision 0a70e33d --output optimizers/profiles/scalene-before.json
./run.py optimizer-profile --scalene --output optimizers/profiles/scalene-after.json
./run.py test test_allocations test_matrix test_cuda
./run.py verify
./run.py package
```

After installing the standalone package, the driver also runs directly with
`python benchmarks/allocations.py --device cuda --output profiles/allocations.json`.
The Git revision option is a repository-runner convenience.

Revision profiling reads a temporary Git archive; it does not switch the
checkout or add local-path runtime dependencies. JSON, raw Chrome traces
and Scalene profiles are generated under ignored `optimizers/profiles/`.
Raw profiles include no throughput claim. Regression tests check allocation
budgets, cache lifetimes, checkpoint continuation, dtype promotion and
numerical recurrences; the complete lab suite retains its frozen historical
trajectories. The isolated wheel runs the same standalone suite with lab
imports blocked. Lean recurrence parity and the existing axiom audit are
retained; no Lean statement or unfinished proof is changed.

Validation completed: `verify` passed 350 lab tests (three existing skips) and
36 standalone tests. The final `package` check passed all 36 tests taken from
the source archive against the isolated wheel. Both distributions include the
source-independent numerical implementation; the source archive also includes
this audit, its data, the benchmark drivers and the test suite.
