# Twenty-four optimizers under validation patience, the model compiled

The [patience benchmark](../patience_benchmark) with the model compiled by
`torch.compile` (Inductor, `reduce-overhead`, CUDA graphs); the optimizer
steps, guards and stopping stay eager. Tests compared the compiled and
eager losses, gradients and steps, and checked that CUDA graphs were
recorded, before any run.

The run was stopped after 6 of its 144 runs, when compiling the whole
training step was requested: the [full-compile benchmark](../full_compile_benchmark)
completed the protocol.

- [`PLAN.md`](PLAN.md): the plan and its checks.
- [`results/preflight/`](results/preflight): tests, recompilation counts and
  throughput, measured before training.
- [`results/rtx3050_all/REPORT.md`](results/rtx3050_all/REPORT.md): the six
  completed runs.
