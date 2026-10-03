# Twenty-four optimizers under validation patience

The 24 methods of the [Magma benchmark](../magma_benchmark), with its model,
data, initializations, batches and selected rates, trained until a shared
validation rule stops them instead of for a fixed 1000 updates. The full
validation split is evaluated every 250 updates; from update 1000 on, a run
stops after 8 evaluations without an improvement of 1e-4, or after 3 at
least 0.1 above its best, and at the latest after 20,000 updates. The best
validation model is restored and tested once.

The run was stopped after 1 of its 144 runs, eager, when CUDA compilation
was requested: the [compiled benchmark](../compiled_benchmark) continued
the protocol.

- [`PLAN.md`](PLAN.md): the stopping protocol.
- [`results/rtx3050_all/REPORT.md`](results/rtx3050_all/REPORT.md): the one
  completed run, its curves and provenance beside it.
