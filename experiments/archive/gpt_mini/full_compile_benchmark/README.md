# Twenty-four optimizers under validation patience, the whole step compiled

The protocol of the [patience benchmark](../patience_benchmark) with the
entire training step in one Inductor graph (`fullgraph=True`): input
windows, the functional GPTMini forward, cross entropy, reverse-mode
gradients and every optimizer's state and update, Magma's masks and the
guards included; evaluation is compiled too. Contracts compared every
optimizer's full step with its eager one before training.

All 144 runs completed (24 methods, softmax and sparsemax attention, seeds
0, 1 and 2). AMSGrad had the lowest mean test cross entropy of the best
validation model under both attentions: 1.62787 ± 0.00075 under softmax
and 1.64600 ± 0.01707 under sparsemax.

- [`PLAN.md`](PLAN.md): the plan and its checks.
- [`results/preflight/`](results/preflight): the contracts and the cost of
  compilation, measured before training.
- [`results/rtx3050_all/REPORT.md`](results/rtx3050_all/REPORT.md): both
  rankings, with the runs, curves and provenance beside it.

The [AMSGrad extensions](../amsgrad_extensions_benchmark) add three methods
to this protocol. The lab trains its recipes as
[`shakespeare_zoo`](../../../shakespeare_zoo).
