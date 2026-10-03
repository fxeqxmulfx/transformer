# Magma: twenty-four optimizers at a fixed budget

The [optimizer benchmark](../optimizer_benchmark) again, with six methods
more: RMSProp, and Magma (arXiv:2602.15322v1, Algorithm 1 and Sections 3–4)
over RMSProp, Adam, AdamW, Muon and SGD. Model, data, initializations,
batches and protocol are frozen from that benchmark, whose 18 measured
methods enter the ranking unchanged: each new method is screened at three
rates for 250 updates on seed 0 and trained at the selected one for 1000
updates on seeds 0, 1 and 2, under softmax and under sparsemax attention.
It is a new experiment on this model, not a reproduction of the paper's
C4/Llama schedule.

Magma over Muon had the lowest mean test cross entropy of all 24 under both
attentions: 1.75293 ± 0.00365 under softmax and 1.78222 ± 0.00687 under
sparsemax.

- [`PLAN.md`](PLAN.md): the protocol, declared before training.
- [`results/rtx3050/REPORT.md`](results/rtx3050/REPORT.md): the combined
  rankings; the new runs, calibration, audits and provenance beside it.

The [patience benchmark](../patience_benchmark) trains the same 24 under a
validation stopping rule.
