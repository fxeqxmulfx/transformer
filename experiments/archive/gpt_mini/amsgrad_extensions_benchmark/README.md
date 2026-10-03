# AMSGradW and AMSGradMD under validation patience

Three methods added to the [full-compile benchmark](../full_compile_benchmark),
under its model, data, initializations, stopping and compiled step:
AMSGradW, AMSGrad with decoupled weight decay (`Transformer.AMSGradW`);
AMSGradMD, its magnitude-direction variant
(`Transformer.MagnitudeDirection`); and a descent guard over AMSGradMD.
Each update is the one the Lean recurrence defines
([`LEAN_AUDIT.md`](LEAN_AUDIT.md) says what the proofs cover). Each method
was screened at three rates for 250 updates on seed 0, then trained under
softmax and under sparsemax attention on seeds 0, 1 and 2: 18 runs.

AMSGradW had the lowest mean test cross entropy of all 27 methods under
both attentions: 1.625375 ± 0.001731 under softmax and 1.638935 ± 0.016796
under sparsemax, against AMSGrad's 1.627871 and 1.646002. The guard over
AMSGradMD accepted 0.16% to 0.67% of its proposals.

- [`PLAN.md`](PLAN.md): the algorithms and the protocol.
- [`results/preflight/`](results/preflight): the Lean build and axiom audit
  and the tests, before training.
- [`results/rtx3050/REPORT.md`](results/rtx3050/REPORT.md): the rankings
  with the 24 earlier methods, and the guard's acceptance.
- [`results/reproduction_20261002/REPRODUCTION.md`](results/reproduction_20261002/REPRODUCTION.md):
  the six AMSGradW runs retrained after the codebase was restructured,
  equal to the originals in every curve, stopping decision and selected
  model.

The lab trains the AMSGradW arm as
[`shakespeare_amsgradw`](../../../shakespeare_amsgradw), and every recipe
as [`shakespeare_zoo`](../../../shakespeare_zoo).
