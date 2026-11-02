# Original GPTMini under Schedule-Free AdamW

Task162 asks whether the previously successful GPTMini still solves hard Basis
recall with the optimizer required for the current GPT comparison. Historical
AdamW successes cannot separate optimizer effects from architecture changes.

| Quantity | Both new GPTMini controls |
| --- | --- |
| Model | Historical width128, depth6, four heads; learned QK temperature, PreNorm, ReLU squared, RoPE, XSA, tied readout |
| Seeds | Model0/1, data1, batch=model+10,000 |
| Problem | Original hard recall:256 symbols,16 writes including8 overwrites,8 queries |
| Data | Original20,000/512/512 rows and exact fingerprints |
| Optimizer | Schedule-Free AdamW, rate0.0003, betas(.9,.98), decay.1 |
| Schedule | Original50-update warmup, then constant rate |
| Batches | Training64; evaluation64; observe every50 updates |
| Runtime | Python3.15 / Torch2.15, FP32 CUDA fullgraph max-autotune, math SDPA |
| Stop | First99% exact validation sequences; initial100,000-update continuation boundary |

The model definition is unchanged from `experiments/basis/experiment.py` at
bd63e509. The original supervised-position readout is retained. `BasisControl`
changes only the stopping/reporting contract: retain the current complete state,
without best-state restore, and keep final test sealed until learning closes.
The foreach Schedule-Free implementation is the same as in task160.
The historical control has1,249,816 parameters; the current canonical model
has1,249,024 at the same width/depth/head count. This is a whole-architecture
comparison at nearly the same parameter count, not a larger-model intervention.

The matched current-model controls are already measured in
[`basis_residual`](../basis_residual): seed0 is censored at100,000 updates;
seed1 reaches first99% at123,550. Their data, batch order convention, rates,
warmup and sequence criterion are fixed references, not new runs or model
selection from final test. Initial model weights differ because the architectures
and initializers differ. Wall-time comparisons are descriptive, not causal.

Before learning, freeze sources, runtime, model descriptions, parameter counts,
full data fingerprints and reference-result hashes. Resuming preserves the full
optimizer and sampler state. At a budget boundary, inspect validation and extend
promising runs explicitly; do not call a finite miss impossible. Every run and
continuation remains in the report. Final test never chooses a hyperparameter.

No architecture repair is inferred from this comparison alone. If GPTMini
solves faster under the same optimizer, the next cohort changes one component
of current GPT at a time. Learned QK temperature, residual normalization,
epsilon, activation and initialization remain separate hypotheses. No production
flag or model replacement is added by this control study.

Use `./run.py check experiments/basis_control`, then
`./run.py basis-control freeze`, `run`, and `report`. The run command holds
an exclusive cohort lock. After an interruption it retains the prior logs
and saved state before resuming; after a completed chunk it requires an
explicit continuation decision. Disclosed final tests prevent further learning.

Results are pending.
