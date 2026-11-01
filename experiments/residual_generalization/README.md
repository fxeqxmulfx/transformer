# Does conditioned depth attention improve transformer generalization?

Task145 compares full-transformer learning, after the query-only repair in task150.
All cases preserve the normalized GPT branches, centered XSA and its prefix guard,
exact RoPE, SwiGLU and tied readout. Python3.15, Torch2.15, CUDA max-autotune and
Schedule-Free AdamW only. The new GPU phase has a two-hour ceiling including
profiling, compilation and evaluation. Successful runs stop early.

| Variant | Depth recurrence | Added learned parameters at W128 D4 |
| --- | --- | ---: |
| add | Ordinary residual addition | 0 |
| scaled | Addition with branch coefficient 1/sqrt(8) | 0 |
| attnres | Paper key-only depth attention, zero queries, learned key gain | 2048 |
| qk | Parameter-free query/key RMS, unit-RMS initial query, sqrt(width) divisor | 1024 |
| conditioned | Same Q/K formula with complete-state query coordinate conditioning | 1024 |

Depth attention reads the embedding and every completed branch once, with a
final mixture before the final norm/readout. There is no extra residual addition
or zero source. Keeping the host Peri output norms is a deviation from the paper.
Conditioning metadata is not a learned gain; it includes the finite-epsilon
correction and transforms query, base iterate and second moment without resetting
optimizer clocks. It changes subsequent optimization geometry.

The initial recipe uses W128 D4 H4, batch256, lr0.003, betas(0.9,0.98), decay0.1
and ten warmup updates. The original half-table training split is retained;
the old held-out half is split into disjoint validation and final test sets.
Only train and validation are observed while learning. Stop at the current
endpoint after three consecutive observations at or after update5000 have both
accuracies >=99%. This is a validation rule, not proof of universal correctness.

Primary cases pair add/raw-QK/conditioned-QK across P97/P193 and seeds11/17.
Secondary key-only and branch-scale controls use P97 and both seeds. Data and
batch-order seed11 are fixed. Runtime allocation will be frozen after diagnostic
profiling, before any learning run; profiling copies are discarded. All failures
and budget-censored trajectories will be retained.

Implementation validation: GPT commit ad76812 has 170 affected CPU tests passing
(15 CUDA cases skipped). The lab's independent source fixture covers all five
modes, ordinary and clipped-XSA inputs, all parameters and gradients. Full-state
resume and the validation stopping rule are tested separately. No learning
result is claimed until the frozen cohort runs.

Final checkpoint scores will be replayed on CPU FP32 after learning closes,
including the reserved test. The report will compare validation counts with
the original CUDA observations. An FP64 audit will keep every RMS epsilon at
its FP32 value so it checks the same finite-epsilon model.

First-pass learning allowances are360s per primary case and180s per mechanism
control. If time remains, one fixed-order round continues time-censored cases
from their complete state, sharing the remaining allowance equally. Successful
runs stop early; no failed run is automatically retried. The durable phase
deadline applies to every process and leaves180s for final GPU checks.
