# Conditional residual amplitude in the current transformer

Task158 tests whether the conditional amplitude path identified by
[`context_gains`](../context_gains) improves actual transformer grokking.
The baseline is the current `diff_gpt` GPT at `df1e9f7`, including the corrected
RMS backward, centered XSA, QK normalization, RoPE, Peri-RMS, SwiGLU, tied
embedding and fixed post-RMS coefficient `1/sqrt(2L)`.

The candidate multiplies each normalized branch by
`2*sigmoid(W_gate*RMS(branch_input))`. Each gate is a bias-free linear row,
initialized to zero. All common weights, initial logits, common parameter
gradients and random-generator state initially agree exactly. The extra
`2*L*width` parameters total 1,024 at the measured size. The gain stays between
zero and two; it can suppress or strengthen a branch based on its input.
This is a model extension, not a correction to a false forward equation.

| Difference | Canonical | Adaptive |
| --- | --- | --- |
| Branch multiplier after output RMS | `1/sqrt(2L)` | `2*sigmoid(W_gate*x_norm)/sqrt(2L)` |
| Additional parameters | 0 | 1,024 |
| Initial gain | 1 | 1 |
| Common initialization | Same seed | Identical common tensors |
| RMS backward | Corrected | Corrected |

Both seeds11/17 use division at P97 and P193, width128, depth4, heads4, the same reserved
train/validation/test split and batch seed11, Schedule-Free AdamW lr.003,
betas(.9,.98), decay.1, ten-update inclusive warmup, batch256, FP32,
Python3.15/Torch2.15 and CUDA fullgraph max-autotune with math SDPA. The second
pair on each prime reverses the execution order. Every update records its gradient norm;
canonical validation, full-state checkpoints and per-parameter gradient/update
diagnostics occur every1,000 updates.

The user requested **stop each promising run at99% validation**. Therefore
the dedicated cohort runner stops at the first canonical crossing, even if
training accuracy is lower, and retains that endpoint. The original
`ResidualPairs` stable-stop definition remains historical; use the cohort
command below for this comparison. The initial100,000-update budget is a
continuation checkpoint, not proof of failure: an uncrossed promising run
must be continued before opening test. Numerical failures remain explicit.

The predeclared integration rule is an earlier first99% crossing in both
paired seeds on both primes, plus passing source, derivative, complete-state and actual GPT
regressions. Report both update counts and elapsed time. Test stays sealed
until all learning closes and never chooses an architecture or stopping step.
Two seeds on two primes cannot establish general language-model superiority. The previous native-RMS fixed-scale runs are historical context,
not the controls for this corrected-RMS cohort.

`python/tests/references/gpt_df1e9f7` contains the independent frozen actual
GPT source. Its manifest pins the original files and import-only local
rewrite. Tests compare the lab baseline's parameters, logits and every
gradient to it, including the clipped-XSA regime. Nonzero candidate gates
are compared to a separately written recurrence over those source modules;
compiled execution and exact Schedule-Free continuation are also checked.
The current port also preserves the exact source subtraction graph: FP32 logits
and every gradient match at both measured primes/seeds and input lengths2/4.

Run `./run.py adaptive-residual freeze`, `profile-run`, then `report`.
`profile-run` launches the training process as the profiler's child, then samples
its main thread for10 seconds at1kHz after each case's first1,000-step observation.
It needs no system tracing-policy change. All cases get the same window; raw
profiles record labels, step boundaries and failures. Sampling reports Python
wall-stack residency, not CUDA kernel durations. The plain `run` action uses
the same training loop without profiling. The protocol
pins every lab/optimizer source, cohort script, split and recipe. Learning
cannot silently continue after source changes or final-test disclosure.
The unused initial P97 protocol is retained in `preflight/`; no GPU learning
ran under it. P193 and diagnostics were added before observing any GPU outcome.
