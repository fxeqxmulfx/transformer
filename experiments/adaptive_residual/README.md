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

The first P193 preparation stopped before any update: the CPU and CUDA
`copy_last_accuracy` diagnostic differed by one float64 ULP, while every
row, label, count and fingerprint agreed. `metadata_amendment.json` retains
the original protocol hash, completed P97 checkpoint hashes, and the exact
replacement runner hashes. The repaired check permits only that one-ULP
diagnostic difference. Model, engine, optimizer, sampler, data, stopping and
selection rules are unchanged; the original protocol remains byte-identical.
Preparation errors are now recorded explicitly before the runner exits.

## Measured outcome

All eight runs crossed99% validation; all135,000 recorded update gradients
were finite and nonzero. The candidate crossed earlier in three pairs and
tied in one. It **did not satisfy the predeclared all-four-earlier promotion
rule**, so production GPT remains `df1e9f7`. The user-requested Basis
comparison is a separate follow-up; final test did not alter this decision.

| Prime / seed | Canonical first99% | Candidate first99% | Canonical test | Candidate test |
| --- | ---: | ---: | ---: | ---: |
| 97 / 11 | 26,000 | 21,000 | 99.5704% | 99.2698% |
| 97 / 17 | 24,000 | 20,000 | 99.1838% | 99.2698% |
| 193 / 11 | 8,000 | 8,000 | 99.9244% | 99.2552% |
| 193 / 17 | 15,000 | 13,000 | 99.8057% | 99.6978% |

Final test was opened after all learning closed. These are CPU replays of
complete Schedule-Free evaluation states; their validation counts match
the canonical CUDA history exactly. A tie at the1,000-update observation
cadence does not establish equal underlying crossing times. The comparison
shows conditional amplitude can help these runs, not a universal grokking
improvement or a flat-spectrum impossibility result.

Training-clock seconds were138.64/115.06,131.30/111.05,43.71/47.60 and
85.90/74.95 for canonical/candidate in table order. Preparation ranged from
22.45 to132.23 seconds. Compilation caches, CPU regression work, and the
recorded metadata interruption make end-to-end times unsuitable as an
isolated throughput comparison. The full phase lasted37m27s including the
interruption; per-case timers and complete histories remain in `results.json`.

Python3.15 sampled all eight warmed cases for ten seconds each, with every
window contained in its intended case. Sampling error rates were about5–6%.
The main thread spent about63–74% of successful samples at the blocking
batch-index upload. That can include waiting for earlier CUDA work, so it
does not prove a CPU preprocessing bottleneck. Task159 records the separate
CUDA-timed transfer experiment needed to test that interpretation.

Replay the measured learning/report sources at commit `23d16a5e`; the
original protocol remains pinned to its engine sources at `598979b6`.
