# SwiGLU step size and loss-of-fit controls

Does the gated Rotate90 model's later loss of fit persist with smaller
AdamW steps? In the [activation study](../rotate90_analytic/README.md),
SwiGLU first reaches 99% quickly but later falls as low as 86.1% on both
splits at learning rate 0.01. This M9.5 control retains W128, hidden width
43, all three tied roles of E, the arithmetic split and continuing sampler,
AdamW betas (0.9, 0.98), epsilon 1e-8 and weight decay 1.

| Prefix | Learning rate | Updates | Epochs | Nominal decay clock at the end |
| --- | ---: | ---: | ---: | ---: |
| lr01 | 0.01 | 10,000 | 1,000 | 99.955 |
| lr003 | 0.003 | 33,330 | 3,333 | 99.9765 |
| lr001 | 0.001 | 99,960 | 9,996 | 99.9555 |

Every prefix has seeds 11, 17 and 23. The nominal clock sums each applied
learning rate times decay. With ten inclusive warmup updates it is
`lr * (updates - 4.5)` after warmup. Compare both the common first 10,000
updates and the complete budgets, so a shorter optimization time cannot
by itself account for the absence of late failures. Matching this scalar
clock does not assert that AdamW follows the same orbit: its moments and
minibatch noise also depend on the update count. The discrete product of
decay factors is not exactly matched.

All splits are evaluated each epoch. Every update records its gradient
norm, and every tenth update also records per-tensor parameter, gradient,
update and retained moment norms. These detached measurements extend the
first study's observers. Checkpoints save every 1,000 updates and at the
end. The locked lab implementation is unchanged from `c59d4426`.

```sh
./run.py check experiments/rotate90_swiglu_stability
./run.py run experiments/rotate90_swiglu_stability
./run.py report experiments/rotate90_swiglu_stability
```

After completion, export the same checkpoint controls and full curves from
`python/`:

```sh
uv run --locked python ../experiments/rotate90_analytic/analyze.py ../experiments/rotate90_swiglu_stability
```

## Findings

All nine runs completed their full budgets, with 100% final train and
held-out accuracy. Each row below lists seeds 11, 17 and 23.

| Learning rate | First held-out 99%, epochs | Later observations below 99% | First-crossing nominal clocks |
| --- | --- | --- | --- |
| 0.01 | 14 / 15 / 16 | 24 / 14 / 12 | 1.355 / 1.455 / 1.555 |
| 0.003 | 30 / 35 / 30 | 0 / 0 / 0 | 0.8865 / 1.0365 / 0.8865 |
| 0.001 | 80 / 92 / 75 | 0 / 0 / 0 | 0.7955 / 0.9155 / 0.7455 |

Both smaller rates remain above 99% at every observed epoch after the
first crossing, through 33,330 and 99,960 updates respectively. They also
have no relapses in the common first 10,000 updates. The longer budgets
extend this observation to nearly the same nominal clock as the high-rate
runs. More optimizer updates are needed for the first crossing at smaller
rates; their nominal clocks are smaller, rather than a mere rescaling of
the high-rate trajectory.

The high-rate controls reproduce every final model tensor of the earlier
activation study exactly. For each seed, all 10,000 gradient records and
1,001 shared observations also agree exactly after removing timings.
The additional detached moment observers preserve these recorded values.
The same high-rate
losses of fit recur, with held-out minima 94.05%, 95.25% and 86.10% and
similar training dips. These controls support a finite-step optimization
explanation for the relapses; they are not held-out-only failures.

All final embeddings still admit the same character ablation: eight
energy-selected pairs retain 100% held-out accuracy in all nine runs, and
all 48 pairs reconstruct every FP32 parameter exactly. These are final
checkpoint tests. They do not prove that the representation or its correct
use persists indefinitely.

[results.json](results.json) includes per-seed outcomes at the common
10,000-update prefix and complete budget, nominal clocks, worst post-crossing
observations, full checkpoint ablations and provenance. [curves.csv](curves.csv)
retains all 42,996 canonical observations. [engine_code.json](engine_code.json)
records the unchanged lab implementation. Every gradient norm was finite;
the analysis verifies each final checkpoint against both full splits.
Per-tensor update and moment records remain in `runs/*/diagnostics.jsonl`.

This finite-horizon result does not prove AdamW convergence, typical-start
attraction or the scalar-input/readout comparison. The ordinary scalar
MLP's representability and gradient access remain separate obligations.
