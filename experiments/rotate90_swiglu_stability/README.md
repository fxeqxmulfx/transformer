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

The nine configurations are valid and training is in progress.
Report every seed's first crossing,
later loss of fit on each split and remaining-above-threshold observations,
with both update and nominal-clock horizons. The study tests a finite-step
stability hypothesis; it does not prove gradient attraction or resolve the
scalar-input/readout comparison.
