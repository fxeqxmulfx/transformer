# Internal measurements during ordinary transformer grokking

Which internal observations distinguish rule formation from memorization
and confidence growth? Check all six proposed measurements: frozen linear
probes, gradient agreement, activation spectra, FFN neuron specialization,
causal ablations, and functional update decomposition.

| Run | Difference from grokking_progress | Budget |
| --- | --- | ---: |
| `gptmini-seed1` | Save every 1,000 updates instead of every 5,000 | 150,000 |

The ordinary GPTMini softmax architecture, native AdamW, data split and
seeds match the earlier delayed-generalization run. Measurements operate
on CPU copies of saved weights. Intermediate weights are retained by
`archive.py`, which also watches the original study's controls. No probe
changes a training model, its optimizer, sampler or random state.

The repeat is active on the GPU, following the completed primary and
preceding the queued seed controls. The CPU reference control continues.
All declared training budgets remain unchanged. Seed 2 briefly started
while the queue was reordered; it was interrupted before its first
checkpoint. Its initial records are preserved under
`runs/gptmini-seed2/interrupted_before_first_checkpoint/` in the original
study. The queued ordinary run will restart from its same initialization.

## Protocol

- Frozen linear readouts use ridge least squares on standardized **train
  features only**, with fixed strength 0.001. Held-out labels score the
  fitted readout. Shuffled training labels supply a negative control.
  Failure of this particular readout does not prove absent information.
- Gradients use `autograd.grad` on eight fixed, disjoint 32-example batches
  per split under the original answer-plus-EOS objective. Record pairwise
  cosines per parameter group and between split means. No gradient is
  applied. High agreement can arise from common targets such as EOS.
- Center activations and eigendecompose covariance on each split. Record
  `exp(entropy(eigenvalue / total))`, stable rank and the rank retaining
  99% energy. This is covariance-energy entropy, not singular-value entropy.
- Actual FFN post-activation neurons are measured before their output
  projection. Record exact zeros, train-RMS-relative zeros, train-selected
  class selectivity and train/held-out agreement of quotient profiles.
  Exclude the trivial zero quotient from neuron profiles.
- Remove each head's actual merged output after XSA and before its output
  projection, at every prompt position. Separately remove eight top or
  bottom train PCs and a matched random subspace at the answer-query
  residual, preserving its train mean. Record answer loss/accuracy changes
  on both splits. Off-manifold interventions measure named sensitivities.
- Between retained endpoints, project row-centered logit change onto the
  previous logits. Record scale and orthogonal-geometry energy shares,
  the fitted scale, changed predictions and scaling-only hypothetical CE.
  A negative fitted scale does not preserve decisions; a shrinking scale
  is not confidence growth. Multi-update intervals remain explicit.

Sources and deviations are documented in each module: Nanda et al.,
arXiv:2301.05217v1, sections 4–5; Golechha, arXiv:2405.12755v1, section
3.1; Prieto et al., arXiv:2501.04697v1, section 4.2. Local manuscripts
are in `papers/`. No mathematical guarantee is borrowed from them for
the actual ordinary-transformer training trajectory.

## Found around the measured transition

The repeated canonical train/held-out metrics agree exactly with the
original at every compared observation. The transition is reproduced:
first 99% held-out answer accuracy at 35,500, after the 33,000 structural
signal. All six probes have now measured the preserved pre-transition
and post-transition weights.

| Measurement | Update 30,000 | Update 35,000 |
| --- | ---: | ---: |
| Model held-out answer accuracy | 8.63% | 98.37% |
| Frozen block 1 residual probe | 8.48% | 97.34% |
| Frozen block 1 neuron probe | 8.42% | 98.41% |
| Shuffled-label residual probe | 1.16% | 0.54% |
| Block 1 residual energy entropy rank | 49.84 | 61.65 |
| Train-batch gradient mean pair cosine, all parameters | 0.0469 | 0.00311 |
| Train/held-out mean gradient cosine, all parameters | 0.0469 | 0.0334 |
| Block 1 FFN exact zero activation fraction | 83.74% | 82.49% |
| Median FFN quotient-profile agreement | 0.556 | 0.983 |
| Largest single-head held-out accuracy drop | 5.09 percentage points | 58.76 percentage points |

The fixed probes first exceed 99% at the 36,000 checkpoint, after the
model's 35,500 crossing. This decoder did not expose a hidden 99% solution
long before the model readout. Gradient agreement and sparsity do not
rise monotonically. Rank fluctuates strongly across nearby checkpoints.
None is a standalone grokking detector.

Between 34,000 and 35,000, **81.63%** of measured held-out logit-change
energy is orthogonal to a global rescaling of the previous logits. The
best fitted global scale is **0.778**, not an increase. The observed
accuracy gain in this interval involves changing decision geometry.

At the final original 150,000 checkpoint, removing block 0 head 2 reduces
held-out answer accuracy from 100% to zero. This demonstrates sensitivity
to removing that head; it does not establish the head's complete algorithm
or show when it first became necessary. The repeated checkpoints supply
the timing study.

The reference control still has roughly 1–2% held-out answer accuracy and
its frozen probe remains near chance at available checkpoints. Its full
budget, the repeat's remaining updates and the queued seed controls are
not replaced by partial-budget conclusions.

Sixteen focused tests pass on CPU and CUDA, including equality of actual
AdamW parameters, optimizer state, existing gradients and RNG through the
next update, intervention/capture cleanup after failures, a training-only
memorizer, fixed-probe held-out-label independence, exact known spectra
and confidence-only counterexamples. The complete `./make.py test` suite
passes: **249 tests in 1,181.248 seconds**, including available CUDA tests.

## Artifacts and thermodynamic interpretation

`internal_suite_results.json` pins checkpoint and source-prefix hashes,
observer code, sampled updates, raw metrics, and threshold crossings.
`six_measurements.svg` and `control_comparison.svg` are scientific figures.
Do not infer missing intermediate weights from connected curve segments.

The finite trajectory is compatible with a transition in representation
dynamics. A thermodynamic phase transition would additionally require
specified order/control parameters and an asymptotic regime. Neither an
effective temperature for AdamW nor finite-size scaling is established
here. Liu et al., arXiv:2205.10343v2, discuss an effective representation
theory; Žunkovič/Ilievski, arXiv:2210.15435v1, analyze solvable phase
transition models. Their assumptions are not transferred to GPTMini.
The active formalization cycle is [grokking_lean_plan.md](../../grokking_lean_plan.md).

From `python/`, reproduce the snapshot with `uv run --locked python
../experiments/grokking_internals/summarize.py`, and figures with
`uv run --locked --with matplotlib==3.10.8 python
../experiments/grokking_internals/plot.py`. Run `archive.py` and `observe.py`
as long-lived read-only workers to retain and measure future checkpoints.
