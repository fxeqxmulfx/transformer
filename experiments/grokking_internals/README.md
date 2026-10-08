# Internal measurements during ordinary transformer grokking

Which internal observations distinguish rule formation from memorization
and confidence growth? Check all six proposed measurements: frozen linear
probes, gradient agreement, activation spectra, FFN neuron specialization,
causal ablations, and functional update decomposition.

The follow-up pair-removal study is documented below: all 28 head pairs
at 23 immutable snapshots, with answer-only CE and explicit zero-quotient
controls. It adds observations, not training variants.

| Run | Difference from grokking_progress | Budget |
| --- | --- | ---: |
| `gptmini-seed1` | Save every 1,000 updates instead of every 5,000 | 150,000 |

The ordinary GPTMini softmax architecture, native AdamW, data split and
seeds match the earlier delayed-generalization run. Measurements operate
on CPU copies of saved weights. Intermediate weights are retained by
`archive.py`, which also watches the original study's controls. No probe
changes a training model, its optimizer, sampler or random state.

The primary, repeat, seed 2 and reference control have completed their
full budgets. The repeat matches all 601 canonical observations of the
primary exactly. Seed 3 also completes all 150,000 updates. Seed 2's
brief earlier interruption before its first checkpoint remains documented
under `runs/gptmini-seed2/interrupted_before_first_checkpoint/`; the complete
run restarted from its same initialization. All budgets remain unchanged.

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

The reference control completes 150,000 updates with 1.503% held-out answer
accuracy and a frozen probe near chance. Seed 2 completes 150,000 updates
with 100% accuracy and first exceeds 99% at 1,250 updates. Seed 3 first
exceeds 99% at 750 updates and completes at 100%. These two initializations
are early-generalizing controls, not replications of seed 1's long plateau.
Seed 2 temporarily loses accuracy at 35,000 and later recovers; that event
is not its first acquisition of a generalizing solution.

Sixteen focused tests pass on CPU and CUDA, including equality of actual
AdamW parameters, optimizer state, existing gradients and RNG through the
next update, intervention/capture cleanup after failures, a training-only
memorizer, fixed-probe held-out-label independence, exact known spectra
and confidence-only counterexamples. The complete `./make.py test` suite
passes: **249 tests in 1,181.248 seconds**, including available CUDA tests.

## Answer versus shared EOS gradients

`compare_objectives.py` adds offline observations on pinned available
checkpoints of the repeat, both seed controls and the reference. It leaves
the six-measurement worker and all training source unchanged. Each sampled
batch supplies three gradients from the same training forward: answer CE,
EOS CE, and the actual mean CE. The EOS position sees the supplied answer,
as it does during training. Batches and unique parameters, including tied
embedding/readout weights, match the existing gradient protocol.

The reader verifies `g_full = (g_answer + g_EOS) / 2` numerically, retains
the signed cross term in squared norms, and decomposes the train/held-out
mean-gradient dot product into four component terms. Those terms are not
nonnegative fractions; cancellation and cross-component alignment matter.

| Initialized model | Full train/held-out cosine | Answer cosine | EOS cosine | EOS–EOS contribution to full cosine |
| --- | ---: | ---: | ---: | ---: |
| GPTMini seed 1 | 0.9524 | 0.3154 | 0.9807 | 0.9297 |
| GPTMini seed 2 | 0.9581 | 0.2818 | 0.9781 | 0.9338 |
| GPTMini seed 3 | 0.9615 | 0.3567 | 0.9819 | 0.9712 |
| Reference control | 0.9886 | 0.7723 | 0.9954 | 0.9767 |

Shared EOS supervision supplies most of the full cosine's numerator at
initialization. It does not explain every early agreement: seed 1 at
1,000 updates has answer-only cosine 0.8567 while EOS contributes less
than 0.000001 to the full cosine. At the actual 30,000–35,000 transition,
answer-only cosine decreases from 0.0469 to 0.0336. Thus removing EOS does
not turn this measurement into a standalone detector.

[The component results](objective_component_results.json) retain 23
observations with checkpoint, source and batch hashes. Missing intermediate
reference weights are explicit; its first retained noninitial snapshot is
45,000. All four measured runs now include their final weights. Maximum
relative gradient-reconstruction error for all parameters is below
`5e-7`. Initialized-model records are distinguished from training weights.

Seven new tests cover opposing answers aligned by a shared signal, signed
cancellation, cross-component dot terms, the actual supervision mask,
unused/tied parameters, failure cleanup and unchanged next AdamW updates
on CPU/CUDA. The full suite passes **256 tests in 942.386 seconds**.
Reproduce from `python/` with `uv run --locked python
../experiments/grokking_internals/compare_objectives.py`; rerunning fills
newly available pinned snapshots without changing the recipe.

## Exact certificates for current answers

`compare_geometry.py` evaluates the Lean cleanup criterion on the same
23 pinned checkpoints. `compare_geometry_all.py` reads all **238**
preserved snapshots: 151 from the repeat, 2 from the original, 31 from each
of seeds 2 and 3, and 23 from the reference. These are observations of the
completed 150,000-update runs; no training budget or optimizer changes.
The pinned reader and the full reader agree exactly on every shared
measurement and checkpoint hash. Missing reference weights before 45,000
remain absent, apart from its initialization.

The reference is the actual held-out cell mean of raw logits, with class
bias retained. Align each observed row to the reference's row mean; this
changes no class ordering. For the correct reference margin `m > 0` and
that row's residual squared energy `R`, `2 * R < m^2` certifies its strict
correct answer. The reference is computed from the current outputs; it
is an offline diagnostic using target labels, not a decoder or supervision
added to training. All measurements below exclude the zero quotient.

`geometry_certificates.py` represents the finite observed binary logits
by exact integers with a common denominator. For `n` observations and
`C` classes, its residual numerator is
`n*C*x - n*row_sum - C*column_sum + total_sum`. With target column-sum
gap `g`, the exact test is `g > 0` and
`2 * sum(residual_numerator^2) < (C*g)^2`. No rounded squared energy
decides the certificate count. Quantiles, energy summaries and the signed
margin/error balance still use float64.

Lean's `Geometry.RowAlignment`, `IntegerEncoding` and `IntegerCertificate`
add eighteen proved identities and sufficient/necessary implications:
the integer test equals the real cleanup inequality, preserves positive
scale invariance and certifies the original raw row. The integer encoding
is an input; Python's float extraction, its program and the transformer's
floating-point forward execution are not formally verified.

| Seed 1 repeat update | Actual nonzero accuracy | Cell-mean nonzero accuracy | Exact certified fraction |
| --- | ---: | ---: | ---: |
| 33,000 | 13.39% | 25.38% | 0.00% |
| 34,000 | 68.97% | 99.11% | 0.22% |
| 35,000 | 98.35% | 100.00% | 27.79% |
| 36,000 | 99.87% | 100.00% | 95.94% |
| 38,000 | 99.96% | 100.00% | 99.72% |
| 40,000 | 99.70% | 100.00% | 50.93% |
| 150,000 | 100.00% | 100.00% | 100.00% |

The first preserved seed 1 cell-mean accuracy above 99% is at 34,000;
the first certified fraction above 99% is at 38,000. The separate canonical
evaluation, every 250 updates and including the zero quotient, first
exceeds 99% at **35,500**. There are no retained weights at 35,500 in this
reader. The cell projection supports the interpretation that a correct
functional component precedes stable raw decisions; it does not identify
an attention/FFN algorithm or prove how AdamW learned that component.

Across 1,159,032 current-input certificate evaluations, **759,256** tests
certify an answer and **zero** certified answers are incorrect. These are
applications across snapshots, not distinct examples. There are zero
exact/float64 predicate disagreements here, with maximum float64 energy
decomposition error below `1e-9`. Every snapshot meets the two-point
nonzero-cell coverage and energy protocol. The conservative certificate
using the global residual sum certifies no answers in any snapshot.

The sufficient certificate is not a monotone onset detector. Seed 1's
coverage falls at 40,000 while accuracy remains high. Seed 3 finishes
with 100% accuracy but 83.72% certificate coverage. Seed 2's loss of
accuracy at 35,000 also changes the cell reference. Seed 1's cell-mean
accuracy already reaches 59.60% at 1,000 updates, then falls during the
long plateau; that early rise alone does not predict sustained success.
The reference ends with 0.49% nonzero
accuracy and no certified answers; its previously reported 1.503%
canonical accuracy includes the zero quotient.

Ten new tests check an independent rational oracle, extreme binary
scales, the strict boundary, wrong symmetric references, class-bias
restoration, masked locality, sum/mean factors, scale invariance and
noninterference. The full suite passes **266 tests in 911.148 seconds**.
[Pinned results](geometry_certificate_results.json),
[all preserved results](geometry_all_results.json) and
[the figure](geometry_certificates.svg) retain the protocol and hashes.
Reproduce from `python/` with `uv run --locked python
../experiments/grokking_internals/compare_geometry.py`, then
`uv run --locked python ../experiments/grokking_internals/compare_geometry_all.py`.
For the figure use `uv run --locked --with matplotlib==3.10.8 python
../experiments/grokking_internals/plot_geometry.py`.

## Artifacts and thermodynamic interpretation

`internal_suite_results.json` now includes all five completed run records
and the canonical-repeat comparison. It pins checkpoint and source-prefix hashes,
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

## Pairwise head interactions: retrospective sensitivity, not a detector

Does a useful multi-part computation become visible before the model
starts generalizing? The composition hypothesis in Nanda et al.,
arXiv:2301.05217v1, appendix Further speculations on grokking, motivates
measuring all four losses for every unordered pair of heads:

`I = L(both present) - L(A removed) - L(B removed) + L(both removed)`.

The Lean scalar specialization at commit `b141ba0` proves a negative
contrast for aligned components supplying a product logit. The real
transformer is measured separately: each head's merged output is removed
at all prompt positions, after XSA and before its projection. All four
answer-only CE losses, accuracies and changed predictions are retained
for both splits, including separate nonzero-quotient scopes. EOS is not
part of this measurement. No pair is fitted, selected using later weights,
or supplied to training. The analysis enumerates all eight heads and all
28 pairs at every available pinned snapshot.

`pair_interactions.py` defines the observation and `compare_interactions.py`
reads the same pinned steps as the objective-components study. Run from
`python/` with `uv run --locked python ../experiments/grokking_internals/compare_interactions.py`.
SHA-256 identities pin source, task rows and each actual checkpoint. Modes,
RNG, weights, buffers and gradients are retained; temporary hooks are
removed even on failure. Missing weights are recorded, not reconstructed.
The four reference snapshots at 5k, 30k, 35k and 40k remain unavailable.
The completed [result](pair_interaction_results.json) contains 23 snapshots
and 644 pair observations. Each declared training run remains at its
completed 150,000-update budget.

Count a pair only when both individual removals increase the measured
loss and `I` is negative beyond the declared heuristic tolerance
`1e-9 + 1e-7 * max(abs(corner losses))`. This tolerance is not a proved
floating-point error bound. The following table uses held-out **nonzero**
quotients throughout:

| Seed 1 update | Answer accuracy | Counted pairs, out of 28 |
| ---: | ---: | ---: |
| 0 | 0.33% | 4 |
| 1,000 | 47.04% | 20 |
| 30,000 | 7.56% | 0 |
| 33,000 | 13.39% | 3 |
| 34,000 | 68.97% | 7 |
| 35,000 | 98.35% | 3 |
| 36,000 | 99.87% | 9 |
| 40,000 | 99.70% | 7 |
| 150,000 | 100% | 13 |

The count is nonmonotone and already substantial at 1k, followed by much
poorer accuracy at 30k. The most negative eligible contrast is about
`-8.03` at 1k, versus `-0.19` at 35k. Neither count nor magnitude alone
is a forecast of the later transition. Early-generalization controls
also have eligible pairs at initialization: six in seed 2 and two in
seed 3. Their first 99% canonical crossings remain 1,250 and 750,
respectively; the sparse intervention snapshots do not redefine them.

At the completed budget, using nonzero quotients on both splits:

| Run at 150k | Held-out accuracy | Counted train pairs | Counted held-out pairs |
| --- | ---: | ---: | ---: |
| Seed 1 | 100% | 13 | 13 |
| Seed 2 | 100% | 14 | 14 |
| Seed 3 | 100% | 4 | 4 |
| Reference, 20% training fraction | 0.49% | 28 | 0 |

Every pair meets this criterion on the failed reference's **training**
examples, while no pair does on its held-out examples. Thus useful
training interactions do not certify a transferable rule. Different
100%-accurate seeds also have different counts. The reference's 0.49%
nonzero accuracy is distinct from its 1.50% all-quotient accuracy; the
trivial zero quotient is an explicit confound rather than hidden evidence.

A negative contrast can arise from opposed additive scores with no
product logit, as the numerical control tests demonstrate. Even paired
removal is an off-manifold intervention with downstream nonlinearities;
it does not recover a unique head algorithm, establish a causal training
mechanism or prove a thermodynamic transition. The full four-corner data
support named sensitivity comparisons, not a universal grokking metric.

Lean also proves the stronger two-example counterexample in
`Transformer.Grokking.Composition.AdditiveContrast`: additive scores
`2*a-b` and `2*b-a` give negative mean-CE interaction and **both** useful
individual removals at `(1,1)`. Their actual mixed score derivatives
and four-corner score contrasts are nevertheless zero. The loss can
create the interaction pattern by itself. The next observation should
retain four-corner logit contrasts before CE, removing common row shifts;
it still must control for normalization and downstream FFN nonlinearities.
This refinement has been proved on the scalar examples and has not yet
been measured as a new real-transformer diagnostic.

Validation on 2026-10-08: `./make.py check experiments/grokking_internals`
and the full `./make.py test` pass (275 tests, including nine new pair
controls/noninterference checks). Lean build, audit, generated index and
forbidden checks pass; no new sorry or additional axioms were introduced.
