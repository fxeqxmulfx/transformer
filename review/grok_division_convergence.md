# Division mod 97: does held-out accuracy tend to 1? Target and gaps

Board tasks #126–#132 and #137. Sources: `papers/arXiv-2201.02177v1/main.tex` (Power et al.),
Figure 1 and the appendix *Additional experimental details*; openai/grok at `3d64b1d8`; the
protocol pinned in `experiments/grok_division/PROTOCOL.md`. Written 2026-11-11.

The paper shows curves of finite runs. It states no theorem, and nothing below proves that any
transformer generalizes. This note fixes what such a theorem would say, lists what the
repository proves about neighbouring objects, and names the bridges that are missing.

## 1. The target

`src/Transformer/Grokking/Protocol/`, namespace `Transformer.Grokking.Protocol`.

| Object | Declarations |
| --- | --- |
| Prediction over all classes, ties to the first of the largest logits | `predict`, `predict_eq_iff` |
| Competitor margin; positive margin gives the prediction, not conversely | `margin`, `predict_eq_of_margin_pos`, `margin_nonneg_of_predict_eq` |
| Accuracy on a finite set; below 1 it is at most `1 - 1/|H|` | `accuracy`, `accuracy_eq_one_iff`, `accuracy_le_of_ne` |
| Pathwise: accuracy tends to 1 iff eventually every decision is right | `tendsto_accuracy_one_iff` |
| Split and batches as functions of shuffles | `trainAt`, `heldout`, `batchSlots`, `nodup_batchSlots`, `length_batchSlots` |
| The author's Adam and AdamW update and run | `Hyper`, `Hyper.source`, `Hyper.rateAt`, `adamStep`, `adamRun` |
| Well-posedness and finite-history bounds | `adamDenominator_pos`, `one_sub_pow_pos`, `adamRun_second_nonneg`, `abs_adamRun_first_le`, `adamRun_second_le`, `abs_adamRun_step_le` |
| Sample space and law of a run, infinitely many epochs | `Sample`, `law`, `uniformPerm`, `trajectory`, `heldoutAccuracy` |
| Modes of convergence | `TendsToOneAlmostSurely`, `TendsToOneInProbability`, `TendsToOneInMean`, `TendsToOneAlmostSurely.inProbability`, `TendsToOneAlmostSurely.inMean` |
| The target and its meaning | `Generalizes`, `generalizes_iff` |
| It is neither empty nor automatic | `generalizes_indicator`, `not_generalizes_const` |

`Generalizes ν logits label h lossGrad eqs hk b` says: for almost every sample (a shuffle of
the equations, initial parameters of law `ν`, a shuffle of the training set for every epoch,
all independent) the held-out accuracy along the Adam trajectory tends to 1. The mode is
almost sure. By `generalizes_iff` this is: almost surely, from some update on, every held-out
equation is answered. It is a statement about decisions. It does not ask the parameters, the
loss or the margins to converge, it is not implied by one good checkpoint, and it is not a
statement about an average over finitely many seeds. It has no premise about the future of
the run: its only inputs are the protocol's.

The infinite continuation is the protocol's own: one more shuffle of the training set for
every further epoch, at the constant learning rate that the schedule has after warmup.

## 2. What the repository proves, and about what

| Where | Declarations | It is about | Why it is not the target |
| --- | --- | --- | --- |
| `FlatSpectrum/Training` | `decayRun`, `IsDecayedMin`, `exists_tendsto_decayRun_ceLoss_isDecayedMin`, `exists_pos_ae_tendsto_decayRun_ceLoss_isDecayedMin` | Gradient descent on a product `U Vᴴ` of complex factors with the penalty `λ(‖U‖² + ‖V‖²)`, `λ > 0`: for almost every start the run converges to a minimum | Another model (no embedding, attention, normalization or ReLU), another optimizer (Euclidean descent, coupled penalty), and `λ > 0` |
| `FlatSpectrum/Training` | `tendsto_margin_div_nuclearNorm`, `margin_eq_maxMargin_of_tendsto_inv_smul`, `tendsto_inv_smul_of_unique` (#108), `tendsto_inv_smul_codeProbe_iff`, `tendsto_inv_smul_cubicKernel_iff` (#109) | The directions of the minima as `λ → 0⁺`: a sequence of separately optimized fits | A limit over problems, not along one trajectory; with `λ = 0` there is no minimum at all |
| `FlatSpectrum/Training/NoDecay` | `not_exists_forall_ceLoss_le`, `not_isDecayedMin_ceLoss_zero`, `not_isCompact_setOf_flatLoss_zero_le` | On separable data the undecayed cross-entropy has no minimizer and unbounded sublevel sets | It removes the compactness that every convergence proof above uses; it says nothing about where the run goes (#118 is open) |
| `FlatSpectrum/Training/UnseenRows` | `IsDecayedMin.row_eq_zero`, `IsDecayedMin.col_eq_zero`, `IsDecayedMin.ceLoss_heldOut_eq` (#110) | A decayed minimum is zero on unseen rows and columns of the product | A property of that model's minima; the decoder has no such product |
| `FlatSpectrum` | `isMinOn_nuclearNorm_iff`, `IsFlat.exists_neg_mul_of_isMinOn_cell`, `IsDecayedMin.ne_smul_of_isFlat` | A flat kernel is not recovered from a proper subset of its cells by the least nuclear norm | Failure to reconstruct a kernel matrix is not failure to classify: the target needs only the largest logit, and the decoder is not a least-nuclear-norm interpolant (#124 is open) |
| `Grokking/Operational` | `FirstCrossing`, `FirstSustained`, `firstCrossing_le_firstSustained`, `finite_prefix_future_ambiguity`, `no_universal_finite_crossing_detector` | Predicates of an observed accuracy trace | They describe records of a run and prove that a finite prefix does not decide the future |
| `Grokking/DivisionOrbits` | `InDivisionCell`, `equal_quotients_iff_common_scale`, `mem_full_cell_iff`, `modulus97_domain_card` | The algebra of the task: equal answers are common scalings | No model in it |
| `Landscape/Neural` | `crossEntropy`, `crossEntropy_nonneg`, `analyticAt_crossEntropy`, `exists_compact_rotate90_adamw_container` | The loss of a row; analyticity and an AdamW box for the Rotate90 model (SiLU, RMS norm, tied readout) | Another architecture and schedule; the paper's decoder has ReLU and is not analytic |
| `Grokking/AdamW`, `EffectiveTheory`, `CircuitEfficiency`, `Composition`, `Geometry` | for example `coupled_first_growth_iff`, `quarticCE_every_rate_can_overshoot` | Effective models and diagnostics of learned circuits | None is the decoder's trajectory |
| board #78 (in progress, another agent) | not on `main` | Quotient-equivariant logits carry two training anchors to every equation | A sufficient structure for held-out correctness; no proof that training produces it |

Two substitutions are therefore excluded, and no statement here makes them:

- Convergence of decayed matrix-factorization minima, or of their directions, is not
  convergence of the undecayed transformer trajectory under Adam.
- The flat-kernel obstruction to reconstruction is not a proof that held-out classification
  fails, and its absence would not be a proof that it succeeds.

## 3. The bridges that are missing

1. **Arithmetic.** The target is in exact reals; every run is in `f32`, where the softmax
   saturates (PROTOCOL.md, Amendment 1). No statement relates the two trajectories.
2. **Measurability.** `TendsToOneAlmostSurely.inProbability` and `.inMean` need the accuracy to
   be measurable in the sample. This is not proved for the decoder.
3. **Exceptional parameters.** The target takes the gradient oracle as an argument. The
   source's backward pass has a rule at hidden inputs equal to 0; the decoder's oracle has to
   be that rule (#127), and that the trajectory almost surely avoids such parameters is not
   proved.
4. **Any control of the trajectory.** Without decay the loss has no minimizer on fitted data
   and the parameters need not stay bounded. The proved bounds are those of the moments and of
   one step; no descent, boundedness or limit of the Adam trajectory is proved or assumed.
5. **Structure gives correct answers** (#129, after #78): a certificate on the decoder's own
   logits that implies the right largest logit on every held-out equation.
6. **Training reaches the structure** (#130): from the declared initialization law, with a
   stated probability.
7. **Composition** (#131): 5 and 6 along one trajectory, for all later updates.

Bridges 4 to 7 are open research problems. The measurements of `experiments/grok_division`
are finite runs in `f32` and close none of them.
