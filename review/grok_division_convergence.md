# Division mod 97: does held-out accuracy tend to 1? Target and gaps

Board tasks #126–#132 and #137. Sources: `papers/arXiv-2201.02177v1/main.tex` (Power et al.),
Figure 1 and the appendix *Additional experimental details*; openai/grok at `3d64b1d8`; the
protocol pinned in `experiments/grok_division/PROTOCOL.md`. Written 2026-11-11; sections 5
and 6 and the enclosures of bridge 5 added 2026-11-12.

The paper shows curves of finite runs. It states no theorem, and nothing below proves that
training makes any transformer generalize. This note fixes what such a theorem would say,
proves that it is false unless something is asked of the split, lists what the repository
proves about neighbouring objects and about parameters of the decoder that are built rather
than trained, and names the bridges that are missing.

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
| The update is the code's, statement by statement | `sourceStep`, `adamStep_eq_sourceStep`, `adamStep_params_of_decay_eq_zero` |
| Well-posedness and finite-history bounds | `adamDenominator_pos`, `one_sub_pow_pos`, `adamRun_second_nonneg`, `abs_adamRun_first_le`, `adamRun_second_le`, `abs_adamRun_step_le` |
| Bounds that do not look at the gradients | `sq_adamRun_first_le` (`m² ≤ K v`), `abs_adamRun_step_le_rate`, `abs_adamRun_params_sub_le` (Adam: linear in `t`), `abs_adamRun_params_le` (AdamW: a fixed interval) |
| Sample space and law of a run, infinitely many epochs | `Sample`, `law`, `uniformPerm`, `trajectory`, `heldoutAccuracy` |
| Modes of convergence | `TendsToOneAlmostSurely`, `TendsToOneInProbability`, `TendsToOneInMean`, `TendsToOneAlmostSurely.inProbability`, `TendsToOneAlmostSurely.inMean` |
| The target and its meaning | `Generalizes`, `generalizes_iff` |
| It is neither empty nor automatic | `generalizes_indicator`, `not_generalizes_const` |
| The decoder in exact reals, all parameter groups | `Sizes`, `Param`, `layerNorm`, `weight`, `attention`, `block`, `decodeTo`, `logits`, `sum_weight`, `logits_congr` |
| Loss of a batch and the gradient of the backward pass | `Example`, `batchLoss`, `gate`, `decodeTo_gate`, `frozenLoss`, `frozenLoss_self`, `sourceGrad` |
| Means over the targets of a batch; the readout's gradient in closed form and its pairing with the readout | `targetMean`, `batchLoss_eq_targetMean`, `sourceGrad_out`, `sum_out_mul_sourceGrad` |
| The logits are bounded by the readout and the second normalizations | `abs_layerNorm_le`, `abs_decodeTo_le`, `abs_logits_le` |
| The task, its tokens and its supervision | `paperSizes`, `Equation`, `equations`, `read`, `read_injective`, `inDivisionCell_operands`, `supervised`, `targetCount_supervised` |
| Initialization law; the law of a run is a probability law | `initLaw`, `initMeasure`, `isProbabilityMeasure_initLaw` |
| The target for the division protocol, and what it means | `division_generalizes_iff`, `division_generalizes_source_iff` |
| What is proved about its trajectory | `abs_trajectory_division_sub_le`, `abs_trajectory_division_le`, `division_eventually_margin_lt`, `ae_abs_logits_divisionRun_le`, `ae_floor_le_batchLoss_divisionRun`, `ae_margin_divisionRun_le` (section 5) |
| The target is false without a condition on the split (section 2) | `not_division_generalizes`, `not_division_tendsToOneInProbability` |

For division the arguments are fixed in `division_generalizes_iff`: the decoder of
`paperSizes`, the logits after `=`, the source's loss on the answer and the closing token, the
backward pass `sourceGrad`, PyTorch's default initialization, 4656 of the 9312 equations, and
batches of 512. That theorem is an equivalence; the repository states the target nowhere as a
fact, an axiom or a `sorry`.

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

## 2. The target is false without a condition on the split

`Protocol/{Symmetry,AdamSymmetry,Uncovered,Exchange,Refutation,RefutationModes}.lean`.

A uniformly random half of the equations can miss a residue. Call a split *uncovered* when no
training equation has the residue 1 or 2 as numerator, denominator or quotient. At least 8734
of the 9312 equations avoid both (`card_avoids_ge`), so uncovered splits exist and have
positive probability (`exists_uncovered`, `uniformPerm_uncovered_pos`); the probability is
roughly `2⁻⁵⁹⁴`, by a count outside Lean.

On such a split the tokens of 1 and 2 are read by no training equation and are the target of
none. Exchanging them, which permutes two rows of the embedding and two of the readout
(`paramPerm`, `exchange`), commutes with every gradient of the run
(`sourceGrad_comp_paramPerm`, `divisionGrad_comp_exchange`), hence with the whole Adam or AdamW
run (`adamRun_comp`, `trajectory_exchange`), and preserves the initialization law
(`measurePreserving_comp_exchange`). The equations `1 / 3 = 65` and `2 / 3 = 33` are held
out, and because the decoder is causal the logits after `=` of the first at exchanged
parameters are those of the second (`answerLogits_exchange`). They cannot both be answered
(`not_both_predicted`): at every update the sample or its exchange has a wrong held-out answer
(`mem_wrongAt_or`). Therefore:

| Statement | Declaration |
| --- | --- |
| At every update, P(some held-out answer is wrong) ≥ ½ P(uncovered split) > 0 | `uniformPerm_uncovered_le` |
| A condition on the split that makes the target true excludes every uncovered split | `not_mem_of_mem_uncovered` |
| The almost sure target fails, for Adam and for AdamW at every weight decay | `not_division_generalizes` |
| The held-out accuracy does not tend to 1 in probability either | `not_division_tendsToOneInProbability` |

This does not say that grokking fails. It says that "for almost every run" is too much: a
true theorem needs a condition on the split, or a probability below 1. The argument works for
any two residues, so the condition must at least forbid that two residues are both absent
from the training set. Whether "every residue occurs in training" suffices is open; so is
every positive statement.

## 3. What the repository proves, and about what

| Where | Declarations | It is about | Why it is not the target |
| --- | --- | --- | --- |
| `FlatSpectrum/Training` | `decayRun`, `IsDecayedMin`, `exists_tendsto_decayRun_ceLoss_isDecayedMin`, `exists_pos_ae_tendsto_decayRun_ceLoss_isDecayedMin` | Gradient descent on a product `U Vᴴ` of complex factors with the penalty `λ(‖U‖² + ‖V‖²)`, `λ > 0`: for almost every start the run converges to a minimum | Another model (no embedding, attention, normalization or ReLU), another optimizer (Euclidean descent, coupled penalty), and `λ > 0` |
| `FlatSpectrum/Training` | `tendsto_margin_div_nuclearNorm`, `margin_eq_maxMargin_of_tendsto_inv_smul`, `tendsto_inv_smul_of_unique` (#108), `tendsto_inv_smul_codeProbe_iff`, `tendsto_inv_smul_cubicKernel_iff` (#109) | The directions of the minima as `λ → 0⁺`: a sequence of separately optimized fits | A limit over problems, not along one trajectory; with `λ = 0` there is no minimum at all |
| `FlatSpectrum/Training/NoDecay` | `not_exists_forall_ceLoss_le`, `not_isDecayedMin_ceLoss_zero`, `not_isCompact_setOf_flatLoss_zero_le` | On separable data the undecayed cross-entropy has no minimizer and unbounded sublevel sets | It removes the compactness that every convergence proof above uses; it says nothing about where the run goes (#118 is open) |
| `FlatSpectrum/Training/UnseenRows` | `IsDecayedMin.row_eq_zero`, `IsDecayedMin.col_eq_zero`, `IsDecayedMin.ceLoss_heldOut_eq` (#110) | A decayed minimum is zero on unseen rows and columns of the product | A property of that model's minima; the decoder has no such product |
| `FlatSpectrum` | `isMinOn_nuclearNorm_iff`, `IsFlat.exists_neg_mul_of_isMinOn_cell`, `IsDecayedMin.ne_smul_of_isFlat` | A flat kernel is not recovered from a proper subset of its cells by the least nuclear norm | Failure to reconstruct a kernel matrix is not failure to classify: the target needs only the largest logit, and the decoder is not a least-nuclear-norm interpolant (#124 is open) |
| `Grokking/Operational` | `FirstCrossing`, `FirstSustained`, `firstCrossing_le_firstSustained`, `finite_prefix_future_ambiguity`, `no_universal_finite_crossing_detector` | Predicates of an observed accuracy trace | They describe records of a run and prove that a finite prefix does not decide the future |
| `Grokking/DivisionOrbits` | `InDivisionCell`, `equal_quotients_iff_common_scale`, `mem_full_cell_iff`, `modulus97_domain_card` | The algebra of the task: equal answers are common scalings | No model in it; `inDivisionCell_operands` ties the tokenized equations to it |
| `Landscape/Neural` | `crossEntropy`, `crossEntropy_nonneg`, `analyticAt_crossEntropy`, `exists_compact_rotate90_adamw_container` | The loss, reused here; analyticity and an AdamW box for the Rotate90 model (SiLU, RMS norm, tied readout) | Another architecture and schedule; the paper's decoder has ReLU and is not analytic |
| `Grokking/AdamW`, `EffectiveTheory`, `CircuitEfficiency`, `Composition`, `Geometry` | for example `coupled_first_growth_iff`, `quarticCE_every_rate_can_overshoot` | Effective models and diagnostics of learned circuits | None is the decoder's trajectory |
| `Grokking/Protocol` (`MarginTransfer`, `QuotientTransfer`, `QuotientWitness`) | `margin_sub_le_of_abs_sub_le`, `predict_eq_residueToken_of_anchor`, `predict_eq_residueToken_of_two_anchors`, `predict_bentLogits`, `predict_answerLogits_zero_ne` | Logits that move with the scalings of numerator and denominator up to an error carry two answers with margins above twice the error, one for each orbit of the scalings, to all 9312 equations, over all 239 tokens; the decoder with every parameter zero has the symmetry and answers nothing | A sufficient structure for held-out correctness. No proof that training produces it. With one anchor for an orbit no trained decoder has it: at each of the 117 checkpoints measured in `experiments/grok_division`, in doubles and again in exact arithmetic on those scores, twice the error exceeds the margin of every training anchor on the equations with a nonzero numerator |
| `Grokking/Protocol` (`DiscreteLog`, `NormScale`, `SolverParams` to `SolverScale`, `OrbitAnchors`, `HeldOutCertificate`, `Enclosure`) | `solver`, `answerLogits_solver`, `exists_param_forall_predict`, `solver_certified`, `heldout_accuracy_eq_one_of_training_anchors`, `heldout_accuracy_eq_one_of_anchors`, `heldout_accuracy_solver`, `mem_anchored_iff`, `uniformPerm_anchored_pos`, `uniformPerm_compl_anchored_pos`, `exists_fit_training_wrong_heldout`, `predict_eq_of_enclosure`, `accuracy_eq_one_of_decided`, `enclosure_gives_certificate`, `heldout_accuracy_eq_one_of_enclosed_anchors` | The decoder itself, in exact reals: built parameters whose logits are a positive multiple of the mark of the answer, so that all 9312 equations are answered; the certificate of held-out accuracy 1 on `answerLogits θ` from training answers, with an anchor for each held-out equation or one for each orbit, which those parameters satisfy with the error `1/52200`; the splits with an anchor in each orbit, which are exactly those that give every held-out equation a training equation in its orbit; parameters that fit all 4656 training answers of a split and miss a held-out one; and what two rows that enclose the logits decide about them: the answers, the accuracy, and the premise of the certificate from a test on the ends | The parameters are built, not reached by the optimizer: this shows that the answers and the structure are in the range of the architecture, and that training accuracy does not force them. Nothing here says that a run arrives at them. At five checkpoints of `experiments/grok_division` every held-out equation passes that test on enclosures of the real-valued logits, which a program computes with every operation rounded outward. Lean proves what the ends give and takes the enclosure as a hypothesis: this is an instance of the theorem up to that computation, and not a Lean theorem about a checkpoint |

Two substitutions are therefore excluded, and no statement here makes them:

- Convergence of decayed matrix-factorization minima, or of their directions, is not
  convergence of the undecayed transformer trajectory under Adam.
- The flat-kernel obstruction to reconstruction is not a proof that held-out classification
  fails, and its absence would not be a proof that it succeeds.

## 4. The bridges that are missing

1. **Arithmetic.** The target is in exact reals; every run is in `f32`, where the softmax
   saturates (PROTOCOL.md, Amendment 1). No statement relates the two trajectories. What is
   bounded is one forward pass: at the parameters of a checkpoint, which are reals exactly, the
   logits of the decoder in exact reals are enclosed (bridge 5), so its answers there are
   known. The parameters are still those of an `f32` trajectory.
2. **Measurability.** `TendsToOneAlmostSurely.inProbability` and `.inMean` need the accuracy to
   be measurable in the sample. This is not proved for the decoder.
3. **Exceptional parameters.** The source's backward pass has a rule at hidden inputs equal to
   0, and `sourceGrad` is that rule: the derivative of the loss with the gates frozen at the
   current parameters (`gate`, `frozenLoss_self`). Not formalized: that where no hidden input
   is 0 this is the partial derivative of `batchLoss` (the gates are then locally constant),
   and that the trajectory almost surely avoids the other parameters.
4. **Any control of the predictions.** Without decay the loss has no minimizer on fitted data
   and the parameters need not stay bounded. What is proved is how far a parameter can move:
   at most `0.017 t` under Adam, and within `max(|θ₀|, 17)` of 0 under AdamW with weight decay 1.
   Two consequences are proved for the runs of the protocol themselves, with their batches,
   both moments and the gradient of the backward pass (section 5): a trajectory of Adam
   without weight decay whose parameters converge does not keep its batches fitted with a
   margin (`division_eventually_margin_lt`), and for almost every run of AdamW the logits
   stay bounded, so the loss of a batch stays above a positive floor and no margin grows
   without bound (`ae_abs_logits_divisionRun_le`, `ae_floor_le_batchLoss_divisionRun`,
   `ae_margin_divisionRun_le`). No descent, no bound of the Adam run uniform in `t`, no limit
   of a trajectory or of its logits, and no control of a held-out prediction is proved or
   assumed.
5. **Structure gives correct answers** (#129): proved, as a sufficient condition.
   `heldout_accuracy_eq_one_of_training_anchors` gives held-out accuracy 1 when every held-out
   equation has a training anchor whose margin is above twice the error of the equation's
   relabelled logits against the anchor's; with one anchor for each orbit it is
   `heldout_accuracy_eq_one_of_anchors`. `solver` satisfies it (`heldout_accuracy_solver`); a
   split gives every held-out equation a training equation in its orbit exactly when it trains
   on an equation of each orbit (`mem_anchored_iff`), which a uniformly random split does with
   positive probability and fails to do with positive probability (the failure, all 96
   equations with numerator zero held out, has probability `∏_{j<96} (4656 − j)/(9312 − j)`,
   about `7.7 · 10⁻³⁰`, by a count outside Lean); and fitting every training answer does not
   suffice (`exists_fit_training_wrong_heldout`). What this leaves: no trained parameters are
   proved to meet the condition. Measured in `experiments/grok_division` on the scores of the
   model in doubles, compared in exact arithmetic: with one anchor for the orbit of nonzero
   numerators no checkpoint of 117 meets it; with an anchor for each held-out equation five of
   the nine checkpoints that answer every equation do, with 0.31 to spare at the least, and
   the other four answer every equation without it, so it is not necessary. The rounding of
   that forward pass is bounded by a computation outside Lean. `quotient --enclose` computes,
   for every logit of the decoder in exact reals at a checkpoint's parameters, two doubles that
   it lies between, with every operation rounded outward
   (`experiments/grok_division/src/enclosure.rs`). On those ends every answer is decided at all
   117 checkpoints, with the counts of the model in doubles, and at the same five checkpoints
   every held-out equation passes the test of `enclosure_gives_certificate`, with 0.31 to spare
   at the least; `heldout_accuracy_eq_one_of_enclosed_anchors` turns that test, for any two
   rows that enclose the logits, into held-out accuracy 1. That the printed ends enclose the
   logits rests on that file and the processor's arithmetic; Lean takes it as a hypothesis.
6. **Training reaches the structure** (#130): from the declared initialization law, with a
   stated probability. Open, and the measured runs give no support for the half that asks the
   structure to be kept. The premise of 5 is met at five retained checkpoints of two AdamW runs
   and lost again in both: `adamw`, seed 0, has it at updates 30 000 and 40 000 and 2635 wrong
   held-out answers at 50 000; `adamw --answer-only`, seed 1, has it at 20 000, 50 000 and
   60 000 and 4154 wrong at 70 000. No checkpoint of the Adam runs has it. Whether the loss of
   accuracy under AdamW is the recipe's or this transcription's is not known: the trainer was
   not compared with a PyTorch run.
7. **Composition** (#131): 5 and 6 along one trajectory, for all later updates, for a target
   with a condition on the split; the unconditional one is refuted in section 2.

Bridges 4, 6 and 7 are open research problems. The measurements of
`experiments/grok_division` are finite runs in `f32` and close none of them.

## 5. Ways to bridge 6: what the runs exclude and what stays open

`Protocol/{ReadoutBound,TargetMean,ReadoutGradient,ReadoutPairing,AdamStationary,ReadoutStationary,ReadoutWitness,DecayedLogits}.lean`.

Bridge 6 (#130) asks that training reaches the structure of bridge 5 and keeps it. Each row
is a way a proof could go and what is known of it. A row that is closed is closed for that way
only: nothing here is evidence that the target with a condition on the split is true or false,
and #130 stays open.

| Way | Status | Declarations or evidence |
| --- | --- | --- |
| Adam without weight decay: the parameters converge, and the limit is analysed | Closed where the run fits its batches with a margin. If every parameter of a trajectory converges, then for every `μ > 0`, from some update on, every batch has an equation whose answer or closing token leads by less than `μ` at the parameters the batch is seen at. A readout coordinate that converges sees gradients that tend to 0, whatever the batches and without continuity of the gradient; the readout paired with its gradient is the mean of the expected logit less the logit of the target, which a margin keeps below a negative number while the logits are bounded. Not excluded: a trajectory that converges while some target of its batches stays tied or wrong | `tendsto_adamRun_grad_zero`, `sourceGrad_out`, `sum_out_mul_sourceGrad`, `sum_out_mul_sourceGrad_le_of_margin`, `eventually_exists_margin_lt`, `division_eventually_margin_lt`; the witness of the hypotheses is the run from zero, `divisionRun_zeroSample` |
| AdamW: the training loss tends to 0, or the margins grow without bound | Closed for almost every run. With weight decay 1 the readout and the second normalizations stay within 17 of zero, so every logit stays within `logitBox`, which is below 455 600; the loss of every batch with a target stays above `log(1 + e^(−2·logitBox))`, which is positive, and no answer leads by more than `2·logitBox`, at every update. The bound is far from what a run shows; what it gives is uniformity in the update | `abs_logits_le`, `ae_initMeasure_abs_le_one`, `ae_abs_logits_divisionRun_le`, `ae_floor_le_batchLoss_divisionRun`, `ae_margin_divisionRun_le`, `logitBox_lt`, `floor_pos` |
| Transfer from the minima of a weight-decayed factorization | Excluded in section 3: another model and another optimizer, no minimum without decay, and a limit of AdamW is not a minimizer of the weight-decayed loss | `not_exists_forall_ceLoss_le`, `not_tendsto_adamRun_factor` |
| A checkpoint with the certificate, then invariance of the certificate along the run | No support. On enclosures of the real-valued logits the certificate holds at five retained checkpoints of two AdamW runs and is lost again in both; no Adam checkpoint has it. Those trajectories are in `f32`; nothing is known of the exact-real one | bridge 5; `experiments/grok_division/README.md` |
| Adam without weight decay: the parameters diverge and their direction converges to a margin maximizer | Open, not attempted. The decoder is not written as a positively homogeneous function of its parameters (the normalization adds `10⁻⁵` to the variance, the positions are added to the embedding); no homogeneity is proved here, and an argument for homogeneous networks would need it. The first row proves non-convergence only: it does not prove that any parameter is unbounded | none |
| AdamW: a stationary regime inside the box, described in law | Open, not attempted. It needs the measurability of bridge 2 first | none |

## 6. What can be claimed today (#131, #132)

There is no end-to-end theorem, so #131 is open and #132 has no final claim to audit. An audit
of the present state finds this:

- **Exact-real, proved in Lean with the three standard axioms.** What the target says
  (`division_generalizes_iff`); that the almost sure target and the target in probability are
  false without a condition on the split (section 2); that a structure of the logits gives
  held-out accuracy 1 and that built parameters have it (bridge 5); what two rows that enclose
  the logits decide (bridge 5); and the two exclusions of section 5. None of these says that
  a trained decoder generalizes.
- **Finite-horizon, native.** Nine runs in `f32` (three settings, seeds 0 to 2) and their 117
  retained checkpoints. At a checkpoint the parameters are reals exactly, and the logits of
  the exact-real decoder there are enclosed by a computation outside Lean; this decides the
  answers of that decoder at those parameters and nothing after the last checkpoint. The
  `f32` trajectory is not related to the exact-real trajectory by any statement (bridge 1), so
  no theorem of section 5 is a statement about an executed run, and no run is an instance of
  the target.
- **Not addressed.** Every other curve and claim of the paper; the time at which accuracy
  rises; whether the certificate is kept. The trainer was not compared with a PyTorch run.
