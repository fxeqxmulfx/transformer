module

public import Transformer.Grokking.FlatSpectrum.Training.Defs
public import Transformer.Grokking.FlatSpectrum.Training.Basic
public import Transformer.Grokking.FlatSpectrum.Training.Lifted
public import Transformer.Grokking.FlatSpectrum.Training.Certificate
public import Transformer.Grokking.FlatSpectrum.Training.Gain
public import Transformer.Grokking.FlatSpectrum.Training.Growth
public import Transformer.Grokking.FlatSpectrum.Training.Continuity
public import Transformer.Grokking.FlatSpectrum.Training.Gradient
public import Transformer.Grokking.FlatSpectrum.Training.Limit
public import Transformer.Grokking.FlatSpectrum.Training.LinearRun
public import Transformer.Grokking.FlatSpectrum.Training.Convergence
public import Transformer.Grokking.FlatSpectrum.Training.SquaredLoss
public import Transformer.Grokking.FlatSpectrum.Training.Vanishing
public import Transformer.Grokking.FlatSpectrum.Training.Recovery
public import Transformer.Grokking.FlatSpectrum.Training.FactorSpace
public import Transformer.Grokking.FlatSpectrum.Training.SquaredBounds
public import Transformer.Grokking.FlatSpectrum.Training.StepSize
public import Transformer.Grokking.FlatSpectrum.Training.FlatLoss
public import Transformer.Grokking.FlatSpectrum.Training.Descent
public import Transformer.Grokking.FlatSpectrum.Training.Convergent
public import Transformer.Grokking.FlatSpectrum.Training.SquaredRun
public import Transformer.Grokking.FlatSpectrum.Training.Sublevel
public import Transformer.Grokking.FlatSpectrum.Training.SmallSteps
public import Transformer.Grokking.FlatSpectrum.Training.ExpLoss
public import Transformer.Grokking.FlatSpectrum.Training.Deficient
public import Transformer.Grokking.FlatSpectrum.Training.FullColumn
public import Transformer.Grokking.FlatSpectrum.Training.Width
public import Transformer.Grokking.FlatSpectrum.Training.AlmostEvery
public import Transformer.Grokking.FlatSpectrum.Training.WidthOne
public import Transformer.Grokking.FlatSpectrum.Training.AnyShape
public import Transformer.Grokking.FlatSpectrum.Training.Logits
public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropy
public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyBounds
public import Transformer.Grokking.FlatSpectrum.Training.CrossEntropyRun
public import Transformer.Grokking.FlatSpectrum.Training.Margin
public import Transformer.Grokking.FlatSpectrum.Training.MaxMargin
public import Transformer.Grokking.FlatSpectrum.Training.MaxMarginLimit
public import Transformer.Grokking.FlatSpectrum.Training.SeenCells
public import Transformer.Grokking.FlatSpectrum.Training.AnswerCode
public import Transformer.Grokking.FlatSpectrum.Training.SeenMargin
public import Transformer.Grokking.FlatSpectrum.Training.UnitBall
public import Transformer.Grokking.FlatSpectrum.Training.MarginDirection
public import Transformer.Grokking.FlatSpectrum.Training.LeastNorm
public import Transformer.Grokking.FlatSpectrum.Training.MirroredCode
public import Transformer.Grokking.FlatSpectrum.Training.CharacterCode
public import Transformer.Grokking.FlatSpectrum.Training.MirroredUnique
public import Transformer.Grokking.FlatSpectrum.Training.SeenDirection
public import Transformer.Grokking.FlatSpectrum.Training.UnseenRows
public import Transformer.Grokking.FlatSpectrum.Training.LargeDecay
public import Transformer.Grokking.FlatSpectrum.Training.NearestCodeword
public import Transformer.Grokking.FlatSpectrum.Training.ClassProbes
public import Transformer.Grokking.FlatSpectrum.Training.MeanLoss
public import Transformer.Grokking.FlatSpectrum.Training.Subsequence
public import Transformer.Grokking.FlatSpectrum.Training.FactorSign
public import Transformer.Grokking.FlatSpectrum.Training.NuclearMin
public import Transformer.Grokking.FlatSpectrum.Training.NuclearMinLimit
public import Transformer.Grokking.FlatSpectrum.Training.NuclearMinDirection
public import Transformer.Grokking.FlatSpectrum.Training.NoDecay
public import Transformer.Grokking.FlatSpectrum.Training.QuarterCode
public import Transformer.Grokking.FlatSpectrum.Training.SignPath
public import Transformer.Grokking.FlatSpectrum.Training.SignKernel
public import Transformer.Grokking.FlatSpectrum.Training.SignDirection

/-! Training the pure factorization with weight decay, `review/flat_spectrum.md`, Lemma 1's
modeling step and *Training reaches the optimal fit* (added).

The model is `X = U Vᴴ` with the weight-decayed loss `L (U Vᴴ) + λ · ½ (‖U‖² + ‖V‖²)` for a
convex `L` with gradient `G`, trained by gradient descent with step `η` (`decayRun`). With the
factors stacked, `W = [U; V]`, a step is `W ← (1 − η M) W` for the Hermitian
`M = [[λ, G], [Gᴴ, λ]]` (WD1). A stationary point, `M W = 0`, with `M` positive semidefinite
has the least weight-decayed loss among the factorizations of every width, and then `U Vᴴ`
minimizes `L + λ ‖·‖_S1` with weight decay equal to `‖U Vᴴ‖_S1` (WD2). A continuous `G` with
the first-order inequality `L X + ⟨G X, Y − X⟩ ≤ L Y` is the gradient of `L`, and the update is
the gradient step of the weight-decayed loss (WD3). A bounded run of
`W ← (1 − η M_t) W` from a start of full row rank, with positive definite steps and `M_t → M∞`,
has `M∞` positive semidefinite: along a negative direction of `M∞` a column of `W_t` would grow
geometrically (WD4). So a convergent run of gradient descent with weight decay, from stacked
factors of full row rank and with positive definite step matrices, ends at a factorization of
least weight-decayed loss among those of every width: it cannot end at a saddle or at a local
minimum that is not global (WD5). The linear loss `−Re X` of a `1 × 1` matrix gives a run that
satisfies every hypothesis and converges to a nonzero limit.

For the squared loss on the seen cells `Ω` of a kernel `K` (WD7), a factorization of least
weight-decayed loss has nuclear norm at most that of every matrix equal to `K` on `Ω` and loss
at most `λ` times that norm. So as `λ → 0⁺` the limits of such trained fits are the optimal fits
of Lemma 1 (`IsOptimalFit`); if `K` does not have the least nuclear norm among the matrices
equal to it on `Ω`, as for a square kernel with equal singular values and no zero entry on a
proper `Ω` (Theorem A), the trained products stay a fixed distance below `‖K‖_S1` for every
`λ > 0`; and if `K` is the only matrix of least nuclear norm, as for `u vᵀ` with `|u_i v_j| = 1`
on connected cells containing a permutation (Theorem E), the trained products converge to `K`.
Trained fits of one width always have a subsequence whose factors converge, to an optimal fit
(`exists_subseq_tendsto_isOptimalFit`); the factors themselves need not converge
(`exists_seq_isDecayedMin_not_tendsto`). The `1 × 1` kernel `1` has the trained product `1 − λ`.

The step condition is explicit when the loss is controlled in a ball (WD6): if `‖G X‖ ≤ γ` and
`⟨G X, X⟩ ≥ −b` for `‖X‖ ≤ ρ/2`, with `4 b ≤ λ ρ` and `2 η (λ² + γ²) ≤ λ`, a run from the ball
`‖U₀‖² + ‖V₀‖² ≤ ρ` stays in it and every step matrix is positive definite, so a convergent run
from full row rank ends at a factorization of least weight-decayed loss. The squared loss with
`‖K‖ ≤ κ` has `γ = ρ/2 + κ` and `b = κ²/4`.

Convergence itself is proved (WD8) for a loss that is real-analytic in the entries and lies
below its quadratic model with a constant `s` in that ball. Listing the entries of `W = [U; V]`
turns the pairs of factors into the real inner product space `EuclideanSpace ℂ ((m ⊕ n) × k)`
(`factors`, `flatten`), on which the weight-decayed loss is real-analytic (`analyticAt_flatLoss`)
with gradient `M W` (`gradient_flatLoss`), so the run is gradient descent there
(`factors_iterate_gradientStep`). A step with `η (λ + γ + 2 s ρ) ≤ 1` lowers the loss by
`(η/2) ‖M W‖²` (`decayedLoss_decayStep_add_le_of_ball`), and the ball is compact, so the run has
a cluster point; Łojasiewicz's theorem and the convergence theorem of
`Transformer.Landscape.SaddleAvoidance` make it the limit. So the run converges from every
start in the ball, of any width and rank, to a stationary point (`exists_tendsto_decayRun`),
and from a start of full row rank to a factorization of least weight-decayed loss
(`exists_tendsto_decayRun_isDecayedMin`). The squared loss has `s = 1`
(`exists_tendsto_decayRun_fitLoss_isDecayedMin`), so such a factorization of width `m + n`
exists for every `λ > 0` (`exists_isDecayedMin_fitLoss`). For a square kernel with equal
singular values and no zero entry on a proper `Ω`, every such run converges a fixed distance
below `‖K‖_S1` (`exists_tendsto_decayRun_nuclearNorm_add_le`).

Without those constants small steps still converge (WD9). For a loss that is real-analytic in
the entries and bounded below, the sublevel sets of the weight-decayed loss are compact on the
factor space (`isCompact_setOf_flatLoss_le`), and the sublevel theorems of
`Transformer.Landscape.SaddleAvoidance` give a step threshold (`exists_pos_forall_sublevel`)
below which every run from the sublevel set converges to a stationary point
(`exists_pos_tendsto_decayRun`), and from full row rank to a factorization of least
weight-decayed loss (`exists_pos_tendsto_decayRun_isDecayedMin`). So such a factorization of
width `m + n` exists for every such loss and every `λ > 0` (`exists_isDecayedMin`). The
exponential loss `Σ exp (Re X_ij)` is an instance (`exists_isDecayedMin_expLoss`).

Full row rank needs width `m + n`; for almost every start the width `min(m, n)` suffices
(WD10). If `[U; V] z = 0` for some `z ≠ 0`, the Hessian of the weight-decayed loss along
`S zᴴ` is `‖z‖² ⟨S, M S⟩` (`inner_fderiv_gradient_flatLoss`), so a critical point with
dependent columns that is not a strict saddle has `M` positive semidefinite
(`isDecayedMin_of_not_isStrictSaddle`). At a stationary point with independent columns and
width at least `min(m, n)`, `G Gᴴ = λ²` or `Gᴴ G = λ²`, and `M` is positive semidefinite again
(`lift_posSemidef_of_injective`). So at these widths a critical point is a decayed minimum or
a strict saddle (`isDecayedMin_of_not_isStrictSaddle_of_card_le`), no local minimum is
spurious (`isDecayedMin_of_isLocalMin_of_card_le`), small steps from almost every start of a
sublevel set converge to a decayed minimum (`exists_pos_ae_tendsto_decayRun_isDecayedMin`),
and a decayed minimum of width `min(m, n)` exists (`exists_isDecayedMin_of_card_le`). For a
`p × p` kernel with equal singular values and no zero entry on a proper `Ω`, almost every such
run of width `p` converges a fixed distance below `‖K‖_S1`
(`exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le`). Below the width `min(m, n)` the claim
fails: for the squared loss on all cells of the `2 × 2` identity and `0 ≤ λ < 1` no
factorization of width one is a decayed minimum (`not_isDecayedMin_width_one`).

Any shape (`AnyShape`, item R4 of *Flat kernels of any shape*): the statements above about a
square kernel with equal singular values use only that `K` is not a minimum of M from `Ω`. They
hold for a flat kernel of any shape, `Kᴴ K = c I` or `K Kᴴ = c I`, when some nonzero cell is
unseen (`IsFlat.exists_nuclearNorm_add_le`, `IsFlat.exists_tendsto_decayRun_nuclearNorm_add_le`,
`IsFlat.exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le`), and with every cell seen a
factorization of width `k` keeps the squared loss `c (min(m, n) − k) / 2`
(`IsFlat.mul_sub_card_le_fitLoss_univ`).

Cross-entropy (*Training with cross-entropy*, items XE1–XE12). A sample `s` reads the product
through probes, with the logit `⟨A s c, X⟩` of the class `c`; the loss is the cross-entropy of
the labels (`ceLoss`, with the gradient `ceGrad`).

* XE1–XE3. The loss is convex, nonnegative and real-analytic, and for probes of norm at most
  `a` it has the constants `γ = 2 a |S|`, `b = |S| log |C|`, `s = |S| a²` of WD6 and WD8. Under
  explicit bounds on the step the run converges from every start, from full row rank to a
  decayed minimum (`exists_tendsto_decayRun_ceLoss_isDecayedMin`). The mean of the losses is a
  rescaling (`decayRun_smul`), and `classProbe` gives a table of logits.
* XE4–XE6. The margin bounds the loss on both sides. The gaps of a trained fit are within
  `log (2 |S| (|C| − 1))` of the margin of a competitor at its norm. As `λ → 0⁺` on separable
  samples the norm of the trained fits tends to infinity and their normalized margin to the
  largest margin `μ*` of the unit ball (`tendsto_margin_div_nuclearNorm`); the least norm at
  margin one is `1 / μ*` (`isLeast_nuclearNorm_of_one_le_margin`). At `λ = 0` such samples
  have no minimum and no trained fit (`not_isDecayedMin_ceLoss_zero`).
* XE7, XE12. For a loss that reads only the seen cells, a trained product is a minimum of M
  for its own values, so no nonzero multiple of a kernel that M does not return; it is zero on
  every row and column without a seen cell (`IsDecayedMin.row_eq_zero`), and `0` for a weight
  decay above `‖G 0‖` (`IsDecayedMin.mul_eq_zero_of_lt`).
* XE8, XE9. For the probes `φ(c) e_ij` of an answer code the larger logit belongs to the
  nearer codeword (`pairing_codeProbe_le_iff`). The trained fits pass the margin of a flat
  kernel with a nonzero unseen cell that shows the labels, and do not converge in direction to
  it (`not_tendsto_inv_nuclearNorm_smul_of_codeProbe`).
* XE10, XE11. The largest margin is attained (`exists_margin_eq_maxMargin`); every limit point
  of `X_j / ‖X_j‖_S1` has it, and the direction converges when it is attained once
  (`tendsto_inv_smul_of_unique`). For a code mirrored at a gap `0 < g < 2` (`IsMirroredCode`;
  `ψ` for `p ≥ 3`) the trained fits converge in direction to a flat kernel that shows the
  labels exactly when every cell is seen (`tendsto_inv_smul_codeProbe_iff`,
  `tendsto_inv_smul_cubicKernel_iff`), and scaled to margin one to `K / g`. For the code
  `(1, i)`, which is not mirrored, the direction on one seen cell converges to another
  product (`not_tendsto_inv_smul_quarterCode`). For the code `±1`, where `g = 2`, the multiples
  of a flat kernel of signs are trained products (`isNuclearMin_smul_signCode`), and the
  direction of some trained fits does not converge
  (`exists_seq_isDecayedMin_not_tendsto_signKernel`).

The minimizers of `L + λ ‖·‖_S1` (`IsNuclearMin`) are the products of the trained fits of every
width (`isNuclearMin_iff`), so XE5, XE6 and XE9–XE11 hold for them
(`forall_seq_isNuclearMin`): the widths of the fits may change along a sequence. -/
