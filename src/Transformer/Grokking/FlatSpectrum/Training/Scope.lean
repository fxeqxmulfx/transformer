module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.Balanced
public import Transformer.Grokking.FlatSpectrum.Training.Scope.BalanceObstruction
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityLimit
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityMin
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityOrigin
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityOriginRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentityStationary
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentitySymmetric
public import Transformer.Grokking.FlatSpectrum.Training.Scope.IdentitySymmetricLimit
public import Transformer.Grokking.FlatSpectrum.Training.Scope.LinearDiverge
public import Transformer.Grokking.FlatSpectrum.Training.Scope.MarginContinuity
public import Transformer.Grokking.FlatSpectrum.Training.Scope.NegStart
public import Transformer.Grokking.FlatSpectrum.Training.Scope.NoDecayRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.OneByOne
public import Transformer.Grokking.FlatSpectrum.Training.Scope.OneByOneLimit
public import Transformer.Grokking.FlatSpectrum.Training.Scope.OneByOneRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PaddedRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PhaseStep
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PhasedDirection
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PhasedRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.RankStart
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignGap
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignRunLimit
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSegment
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSegmentCore
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSet
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSystem
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SmoothLoss
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogit
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogitBall
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogitRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.StackRank
public import Transformer.Grokking.FlatSpectrum.Training.Scope.TwoByTwo
public import Transformer.Grokking.FlatSpectrum.Training.Scope.UnseenCell
public import Transformer.Grokking.FlatSpectrum.Training.Scope.UnseenCellLimit
public import Transformer.Grokking.FlatSpectrum.Training.Scope.UnseenCellRun

/-! The scope of the training theorems, `review/flat_spectrum.md`, the items "Assumed or not
covered" of WD11 (*Training reaches the optimal fit*) and XE13 (*Training with
cross-entropy*): for each hypothesis listed there, a counterexample that shows it cannot be
dropped, or the part of the open statement that is proved.

* A loss that is not bounded below (WD11, WD12). For the linear loss `−Re tr X` with a weight
  decay `λ < 1` the run from every start with `U₀ + V₀ ≠ 0` has factors whose norm tends to
  infinity, at every step size, so it has no limit (`not_tendsto_decayRun_linear`), and no
  factorization is a decayed minimum (`not_isDecayedMin_linear`). Convergence is therefore a
  hypothesis of WD5 that the other two, full row rank and positive definite step matrices, do
  not imply.
* Logits that are not linear in the product (XE13, XE14). Convexity is not needed for the
  convergence of small steps: for a loss that is real-analytic in the entries and bounded
  below, with a continuous gradient, every run from a sublevel set converges to a stationary
  point (`exists_pos_tendsto_decayRun_of_analytic`). It is needed for the limit to be a
  decayed minimum: for the logits `±(Re x)²` of a `1 × 1` product (`sqLogitLoss`) every run
  from the ball `‖U₀‖² + ‖V₀‖² ≤ λ/4`, which has starts of full row rank, converges to the
  zero factorization, and that is not a decayed minimum
  (`exists_tendsto_decayRun_sqLogit_not_isDecayedMin`).
* An unseen cell of a seen row and a seen column (XE13, XE15). For the kernel of signs
  `[[1, 1], [−1, 1]]`, which is the kernel `ψ(a³ + a b)` for `p = 2`, with the code `±1` and
  every cell seen but `(1, 1)`, one product of the unit ball of the nuclear norm has the
  largest margin: `½ [[1, 1], [−1, −1]]`, of rank one (`maxMargin_threeCells`,
  `eq_signFill_of_margin`). The trained products converge in direction to it
  (`tendsto_inv_smul_threeCells`), so for every small weight decay they answer the unseen
  cell with the class of `−1`, where the kernel shows `+1` (`exists_pos_forall_unseen_wrong`),
  and so does the limit of the run of XE3 from a start of full row rank
  (`exists_pos_forall_tendsto_decayRun_unseen`, `exists_tendsto_decayRun_unseen_wrong`). The
  tool is the bound `‖Z‖_S2² + 2 |det Z| ≤ ‖Z‖_S1²` of a `2 × 2` matrix
  (`frobeniusSq_add_two_mul_norm_det_le`).
* What the weight decay leaves free for the code `±1` (XE13, XE16). With every cell seen once
  and `0 < λ ≤ √c`, a matrix is a trained product exactly when it has the real parts of
  `t(λ) K` and at most its nuclear norm (`isNuclearMin_signCode_iff`). For the kernel of
  signs the trained products are the segment `t(λ) K + i a [[1, −1], [1, 1]]`, `|a| ≤ t(λ)`
  (`isNuclearMin_signKernel_iff`), and the run of XE3 from a start of full row rank ends at a
  factorization of one of its points (`exists_tendsto_decayRun_signKernel`).
* No weight decay (XE13, XE17). On samples that some product classifies with a positive
  margin, the run with `λ = 0` from a start of full row rank has no limit when `2 a |S| η < 1`
  (`not_tendsto_decayRun_ceLoss_zero`), and it is outside every ball `‖U‖² + ‖V‖² ≤ ρ` with
  `η (2 a |S| + 2 |S| a² ρ) < 1` at infinitely many steps
  (`frequently_lt_frobeniusSq_decayRun_ceLoss_zero`). That its norm tends to infinity for one
  fixed step, and the direction of its factors, are proved for one sample only (XE19).
* No weight decay on one sample (XE13, XE19). For the sample of XE3 and XE17, a `1 × 1`
  product read with the code `±1`, a step multiplies `U + V` by `1 + η w` and `U − V` by
  `1 − η w`, with the weight `w = 2 / (1 + exp (2 Re x))` of the gradient
  (`decayRun_signProbe_add`, `decayRun_signProbe_sub`). So for every step `η > 0` and every
  start with `U₀ + V₀ ≠ 0`, at every width, the norm of the factors tends to infinity
  (`tendsto_frobeniusSq_decayRun_signProbe_atTop`), the cross-entropy tends to `0`
  (`tendsto_ceLoss_signProduct`), the margin for the nuclear norm tends to the largest margin
  (`tendsto_margin_div_signProduct`) and the direction of the product to the product of
  largest margin (`tendsto_inv_smul_signProduct`). From a start with `U₀ + V₀ = 0` the loss
  stays at least `log 2` (`log_two_le_ceLoss_signProduct`). More than one sample is open.
* One run with a decreasing weight decay (XE13, XE18). Let the weight decay pass through a
  positive sequence `λ_j → 0`, phase `j` making `T j` steps with `λ_j` and the step of XE3 for
  the ball of its start (`phaseRun`, `phaseStep`). From a start of full row rank there are
  lengths `T j` for which, at the phase ends, the nuclear norm of the product tends to
  infinity and its margin for the nuclear norm to the largest margin
  (`exists_tendsto_margin_div_phaseProduct`). So the products at the phase ends converge in
  direction to the flat kernel when every cell is seen, for a mirrored code
  (`exists_tendsto_inv_smul_phaseProduct_codeProbe`), and they answer the unseen cell of the
  kernel of signs with the wrong class (`exists_eventually_phaseProduct_unseen_wrong`). The
  lengths of the phases are not explicit, and the margin is not followed inside a phase.
* The exceptional starts at a narrow width (WD11, WD14). A run does not raise the rank of its
  stack (`rank_stack_decayRun_le`, `rank_mul_conjTranspose_le_of_tendsto`), so a start whose
  stack has a rank below that of every trained product reaches no decayed minimum
  (`not_isDecayedMin_of_rank_lt`). For the `2 × 2` identity these are the starts of rank at
  most one, at every width, and under the bounds of WD8 their runs converge to stationary
  points that are not decayed minima (`exists_tendsto_decayRun_one_not_isDecayedMin`). The
  run from a start with zero columns is the narrower run (`decayRun_fromCols_zero`). A
  stationary point with `λ ≠ 0` is balanced, `Uᴴ U = Vᴴ V`
  (`conjTranspose_mul_self_eq_of_stationary`), its weight decay is the nuclear norm of its
  product (`nuclearNorm_mul_conjTranspose_of_stationary`), and for `λ > 0` it is a decayed
  minimum exactly when its product is a trained product
  (`isDecayedMin_iff_isNuclearMin_of_stationary`). The exceptional set of WD10 contains more
  than the starts of low rank: for the `2 × 2` identity the run from `V₀ = −U₀` keeps
  `V_t = −U_t` (`decayRun_neg`) and reaches no decayed minimum
  (`not_isDecayedMin_one_of_neg`), though the stack `[U₀; −U₀]` has the rank of `U₀`
  (`rank_stack_neg`), which can be two. An exact description of that set is not given.
* The `1 × 1` identity (WD11, WD15). For the squared loss on the `1 × 1` identity,
  `0 < λ < 1` and the bounds of WD8, at every width, the run from a start in the ball
  converges, and its limit is a decayed minimum exactly when `U₀ + V₀ ≠ 0`
  (`exists_tendsto_decayRun_one_isDecayedMin_iff`): a step keeps the sum of the factors
  nonzero and does not decrease it near the origin (`frobeniusSq_add_decayStep_one`), and a
  stationary point other than the origin is a decayed minimum
  (`isDecayedMin_one_of_stationary`). For `λ ≥ 1` every run from the ball ends at a decayed
  minimum (`exists_tendsto_decayRun_one_isDecayedMin_of_one_le`). So here the exceptional
  starts are those of the subspace `U₀ + V₀ = 0` when `λ < 1` and there are none when `λ ≥ 1`,
  the step is explicit, and width one is included.
* The identity of any size (WD11, WD16). For the squared loss on all cells of the identity
  and `λ > 0`, a pair is stationary exactly when its factors are equal and
  `U Uᴴ U = (1 − λ) U` (`stationary_identity_iff`), so the product `X` of a stationary point
  is positive semidefinite with `X² = (1 − λ) X` (`mul_self_of_stationary_identity`): `1 − λ`
  times an orthogonal projection. For
  `0 ≤ λ ≤ 1` the only trained product is `(1 − λ) · 1` (`isNuclearMin_identity_iff`), a
  stationary point is a decayed minimum exactly when it has that product
  (`isDecayedMin_identity_iff_of_stationary`), and for `λ < 1` no factorization of a width
  below the size of the identity is one (`not_isDecayedMin_identity_of_card_lt`). Under the
  bounds of WD8 every run converges to a pair `(U, U)` of this kind
  (`exists_tendsto_decayRun_identity`), to the origin when `λ ≥ 1`
  (`tendsto_decayRun_identity_zero`) and from `V₀ = −U₀` (`tendsto_decayRun_identity_neg`).
  For `0 < λ < 1` the limit is the origin exactly when `U₀ + V₀ = 0`
  (`tendsto_decayRun_identity_zero_iff`): a step keeps the sum of the factors nonzero and
  does not decrease it near the origin (`frobeniusSq_add_decayStep_identity`). From a start
  `V₀ = U₀` the product `X` of the limit has `X U₀ = (1 − λ) U₀`
  (`exists_tendsto_decayRun_identity_self`), so the limit is a decayed minimum when `U₀` has
  a right inverse (`exists_tendsto_decayRun_identity_self_isDecayedMin`). Which projection a
  run from another start reaches, and so the exceptional starts of a kernel larger than
  `1 × 1`, are open. -/
