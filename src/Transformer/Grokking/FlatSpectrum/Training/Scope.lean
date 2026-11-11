module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.LinearDiverge
public import Transformer.Grokking.FlatSpectrum.Training.Scope.MarginContinuity
public import Transformer.Grokking.FlatSpectrum.Training.Scope.NoDecayRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PhaseStep
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PhasedDirection
public import Transformer.Grokking.FlatSpectrum.Training.Scope.PhasedRun
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSegment
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSegmentCore
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SignSet
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SmoothLoss
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogit
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogitBall
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogitRun
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
  fixed step, and the direction of its factors, are not proved.
* One run with a decreasing weight decay (XE13, XE18). Let the weight decay pass through a
  positive sequence `λ_j → 0`, phase `j` making `T j` steps with `λ_j` and the step of XE3 for
  the ball of its start (`phaseRun`, `phaseStep`). From a start of full row rank there are
  lengths `T j` for which, at the phase ends, the nuclear norm of the product tends to
  infinity and its margin for the nuclear norm to the largest margin
  (`exists_tendsto_margin_div_phaseProduct`). So the products at the phase ends converge in
  direction to the flat kernel when every cell is seen, for a mirrored code
  (`exists_tendsto_inv_smul_phaseProduct_codeProbe`), and they answer the unseen cell of the
  kernel of signs with the wrong class (`exists_eventually_phaseProduct_unseen_wrong`). The
  lengths of the phases are not explicit, and the margin is not followed inside a phase. -/
