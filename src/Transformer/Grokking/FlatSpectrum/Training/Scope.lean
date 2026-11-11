module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.LinearDiverge
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SmoothLoss
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogit
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogitBall
public import Transformer.Grokking.FlatSpectrum.Training.Scope.SquaredLogitRun

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
  (`exists_tendsto_decayRun_sqLogit_not_isDecayedMin`). -/
