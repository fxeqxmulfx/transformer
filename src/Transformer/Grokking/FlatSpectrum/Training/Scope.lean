module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.LinearDiverge

/-! The scope of the training theorems, `review/flat_spectrum.md`, the items "Assumed or not
covered" of WD11 (*Training reaches the optimal fit*) and XE13 (*Training with
cross-entropy*): for each hypothesis listed there, a counterexample that shows it cannot be
dropped, or the part of the open statement that is proved.

* A loss that is not bounded below (WD11, WD12). For the linear loss `−Re tr X` with a weight
  decay `λ < 1` the run from every start with `U₀ + V₀ ≠ 0` has factors whose norm tends to
  infinity, at every step size, so it has no limit (`not_tendsto_decayRun_linear`), and no
  factorization is a decayed minimum (`not_isDecayedMin_linear`). Convergence is therefore a
  hypothesis of WD5 that the other two, full row rank and positive definite step matrices, do
  not imply. -/
