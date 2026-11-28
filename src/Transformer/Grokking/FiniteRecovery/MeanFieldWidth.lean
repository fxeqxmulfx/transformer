module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Coupling
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Sampling
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Residual
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Grid
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.ErrorBound
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Transfer
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Bounds
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Certificate
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Recovery
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Literal
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Obstruction

/-! A mean-field approximation of AdamW training in width. The particles
are width units carrying their weights and both moment buffers; they
interact through finitely many empirical averages.

* `Coupling` bounds the distance of the N-particle system to N coupled
  limiting particles, and `Sampling` the expected sampling error of those
  limiting particles by C/√N under the product of the initialization laws.
* `Residual` and `Grid` bound the gap between the residual of a mean-field
  certificate and its limiting counterpart pathwise; `ErrorBound` and
  `Transfer` give the explicit error `widthError` for its expected maximum
  over updates and fractions, its vanishing in width at a fixed horizon,
  and the excess that the recovery theorem consumes.
* `Bounds` keeps weights and displacements in a box fixed by the initial
  weights, and `Certificate` identifies, by the chain rule, the recovery
  residual of a certificate F ∘ mean ψ with the `residual` of the transfer
  theorem.
* `Recovery` gives the tail allowance after the horizon from a contraction
  of the coupled distance and run-wise bounds on the later fluctuations and
  segment errors, and permanent recovery once
  max(r̄, 0) + widthError + E[tail] is below the recovery margin.
* `Literal` identifies the literal `AdamW.run` with floor 10⁻⁸, clipping
  and decoupled weight decay on N width units with the system of floor
  N · 10⁻⁸ while clipping does not act.
* `Obstruction` evaluates the bound at the division width 64: one Gronwall
  step at the peak rate is at least 56251, so the error is vacuous there,
  and the factor 1/eps is intrinsic to the first AdamW direction. -/
