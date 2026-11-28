module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Coupling
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Sampling
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Residual
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Grid
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.ErrorBound
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Transfer

/-! A mean-field approximation of AdamW training in width. The particles
are width units carrying their weights and both moment buffers; they
interact through finitely many empirical averages. `Coupling` bounds the
distance of the N-particle system to N coupled limiting particles.
`Sampling` bounds the expected sampling error of those limiting particles
by C/√N under the product of the initialization laws. `Residual` and
`Grid` bound the gap between the residual of a mean-field certificate and
its limiting counterpart pathwise; `ErrorBound` and `Transfer` give the
explicit error `widthError` for its expected maximum over updates and
fractions, and its vanishing in width at a fixed horizon. -/
