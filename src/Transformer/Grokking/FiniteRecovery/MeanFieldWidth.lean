module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Coupling
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Sampling
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Residual
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Grid
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.ErrorBound
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Transfer
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Bounds
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Certificate
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Handoff
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Recovery
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Tail
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Literal

/-! A mean-field approximation of AdamW training in width. The particles
are width units carrying their weights and both moment buffers; they
interact through finitely many empirical averages. `Coupling` bounds the
distance of the N-particle system to N coupled limiting particles.
`Sampling` bounds the expected sampling error of those limiting particles
by C/√N under the product of the initialization laws. `Residual` and
`Grid` bound the gap between the residual of a mean-field certificate and
its limiting counterpart pathwise; `ErrorBound` and `Transfer` give the
explicit error `widthError` for its expected maximum over updates and
fractions, and its vanishing in width at a fixed horizon. `Bounds` keeps weights and
displacements in a box fixed by the initial weights, and `Certificate`
identifies, by the chain rule, the recovery residual of a certificate
F ∘ mean ψ with the `residual` of the transfer theorem. `Handoff` turns the
majorant into the excess of the recovery theorem, and `Recovery` gives
permanent recovery once max(r̄, 0) + widthError + E[tail] is below the
recovery margin. `Tail` gives that allowance after the horizon from a
contraction of the coupled distance and run-wise bounds on the later
fluctuations and segment errors. `Literal` identifies the literal
`AdamW.run` with floor 10⁻⁸, clipping and decoupled weight decay on N
width units with the system of floor N · 10⁻⁸ while clipping does not act. -/
