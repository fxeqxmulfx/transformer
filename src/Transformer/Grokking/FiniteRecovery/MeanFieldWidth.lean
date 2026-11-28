module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Coupling
public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Sampling

/-! A mean-field approximation of AdamW training in width. The particles
are width units carrying their weights and both moment buffers; they
interact through finitely many empirical averages. `Coupling` bounds the
distance of the N-particle system to N coupled limiting particles.
`Sampling` bounds the expected sampling error of those limiting particles
by C/√N under the product of the initialization laws. -/
