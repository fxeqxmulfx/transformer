module

public import Transformer.Grokking.FiniteRecovery.MeanFieldWidth.Coupling

/-! A mean-field approximation of AdamW training in width. The particles
are width units carrying their weights and both moment buffers; they
interact through finitely many empirical averages. `Coupling` bounds the
distance of the N-particle system to N coupled limiting particles. -/
