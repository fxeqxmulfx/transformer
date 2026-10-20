import Transformer.Causal.Section4_SubspaceTwoTokens

/-!
# Axiom inspection of the complete two-token subspace proof

ArXiv:2411.04990v2, Section 4, Conjecture `thm2`, restricted to two tokens.
All 29 new results must use only `propext`, `Classical.choice`, and
`Quot.sound`. The original assertion for three or more tokens remains
unfinished in `Causal.MainTheorem` and is not used by these proofs.

Run with `lake env lean scripts/CausalSubspaceAxioms.lean`.
-/

#print axioms Transformer.Causal.inner_value_of_mem_leading
#print axioms Transformer.Causal.rayleigh_defect_eq_transverse
#print axioms Transformer.Causal.rayleigh_le_leading
#print axioms Transformer.Causal.hasDerivAt_csaLeadingOverlap
#print axioms Transformer.Causal.leadingOverlapInteraction_approx
#print axioms Transformer.Causal.inner_starProjection_of_mem_left
#print axioms Transformer.Causal.leadingTransverse_mem
#print axioms Transformer.Causal.leadingTransverse_inner_leader
#print axioms Transformer.Causal.leadingTransverse_inner_self
#print axioms Transformer.Causal.leadingTransverse_witness
#print axioms Transformer.Causal.inner_eq_zero_of_leadingTransverse_eq_zero
#print axioms Transformer.Causal.exists_leading_transverse_normal
#print axioms Transformer.Causal.continuous_leadingTransverse
#print axioms Transformer.Causal.ae_leading_transverse_witness
#print axioms Transformer.Causal.subspace_cluster_pair_of_transverse
#print axioms Transformer.Causal.subspace_cluster_twoTokens
#print axioms Transformer.Causal.scalar_eq_initial_mul_exp_integral
#print axioms Transformer.Causal.scalar_pos_of_homogeneous
#print axioms Transformer.Causal.continuous_csaLeadingCoefficient
#print axioms Transformer.Causal.csaLeader_leading_component_scale
#print axioms Transformer.Causal.csaLeader_leading_orthogonal
#print axioms Transformer.Causal.hasDerivAt_csaSecond_leading
#print axioms Transformer.Causal.csaSecond_pos_transverse_leading
#print axioms Transformer.Causal.quotient_tendsto_atTop_of_positive_forcing
#print axioms Transformer.Causal.eventually_pos_of_quotient_tendsto_atTop
#print axioms Transformer.Causal.csaSecond_eventually_pos_leading_of_transverse
#print axioms Transformer.Causal.csaSecond_leading_field_lower
#print axioms Transformer.Causal.hasDerivAt_csaSecond_fieldOverlap
#print axioms Transformer.Causal.csaSecond_tendsto_of_eventually_nonneg_leading
