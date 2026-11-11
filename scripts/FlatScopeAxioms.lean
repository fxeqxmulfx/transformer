import Transformer

/-! Audit the scope items of `review/flat_spectrum.md`: WD12 of *Training reaches the optimal
fit* and the answers to the items "Assumed or not covered" of WD11 and XE13
(`Transformer.Grokking.FlatSpectrum.Training.Scope`). -/

#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.linear_first_order
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.pairing_neg_one_smul_one
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.exists_linear_lt
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.analyticAt_linear
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.decayRun_linear_decay
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.decayRun_linear_add
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.tendsto_frobeniusSq_decayRun_linear_atTop
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.not_tendsto_decayRun_linear
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.decayedLoss_linear_smul
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.not_isDecayedMin_linear
#print axioms Transformer.Grokking.FlatSpectrum.Training.hasDerivAt_decayedLoss_of_hasDerivAt
#print axioms Transformer.Grokking.FlatSpectrum.Training.hasGradientAt_flatLoss_of_hasDerivAt
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.hasDerivAt_comp_add_smul_of_analytic
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.gradient_flatLoss_of_analytic
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.factors_iterate_gradientStep_of_analytic
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.exists_pos_tendsto_decayRun_of_analytic
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.sqWeight_nonneg
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.sqWeight_le_half
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.sqLogitLoss_eq_crossEntropy
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.sqLogitLoss_nonneg
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.sqLogitLoss_zero
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.sqLogitGrad_zero
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.pairing_sqLogitGrad
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.hasDerivAt_sqLogitLoss
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.continuous_sqLogitGrad
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.analyticAt_sqLogitLoss
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.exists_pos_tendsto_decayRun_sqLogit
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.sqLogitLoss_lt_zero
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.not_first_order_sqLogit
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.frobeniusSq_sqLogitGrad_le
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.neg_le_pairing_sqLogitGrad
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.tendsto_zero_of_tendsto_frobeniusSq
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.re_sq_le_frobeniusSq
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.frobeniusSq_sqLogitGrad_mul_le
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.frobeniusSq_decayStep_sqLogit_le
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.frobeniusSq_decayRun_sqLogit_le
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.tendsto_decayRun_sqLogit_zero
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.one_sub_smul_lift_posDef_sqLogit
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.not_isDecayedMin_sqLogit_zero
#print axioms Transformer.Grokking.FlatSpectrum.Training.Scope.exists_tendsto_decayRun_sqLogit_not_isDecayedMin
