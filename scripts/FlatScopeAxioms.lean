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
