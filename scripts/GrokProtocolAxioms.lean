import Transformer

/-! Audit the protocol and convergence target of `review/grok_division_convergence.md`. -/

#print axioms Transformer.Grokking.Protocol.predict_eq_iff
#print axioms Transformer.Grokking.Protocol.predict_eq_of_margin_pos
#print axioms Transformer.Grokking.Protocol.margin_nonneg_of_predict_eq
#print axioms Transformer.Grokking.Protocol.accuracy_eq_one_iff
#print axioms Transformer.Grokking.Protocol.accuracy_le_of_ne
#print axioms Transformer.Grokking.Protocol.tendsto_accuracy_one_iff
#print axioms Transformer.Grokking.Protocol.heldout_nonempty
#print axioms Transformer.Grokking.Protocol.nodup_batchSlots
#print axioms Transformer.Grokking.Protocol.length_batchSlots
#print axioms Transformer.Grokking.Protocol.Hyper.rateAt_one
#print axioms Transformer.Grokking.Protocol.Hyper.rateAt_of_lt
#print axioms Transformer.Grokking.Protocol.Hyper.rateAt_le
#print axioms Transformer.Grokking.Protocol.one_sub_pow_pos
#print axioms Transformer.Grokking.Protocol.adamDenominator_pos
#print axioms Transformer.Grokking.Protocol.adamRun_one_params
#print axioms Transformer.Grokking.Protocol.adamRun_second_nonneg
#print axioms Transformer.Grokking.Protocol.abs_adamRun_first_le
#print axioms Transformer.Grokking.Protocol.adamRun_second_le
#print axioms Transformer.Grokking.Protocol.abs_adamRun_step_le
#print axioms Transformer.Grokking.Protocol.TendsToOneAlmostSurely.inProbability
#print axioms Transformer.Grokking.Protocol.TendsToOneAlmostSurely.inMean
#print axioms Transformer.Grokking.Protocol.generalizes_iff
#print axioms Transformer.Grokking.Protocol.heldoutAccuracy_mem
#print axioms Transformer.Grokking.Protocol.generalizes_indicator
#print axioms Transformer.Grokking.Protocol.not_generalizes_const
#print axioms Transformer.Grokking.Protocol.tendsToOneAlmostSurely_step
