import Transformer

/-! Audit *The carry of the codec*, items CC1 to CC3, and *The decoder holds the level*, item
DL1, of `review/diffgpt/steps.md`. -/

#print axioms Transformer.Grokking.DiffGPT.floor_add_eq_ite
#print axioms Transformer.Grokking.DiffGPT.carriedStep
#print axioms Transformer.Grokking.DiffGPT.carriedStep_eq
#print axioms Transformer.Grokking.DiffGPT.carriedStep_eq_or
#print axioms Transformer.Grokking.DiffGPT.carriedStep_of_int
#print axioms Transformer.Grokking.DiffGPT.carriedStep_intCast
#print axioms Transformer.Grokking.DiffGPT.carriedStep_two
#print axioms Transformer.Grokking.DiffGPT.eq_modStep_iff
#print axioms Transformer.Grokking.DiffGPT.modStep_add_mul_not_mem
