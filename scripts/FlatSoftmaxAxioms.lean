import Transformer

/-! Audit *Softmax attention escapes the cut* of `review/flat_spectrum.md`, items SA1–SA4. -/

#print axioms Transformer.Grokking.FlatSpectrum.Recall.abs_softmaxAttention_sub_le
#print axioms Transformer.Grokking.FlatSpectrum.Recall.mul_sum_exp_le_sum_erase
#print axioms Transformer.Grokking.FlatSpectrum.Recall.mul_abs_sub_le_abs_sub_softmaxAttention
#print axioms Transformer.Grokking.FlatSpectrum.Recall.sum_queryProj_mul_keyProj
#print axioms Transformer.Grokking.FlatSpectrum.Recall.sum_queryProj_mul_keyProj_add_le
#print axioms Transformer.Grokking.FlatSpectrum.Recall.softmaxAttention_recall_error
#print axioms Transformer.Grokking.FlatSpectrum.Recall.softmaxAttention_self_recall_error
#print axioms Transformer.Grokking.FlatSpectrum.Recall.exists_softmaxAttention_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.exists_softmaxAttention_recall_sign
#print axioms Transformer.Grokking.FlatSpectrum.Recall.sum_sum_update
#print axioms Transformer.Grokking.FlatSpectrum.Recall.exists_forall_mul_abs_sub_le_softmaxAttention
#print axioms Transformer.Grokking.FlatSpectrum.Recall.mul_le_sum_softmaxAttention_sq
#print axioms Transformer.Grokking.FlatSpectrum.Recall.sub_one_mul_le_of_softmaxAttention_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.softmaxAttention_oneHot_recall_error
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_normalized_sign
#print axioms Transformer.Grokking.FlatSpectrum.Recall.softmaxAttention_option
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_queryFreeAttention_sign
