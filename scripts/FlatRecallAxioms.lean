import Transformer

/-! Audit *Associative recall has a flat spectrum* of `review/flat_spectrum.md`, items AR1–AR9. -/

#print axioms Transformer.Grokking.FlatSpectrum.Recall.sum_apply
#print axioms Transformer.Grokking.FlatSpectrum.Recall.sum_apply_mul_apply
#print axioms Transformer.Grokking.FlatSpectrum.Recall.recallKernel_mul_conjTranspose
#print axioms Transformer.Grokking.FlatSpectrum.Recall.isFlat_recallKernel
#print axioms Transformer.Grokking.FlatSpectrum.Recall.scale_pos
#print axioms Transformer.Grokking.FlatSpectrum.Recall.rank_recallKernel
#print axioms Transformer.Grokking.FlatSpectrum.Recall.effectiveRank_recallKernel
#print axioms Transformer.Grokking.FlatSpectrum.Recall.isMinOn_recallKernel_iff
#print axioms Transformer.Grokking.FlatSpectrum.Recall.not_isMinOn_recallKernel
#print axioms Transformer.Grokking.FlatSpectrum.Recall.sum_sq_le_sum_sub_sq
#print axioms Transformer.Grokking.FlatSpectrum.Recall.exists_centering
#print axioms Transformer.Grokking.FlatSpectrum.Recall.HasCutWidth.mul_le_sum_sq
#print axioms Transformer.Grokking.FlatSpectrum.Recall.hasCutWidth_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.HasCutWidth.card_le_of_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.hasCutWidth_recall_iff
#print axioms Transformer.Grokking.FlatSpectrum.Recall.HasCutWidth.comp_context
#print axioms Transformer.Grokking.FlatSpectrum.Recall.HasCutWidth.card_le_of_sign
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_argmax
#print axioms Transformer.Grokking.FlatSpectrum.Recall.prompt_eq_add
#print axioms Transformer.Grokking.FlatSpectrum.Recall.causalConvolution_add
#print axioms Transformer.Grokking.FlatSpectrum.Recall.coyoteLayerCyclic_last
#print axioms Transformer.Grokking.FlatSpectrum.Recall.hasCutWidth_coyoteLayer
#print axioms Transformer.Grokking.FlatSpectrum.Recall.coyoteLayer_recall_error
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_coyoteLayer_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_coyoteLayer_argmax
#print axioms Transformer.Grokking.FlatSpectrum.Recall.exists_coyoteLayer_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.hasCutWidth_retNet
#print axioms Transformer.Grokking.FlatSpectrum.Recall.retNet_recall_error
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_retNet_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_retNet_argmax
#print axioms Transformer.Grokking.FlatSpectrum.Recall.hasCutWidth_linearAttention
#print axioms Transformer.Grokking.FlatSpectrum.Recall.linearAttention_recall_error
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_linearAttention_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_linearAttention_argmax
#print axioms Transformer.Grokking.FlatSpectrum.Recall.hasCutWidth_biasedAttention
#print axioms Transformer.Grokking.FlatSpectrum.Recall.biasedAttention_recall_error
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_biasedAttention_recall
#print axioms Transformer.Grokking.FlatSpectrum.Recall.card_le_of_biasedAttention_argmax
#print axioms Transformer.Grokking.FlatSpectrum.Recall.mul_sub_card_le_fitLoss_recallKernel
#print axioms Transformer.Grokking.FlatSpectrum.Recall.exists_nuclearNorm_add_le_recallKernel
#print axioms Transformer.Grokking.FlatSpectrum.Recall.exists_pos_ae_tendsto_decayRun_recallKernel
