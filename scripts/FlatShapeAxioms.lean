import Transformer

/-! Audit *Flat kernels of any shape* of `review/flat_spectrum.md`, items R1–R4. -/

#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.conjTranspose
#print axioms Transformer.Grokking.FlatSpectrum.isFlat_one
#print axioms Transformer.Grokking.FlatSpectrum.isFlat_cubicKernel
#print axioms Transformer.Grokking.FlatSpectrum.isFlat_row_ones
#print axioms Transformer.Grokking.FlatSpectrum.conjTranspose_smul_mul_smul_eq_one
#print axioms Transformer.Grokking.FlatSpectrum.isFlat_of_singularValues_eq
#print axioms Transformer.Grokking.FlatSpectrum.singularValues_eq_of_conjTranspose_mul_self
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.gram_mul_gram
#print axioms Transformer.Grokking.FlatSpectrum.rank_eq_card_of_conjTranspose_mul_self
#print axioms Transformer.Grokking.FlatSpectrum.rank_eq_card_of_mul_conjTranspose_self
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.rank_eq
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.effectiveRank_eq
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.frobeniusSq_eq
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.nuclearNorm_eq
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.isContraction
#print axioms Transformer.Grokking.FlatSpectrum.isMinOn_iff_of_full_row_rank
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.isMinOn_iff
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.not_isMinOn_of_ne_zero
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.not_isMinOn
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.not_forall_isMinOn_iff
#print axioms Transformer.Grokking.FlatSpectrum.nuclearNorm_sq_le_rank_mul_frobeniusSq
#print axioms Transformer.Grokking.FlatSpectrum.effectiveRank_le_rank
#print axioms Transformer.Grokking.FlatSpectrum.effectiveRank_le_min
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.pairing_le
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.mul_sub_rank_le_frobeniusSq_sub
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.mul_sub_card_le_frobeniusSq_sub
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.card_le_of_eq_mul
#print axioms Transformer.Grokking.FlatSpectrum.Training.exists_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn
#print axioms Transformer.Grokking.FlatSpectrum.Training.exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le_of_not_isMinOn
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.not_exists_isOptimalFit
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.exists_nuclearNorm_add_le
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.exists_tendsto_decayRun_nuclearNorm_add_le
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.exists_pos_ae_tendsto_decayRun_nuclearNorm_add_le
#print axioms Transformer.Grokking.FlatSpectrum.IsFlat.mul_sub_card_le_fitLoss_univ
