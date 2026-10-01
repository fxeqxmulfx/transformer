import Transformer.DoubleDescent.Section7_TranslationLog

/-!
# Verified sample-size non-monotonicity

arXiv:1912.02292v1, Section 7, Figure 11(b), and Section 1, Figure 3.
The actual logs certify worse exported test loss with more train samples
for both fixed model sizes. These comparisons do not equate a sentence
count with an iid token count, or certify the expected population error.
-/

namespace Transformer.DoubleDescent

/-- Sections 1 and 7: 18000 samples have higher recorded loss than 4000
for the same Transformer with embedding dimension 64. -/
theorem sample64_moreDataHurts : RecordedMoreDataHurts sample64Log := by
  refine ⟨⟨0, by decide⟩, ⟨7, by decide⟩, ?_⟩
  norm_num [sample64Log]

/-- Section 7, Figure 11(b): the same non-monotonicity occurs at dimension 80. -/
theorem sample80_moreDataHurts : RecordedMoreDataHurts sample80Log := by
  refine ⟨⟨0, by decide⟩, ⟨7, by decide⟩, ?_⟩
  norm_num [sample80Log]

/-- Section 1, Figure 3: the actual data increase is a factor of 4.5,
and the recorded loss increases for both model sizes. -/
theorem four_and_a_half_times_more_data_hurts :
    2 * sample64Log[7].parameter = 9 * sample64Log[0].parameter ∧
    sample64Log[0].value < sample64Log[7].value ∧
    2 * sample80Log[7].parameter = 9 * sample80Log[0].parameter ∧
    sample80Log[0].value < sample80Log[7].value := by
  norm_num [sample64Log, sample80Log]

/-- Section 7: the harm is localized in sample size; in the same dimension-64
log, increasing from 18000 to 36000 samples improves test loss. -/
theorem sample64_larger_dataset_recovers :
    sample64Log[7].parameter < sample64Log[16].parameter ∧
      sample64Log[16].value < sample64Log[7].value := by
  norm_num [sample64Log]

end Transformer.DoubleDescent
