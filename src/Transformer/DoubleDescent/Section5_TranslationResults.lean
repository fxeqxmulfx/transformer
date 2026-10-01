import Transformer.DoubleDescent.Section5_TranslationLog

/-!
# Verified model-wise translation observations

arXiv:1912.02292v1, Section 5, Figure 8. These theorems certify exact
inequalities between the authors' exported numbers, not a statement about
the population risks of all Transformers or retraining with fresh seeds.
-/

namespace Transformer.DoubleDescent

/-- Section 5, Figure 8, IWSLT: first descent from width 8 to 72, increase
from 72 to 208, and second descent from 208 to 512. -/
theorem iwslt_model_doubleDescent : RecordedDoubleDescent iwsltModelLog := by
  refine ⟨⟨0, by decide⟩, ⟨8, by decide⟩, ⟨25, by decide⟩, ⟨63, by decide⟩, ?_⟩
  norm_num [iwsltModelLog]

/-- Section 5, Figure 8, WMT: the exported widths 8, 72, 208, and 496
also witness a first descent, ascent, and second descent. -/
theorem wmt_model_doubleDescent : RecordedDoubleDescent wmtModelLog := by
  refine ⟨⟨0, by decide⟩, ⟨8, by decide⟩, ⟨25, by decide⟩, ⟨61, by decide⟩, ?_⟩
  norm_num [wmtModelLog]

/-- Section 5, Figure 8: increasing the IWSLT Transformer width from 72
to 208 worsens recorded test loss, despite substantially lower training loss
in the CSV. Only the test-loss inequality is asserted here. -/
theorem bigger_iwslt_model_has_worse_recorded_loss :
    iwsltModelLog[8].parameter < iwsltModelLog[25].parameter ∧
      iwsltModelLog[8].value < iwsltModelLog[25].value := by
  norm_num [iwsltModelLog]

end Transformer.DoubleDescent
