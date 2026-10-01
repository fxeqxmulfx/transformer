import Transformer.DoubleDescent.Section6_ResNetLog

/-!
# Verified epoch-wise double descent

arXiv:1912.02292v1, Section 6. The exact binary64 observations are taken
from the authors' ResNet/Adam log. The four epochs below certify the
reported qualitative phenomenon; they do not certify global minima or
monotonicity between the checkpoints.
-/

namespace Transformer.DoubleDescent

/-- Section 6, Figure 10: width-64 ResNet test error decreases from epoch
1 to 27, increases from 27 to 87, and decreases again by epoch 3999. -/
theorem resnet_epoch_doubleDescent : RecordedDoubleDescent resnetEpochLog := by
  refine ⟨⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩, ?_⟩
  norm_num [resnetEpochLog]

/-- Section 6, last paragraph: this run finishes below the earlier minimum
at epoch 27, so extended training corrects the observed overfitting. -/
theorem resnet_longer_training_corrects_overfitting : RecordedCorrection resnetEpochLog := by
  refine ⟨⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩, ?_⟩
  norm_num [resnetEpochLog]

end Transformer.DoubleDescent
