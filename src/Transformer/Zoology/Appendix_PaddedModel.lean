/-
# Input and output dimensions of a Coyote model

Arora et al., arXiv:2312.04927v1, Appendix `def: gated-conv` and
`def: W-kmat`.  A model may use larger inner sequence and feature dimensions.
The input is embedded in the upper-left corner with zeros elsewhere, then
the same corner is extracted after the stacked layers.
-/

import Transformer.Zoology.Appendix_Network

namespace Transformer.Zoology

/-- Embed an `n × d` sequence into the upper-left corner of an `N × D`
inner sequence.  Source: Appendix `def: gated-conv`, display after the
definition. -/
def padSequence {n d N D : ℕ} (u : RealSequence n d) :
    RealSequence N D :=
  fun i q =>
    if hi : i.val < n then
      if hq : q.val < d then u ⟨i.val, hi⟩ ⟨q.val, hq⟩ else 0
    else 0

/-- Extract the upper-left output corner.  Source: Appendix
`def: gated-conv`, paragraph after the zero-padding display. -/
def cropSequence {n d N D : ℕ} (hn : n ≤ N) (hd : d ≤ D)
    (u : RealSequence N D) : RealSequence n d :=
  fun i q =>
    u ⟨i.val, lt_of_lt_of_le i.isLt hn⟩
      ⟨q.val, lt_of_lt_of_le q.isLt hd⟩

/-- Padding and then cropping returns the original input exactly.
Source: Appendix `def: gated-conv`. -/
theorem crop_pad_sequence {n d N D : ℕ} (hn : n ≤ N) (hd : d ≤ D)
    (u : RealSequence n d) :
    cropSequence hn hd (padSequence u : RealSequence N D) = u := by
  funext i q
  simp [cropSequence, padSequence, i.isLt, q.isLt]

/-- A Coyote stack together with its external and inner dimensions.
Source: Appendix `def: gated-conv`, specialized to the Coyote operator. -/
structure PaddedCoyoteModel (n d : ℕ) where
  innerLength : ℕ
  innerWidth : ℕ
  lengthBound : n ≤ innerLength
  widthBound : d ≤ innerWidth
  network : CoyoteNetwork innerLength innerWidth

/-- Execute the zero-padded model and crop its output.
Source: Appendix `def: gated-conv`. -/
def PaddedCoyoteModel.run {n d : ℕ} (model : PaddedCoyoteModel n d)
    (u : RealSequence n d) : RealSequence n d :=
  cropSequence model.lengthBound model.widthBound
    (model.network.run (padSequence u))

/-- Its depth is the number of Coyote layers.
Source: Appendix `def: gated-conv`. -/
def PaddedCoyoteModel.layerCount {n d : ℕ}
    (model : PaddedCoyoteModel n d) : ℕ :=
  model.network.layerCount

/-- A model with no layers and no expansion computes the identity.
This establishes that the model class is nonempty at every input shape.
Source: Appendix `def: gated-conv`, input/output embedding. -/
theorem empty_padded_model_identity {n d : ℕ} (u : RealSequence n d) :
    (PaddedCoyoteModel.mk n d (le_refl n) (le_refl d)
      ⟨[]⟩).run u = u := by
  simp [PaddedCoyoteModel.run, CoyoteNetwork.run, crop_pad_sequence]

end Transformer.Zoology
