/-
# Stacking Coyote layers

Arora et al., arXiv:2312.04927v1, Appendix Lemma `lem: stacking-layers`
in `circuit/primitives/primitives.tex`.  A finite list of layers has an
ordinary compositional semantics, and concatenating lists composes their
functions while adding their layer counts.
-/

import Transformer.Zoology.Section4_Coyote

namespace Transformer.Zoology

/-- A finite stack of equal-width Coyote layers.
Source: Appendix Lemma `lem: stacking-layers`. -/
structure CoyoteNetwork (n d : ℕ) where
  layers : List (CoyoteParameters n d)

/-- Execute layers from the head of the list to the tail.
Source: Appendix Lemma `lem: stacking-layers`. -/
def CoyoteNetwork.run {n d : ℕ} (net : CoyoteNetwork n d)
    (u : RealSequence n d) : RealSequence n d :=
  net.layers.foldl (fun state p => coyoteLayer p state) u

/-- Number of stacked Coyote layers.
Source: Appendix Lemma `lem: stacking-layers`. -/
def CoyoteNetwork.layerCount {n d : ℕ} (net : CoyoteNetwork n d) : ℕ :=
  net.layers.length

/-- List concatenation is network composition, including the exact additive
layer count.  Source: Appendix Lemma `lem: stacking-layers`, for equal inner
sequence and feature dimensions. -/
theorem coyote_stacking {n d : ℕ} (first second : CoyoteNetwork n d)
    (u : RealSequence n d) :
    ({layers := first.layers ++ second.layers} : CoyoteNetwork n d).run u =
        second.run (first.run u) ∧
      ({layers := first.layers ++ second.layers} : CoyoteNetwork n d).layerCount =
        first.layerCount + second.layerCount := by
  constructor
  · simp [CoyoteNetwork.run, List.foldl_append]
  · simp [CoyoteNetwork.layerCount]

/-- The identity matrix acting on feature coordinates.
Source: Appendix `lmm:primitives`, linear-projection construction. -/
def identityWeight (d : ℕ) : Fin d → Fin d → ℝ :=
  fun k q => if k = q then 1 else 0

/-- A linear projection by the identity matrix does nothing.
Source: Appendix `lmm:primitives`. -/
theorem linearProjection_identity {n d : ℕ} (u : RealSequence n d) :
    linearProjection u (identityWeight d) = u := by
  classical
  funext i q
  simp [linearProjection, identityWeight]

/-- A Coyote layer can be used as an identity layer when padding a stack.
Source: Appendix `lmm:primitives`, linear-projection construction. -/
theorem coyote_identity {n d : ℕ} (u : RealSequence n d) :
    coyoteLayer (linearCoyoteParameters (identityWeight d)) u = u := by
  rw [coyote_realizes_linear, linearProjection_identity]

/-- The assumptions of the paper's stacking lemma are realized by two
identity networks of equal inner dimensions. -/
example : ∃ first second : CoyoteNetwork 2 1,
    ∀ u : RealSequence 2 1, first.run u = u ∧ second.run u = u := by
  let p : CoyoteParameters 2 1 := linearCoyoteParameters (identityWeight 1)
  refine ⟨{layers := [p]}, {layers := [p]}, ?_⟩
  intro u
  constructor <;> simp [CoyoteNetwork.run, p, coyote_identity]

end Transformer.Zoology
