module

public import Transformer.Phasor.EmbeddingInput
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Width of the learned complex-product feed-forward branch

Source: the compact complex feed-forward branch in
`experiments/phasor_value_transport/README.md` and DiffGPT `5a37c51`,
`research/main_model/phase_score_native/src/layout.rs`, functions
`hidden` and `pairs`. The candidate uses one complex hidden unit per real
input channel. The original uses the nearest even real hidden width specified
by the pinned layout. Embedding dimension, heads, depth and routing are fixed.

Splitting the units gives a sum of two raw branches before their common RMS.
Padding every compact branch with zero units preserves all four states and
the learned readout on arbitrary sequences of embeddings at any depth.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.ComplexFFN Transformer.MHC.PeriRMS

variable {D d H F G : ℕ}

/-- Concatenate the learned units in two branches. The complex rows each
retain both output columns. -/
def FFN.append (f : FFN D F) (g : FFN D G) : FFN D (F+G) where
  gate := Fin.addCases f.gate g.gate
  value := Fin.addCases f.value g.value
  outRe := Fin.addCases f.outRe g.outRe
  outIm := Fin.addCases f.outIm g.outIm

/-- The contribution of the removed units is exactly their raw complex
product branch for every input and every setting of the weights. -/
theorem FFN.plain_append (f : FFN D F) (g : FFN D G) (z : EucSpace D) :
    plainFFN (f.append g).gate (f.append g).value (f.append g).outRe (f.append g).outIm z =
      plainFFN f.gate f.value f.outRe f.outIm z +
        plainFFN g.gate g.value g.outRe g.outIm z := by
  unfold plainFFN complexReadout
  rw [Fin.sum_univ_add]
  simp only [FFN.append, plainHidden, Fin.addCases_left, Fin.addCases_right]

/-- Zero extra units preserve both routing and the raw branch at every
state, before the model applies the original residual and RMS. -/
theorem FFN.sublayer_append_zero (f : FFN D F) (g : ℕ) (R : Router D) (eps : ℝ) :
    (f.append (FFN.zero D g)).sublayer R eps = f.sublayer R eps := by
  unfold FFN.sublayer
  simp only [FFN.plain_append]
  simp [FFN.zero, plainFFN, complexReadout, plainHidden]

/-- Enlarge all feed-forward layers by the same number of inactive units.
The attention parameters, rotary scales and both routers are copied. -/
noncomputable def EmbeddingBackbone.padFFN (A : EmbeddingBackbone D d H F) (g : ℕ) :
    EmbeddingBackbone D d H (F+g) where
  alpha := A.alpha
  layer l :=
    { attention := (A.layer l).attention, attentionRouter := (A.layer l).attentionRouter,
      ffn := (A.layer l).ffn.append (FFN.zero D g), ffnRouter := (A.layer l).ffnRouter }

/-- The wider model realizes the compact model exactly when its added
units are zero. The conclusion covers every layer, position and stream. -/
theorem EmbeddingBackbone.states_padFFN (A : EmbeddingBackbone D d H F)
    (g : ℕ) (eps scale : ℝ) (x : ℕ → EucSpace D) (l : ℕ) :
    (A.padFFN g).states eps scale x l = A.states eps scale x l := by
  induction l with
  | zero => rfl
  | succ l ih =>
    funext i
    rw [EmbeddingBackbone.states, EmbeddingBackbone.states, ih]
    simp only [EmbeddingBackbone.padFFN, Layer.apply, FFN.sublayer_append_zero]

/-- Every classifier applied to the learned stream read receives the same
normalized features after padding the compact network. -/
theorem EmbeddingBackbone.feature_padFFN (A : EmbeddingBackbone D d H F)
    (g : ℕ) (θ : StreamReadWeights D) (eps scale : ℝ) (x : ℕ → EucSpace D) (l i : ℕ) :
    (A.padFFN g).feature θ eps scale x l i = A.feature θ eps scale x l i := by
  unfold EmbeddingBackbone.feature
  rw [EmbeddingBackbone.states_padFFN]

/-- Complex-unit count of the pinned real layout after rounding its
`(8*D+1)/3` integer result up to an even real width. -/
def originalComplexUnits (D : ℕ) : ℕ := ((8*D+1)/3+1)/2

/-- The compact width fits the original at every dimension, including
nonmultiples of three. -/
theorem compact_units_le_original (D : ℕ) : D ≤ originalComplexUnits D := by
  unfold originalComplexUnits
  omega

/-- Expanded entries in two input projections and one output projection.
This also counts their real multiply-accumulate operations per token. -/
def ffnProjectionEntries (D units : ℕ) : ℕ := 3 * D * (2 * units)

/-- The exact saving per layer when the compact model uses one complex
unit per real channel. -/
theorem compact_projection_saving (D : ℕ) :
    ffnProjectionEntries D (originalComplexUnits D) =
      ffnProjectionEntries D D + 6 * D * (originalComplexUnits D - D) := by
  unfold ffnProjectionEntries
  calc
    _ = 6 * D * (D + (originalComplexUnits D - D)) := by
      rw [Nat.add_sub_cancel' (compact_units_le_original D)]
      ring
    _ = _ := by ring

example : (originalComplexUnits 64, originalComplexUnits 128, originalComplexUnits 256) =
    (86, 171, 342) := by decide

example : ffnProjectionEntries 128 (originalComplexUnits 128) - ffnProjectionEntries 128 128 =
    33024 := by decide

end Transformer.Phasor
