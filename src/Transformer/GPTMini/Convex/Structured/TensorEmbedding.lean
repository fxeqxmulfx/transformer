import Transformer.GPTMini.Convex.Structured.BasisTraining
import Transformer.GPTMini.RMSNorm

/-!
# Actual compact Euclidean embedding coordinates

Source: SharedSlots' 52 free token fields and OutputCodes' ten signed
decoder axes, realized in the actual `EucSpace` used by GPTMini at
62f1f6f. This proposal changes the embedding parameterization: its
first 52 coordinates are unrestricted learned potentials, the next
ten carry fixed vocabulary codes, coordinate 62 is protected at one,
and coordinate 63 receives a freely learned positional potential.
Widths 64 and 128 both fit this same layout. Extra axes are zero.

The tied token table has no positional component. Position is added
to the residual input separately; no semantic table mask, reference
state, matching label or prepared prefix enters either embedding.
These coordinate computations do not yet realize the attention head,
its residual, final tied readout or a complete changed model.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped Classical
noncomputable section

variable {V C d : ℕ}

/-- The actual residual axis for a free token field.
Source: SharedSlots' first 52 embedding coordinates. -/
def tensorFieldAxis (hwidth : 64 ≤ d) (slot : Fin 52) : Fin d :=
  ⟨slot.val, by have hs := slot.isLt; omega⟩

/-- The actual residual axis for a readonly output-code coordinate.
Source: OutputCodes' ten signed axes after the learned token fields. -/
def tensorCodeAxis (hwidth : 64 ≤ d) (slot : Fin 10) : Fin d :=
  ⟨52 + slot.val, by have hs := slot.isLt; omega⟩

/-- Protected constant used to recover free fields after genuine prenorm.
Source: the proposed separate residual anchor, rather than an external RMS multiplier. -/
def tensorAnchorAxis (hwidth : 64 ≤ d) : Fin d := ⟨62, by omega⟩

/-- The residual axis receiving the actual free positional potential.
Source: SharedPointer's learned absolute-position field, independent of the token table. -/
def tensorPositionAxis (hwidth : 64 ≤ d) : Fin d := ⟨63, by omega⟩

/-- Actual token-local Euclidean embedding, with all 52 token potentials free.
Source: SharedSlots and OutputCodes; fixed decoder/anchor axes are architecture constants, not fixed Q/K/V. -/
def tensorEmbedding (hsize : V ≤ 1024) (θ : SharedParameters V C)
    (token : Fin V) : EucSpace d :=
  WithLp.toLp 2 (fun axis =>
    if hs : axis.val < 52 then θ (.inl (token, ⟨axis.val, hs⟩))
    else if hc : axis.val < 62 then
      outputCoordinate (outputDigit (vocabularyCode hsize token))
        ⟨axis.val - 52, by omega⟩
    else if axis.val = 62 then 1 else 0)

/-- Actual input to the residual stream: token table plus one learned position component.
Source: the proposed embedding/attention replacement, with physical position supplied by the usual tensor index. -/
def tensorInput (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) (position : Fin C) : EucSpace d :=
  tensorEmbedding hsize θ token +
    EuclideanSpace.single (tensorPositionAxis hwidth) (θ (.inr (.inl position)))

/-- Reading any actual learned embedding axis returns the corresponding unrestricted shared parameter.
Source: the true Euclidean embedding above, with no token-prefix oracle. -/
theorem tensorEmbedding_field (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) (slot : Fin 52) :
    tensorEmbedding hsize θ token (tensorFieldAxis hwidth slot) = θ (.inl (token, slot)) := by
  have hs := slot.isLt
  simp only [tensorEmbedding, PiLp.toLp_apply, tensorFieldAxis, dite_eq_left hs]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 := by omega

/-- The actual tied token table contains the collision-free readonly code in its ten decoder axes.
Source: OutputCodes' code for the genuine vocabulary ID, without modulo aliasing. -/
theorem tensorEmbedding_code (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) (slot : Fin 10) :
    tensorEmbedding hsize θ token (tensorCodeAxis hwidth slot) =
      outputCoordinate (outputDigit (vocabularyCode hsize token)) slot := by
  have hs := slot.isLt
  have hfield : ¬52 + slot.val < 52 := by omega
  have hcode : 52 + slot.val < 62 := by omega
  simp only [tensorEmbedding, PiLp.toLp_apply, tensorCodeAxis,
    dite_eq_right hfield, dite_eq_left hcode, Nat.add_sub_cancel_left]

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 := by omega

/-- Every genuine token embedding has the same protected constant, regardless of free weights.
Source: the actual residual anchor coordinate, needed to undo prenorm internally. -/
theorem tensorEmbedding_anchor (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) :
    tensorEmbedding hsize θ token (tensorAnchorAxis hwidth) = 1 := by
  norm_num [tensorEmbedding, tensorAnchorAxis, PiLp.toLp_apply]

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 := by omega

/-- The tied token embedding has zero position axis, so free positions are not accidentally tied output weights.
Source: the genuine table/input separation in this embedding parameterization. -/
theorem tensorEmbedding_position (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) :
    tensorEmbedding hsize θ token (tensorPositionAxis hwidth) = 0 := by
  norm_num [tensorEmbedding, tensorPositionAxis, PiLp.toLp_apply]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 := by omega

/-- Adding learned position leaves every one of the true learned token fields intact.
Source: the actual Euclidean addition, and the position axis disjoint from all Q/K/value/transition fields. -/
theorem tensorInput_field (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) (position : Fin C) (slot : Fin 52) :
    tensorInput hsize hwidth θ token position (tensorFieldAxis hwidth slot) = θ (.inl (token, slot)) := by
  have hne : tensorFieldAxis hwidth slot ≠ tensorPositionAxis hwidth := by
    intro he; have hv := congrArg Fin.val he; have hs := slot.isLt
    simp only [tensorFieldAxis, tensorPositionAxis] at hv
    omega
  simp only [tensorInput, PiLp.add_apply, tensorEmbedding_field,
    PiLp.single_apply, ite_eq_right hne, add_zero]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 := by omega

/-- Adding position also preserves all ten genuine tied decoder axes.
Source: the actual position coordinate is disjoint from the readonly output-code range. -/
theorem tensorInput_code (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) (position : Fin C) (slot : Fin 10) :
    tensorInput hsize hwidth θ token position (tensorCodeAxis hwidth slot) =
      outputCoordinate (outputDigit (vocabularyCode hsize token)) slot := by
  have hne : tensorCodeAxis hwidth slot ≠ tensorPositionAxis hwidth := by
    intro he; have hv := congrArg Fin.val he; have hs := slot.isLt
    simp only [tensorCodeAxis, tensorPositionAxis] at hv
    omega
  simp only [tensorInput, PiLp.add_apply, tensorEmbedding_code,
    PiLp.single_apply, ite_eq_right hne, add_zero]

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 := by omega

/-- The actual input retains its protected constant under every unrestricted learned positional weight.
Source: the distinct physical anchor/position coordinates. -/
theorem tensorInput_anchor (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) (position : Fin C) :
    tensorInput hsize hwidth θ token position (tensorAnchorAxis hwidth) = 1 := by
  have hne : tensorAnchorAxis hwidth ≠ tensorPositionAxis hwidth := by
    intro he; have hv := congrArg Fin.val he; norm_num [tensorAnchorAxis, tensorPositionAxis] at hv
  simp only [tensorInput, PiLp.add_apply, tensorEmbedding_anchor,
    PiLp.single_apply, ite_eq_right hne, add_zero]

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 := by omega

/-- The actual input position axis carries its free parameter exactly.
Source: learned position addition to the genuine zero-position token embedding. -/
theorem tensorInput_position (hsize : V ≤ 1024) (hwidth : 64 ≤ d)
    (θ : SharedParameters V C) (token : Fin V) (position : Fin C) :
    tensorInput hsize hwidth θ token position (tensorPositionAxis hwidth) = θ (.inr (.inl position)) := by
  simp only [tensorInput, PiLp.add_apply, tensorEmbedding_position, PiLp.single_apply,
    ite_true, zero_add]

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 := by omega

end
end Transformer.GPTMini.Convex.Structured
