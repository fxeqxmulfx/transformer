import Transformer.GPTMini.Semantics.DepthRecurrence
import Transformer.Basis.Tasks

/-!
# Actual token-local depth embedding in the original model widths

Source: Basis AlternatingBlocks/vocabulary at cbafbe9 and GPTMini at
f11b6e2, as ported in experiments/basis/experiment.py. Easy uses the
64-wide two-layer model; hard uses the 128-wide six-layer model. Both
retain four heads, the original four-times-width FFN and RoPE theta 10000.
The context is 128, including Basis's held-out depth-length evaluation.

All 36 vocabulary entries are ordinary tied embedding vectors. BOS and
neutral tokens have only a protected constant; A and B add their own
token-local type axis. ACCEPT and REJECT have opposite readout codes.
No entry reads a prefix, ordered occurrence or desired language label.

The 24-coordinate working layout fits both widths. Raw validated depth
inputs have norm at most two and zero detector/readout channels. The
large label embeddings are excluded by the actual raw-input grammar;
no uniform bound on the whole tied vocabulary is claimed here.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- The original small/easy and large/hard depth dimensions, with the actual OOD context.
Source: experiments/basis/experiment.py and Basis.contextSize depth. -/
noncomputable abbrev depthConfig (mode : Mode) : Config where
  vocab_size := 36
  n_layers := if mode = .easy then 2 else 6
  n_heads := 4
  d_model := if mode = .easy then 64 else 128
  d_ff := if mode = .easy then 256 else 512
  max_seq_len := 128
  rope_theta := 10000
  divides := by cases mode <;> decide
  head_even := by cases mode <;> decide
  n_heads_pos := by decide
  n_layers_pos := by cases mode <;> decide
  d_model_pos := by cases mode <;> decide
  d_ff_pos := by cases mode <;> decide
  vocab_pos := by decide
  max_seq_len_pos := by decide
  theta_pos := by norm_num

/-- The finite working layout fits the unchanged residual widths.
Source: the original width-64 and width-128 recipes, without added coordinates. -/
theorem depthConfig_width (mode : Mode) :
    64 ≤ (depthConfig mode).d_model ∧ (depthConfig mode).d_model ≤ 128 := by
  cases mode <;> decide

/-- Seven ordinary FFN units, including the later readout scale unit, fit either original FFN.
Source: the original expansion factor four; the detectors themselves use only six units. -/
theorem depthConfig_ffn (mode : Mode) : 7 ≤ (depthConfig mode).d_ff := by
  cases mode <;> decide

/-- A working axis is an actual coordinate of this same residual stream.
Source: the explicit 24-coordinate depth layout. -/
def depthCoordinate (mode : Mode) (axis : Fin 24) : Fin (depthConfig mode).d_model :=
  ⟨axis.val, by have ha := axis.isLt; have hd := (depthConfig_width mode).1; omega⟩

/-- Distinct working channels remain distinct in each original model width.
Source: the unchanged integer coordinate embedding. -/
theorem depthCoordinate_injective (mode : Mode) : Function.Injective (depthCoordinate mode) := by
  intro a b hab
  apply Fin.ext
  have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) hab
  exact hv

/-- An ordinary unit vector of the actual residual stream.
Source: coordinate insertion in GPTMini's existing Euclidean-space parameter types. -/
noncomputable def depthAxis (mode : Mode) (axis : Fin 24) : EucSpace (depthConfig mode).d_model :=
  EuclideanSpace.single (depthCoordinate mode axis) 1

/-- Reading an inserted unit axis uses exactly the same working-coordinate identity.
Source: the actual finite coordinate injection, not a semantic feature definition. -/
theorem depthAxis_coordinate (mode : Mode) (axis coordinate : Fin 24) :
    depthAxis mode axis (depthCoordinate mode coordinate) = if coordinate = axis then 1 else 0 := by
  simp only [depthAxis, PiLp.single_apply]
  have he : depthCoordinate mode coordinate = depthCoordinate mode axis ↔ coordinate = axis :=
    (depthCoordinate_injective mode).eq_iff
  simp only [he]

/-- Every inserted unit axis has norm one in either actual width.
Source: the original L2 residual norm and a genuine single-coordinate vector. -/
theorem depthAxis_norm (mode : Mode) (axis : Fin 24) : ‖depthAxis mode axis‖ = 1 := by
  simp only [depthAxis, PiLp.norm_single, Real.norm_eq_abs, abs_one]

/-- The tied label direction reserves positive, negative and common-scale readout channels.
Source: the later strict E_2/E_4 score comparison, with coefficients 1,-2,-1/2. -/
noncomputable def depthReadoutDirection (mode : Mode) : EucSpace (depthConfig mode).d_model :=
  depthAxis mode 17 - (2 : ℝ) • depthAxis mode 18 - (1 / 2 : ℝ) • depthAxis mode 19

/-- All actual vocabulary embeddings depend only on their own checked integer token ID.
Source: cbafbe9's A=9, B=10, ACCEPT=16, REJECT=15; unused entries retain the constant. -/
noncomputable def depthRawEmbedding (mode : Mode) (gain : ℝ) (token : Fin 36) :
    EucSpace (depthConfig mode).d_model :=
  depthAxis mode 0 +
    if token.val = 9 then depthAxis mode 1
    else if token.val = 10 then depthAxis mode 2
    else if token.val = 16 then gain • depthReadoutDirection mode
    else if token.val = 15 then -gain • depthReadoutDirection mode
    else 0

/-- Exactly the actual raw depth-input alphabet, including BOS and neutral.
Source: Basis.DepthPrefix's serialization; labels never occur in the following input stream. -/
def DepthRawToken (token : Fin 36) : Prop :=
  token.val = 1 ∨ token.val = 9 ∨ token.val = 10 ∨ token.val = 11

/-- A validated raw token has only its constant and its genuine local A/B type coordinate.
Source: the complete ordinary vocabulary table, restricted by the actual input grammar. -/
theorem depthRawEmbedding_input (mode : Mode) (gain : ℝ) (token : Fin 36) (ht : DepthRawToken token) :
    depthRawEmbedding mode gain token = depthAxis mode 0 +
      if token.val = 9 then depthAxis mode 1 else if token.val = 10 then depthAxis mode 2 else 0 := by
  rcases ht with ht | ht | ht | ht <;> simp [depthRawEmbedding, ht]

example : DepthRawToken ⟨1, by decide⟩ := Or.inl rfl

/-- Every actual raw input preserves the same protected constant coordinate.
Source: the token-local table and distinct ordinary A/B axes. -/
theorem depthRawEmbedding_constant (mode : Mode) (gain : ℝ) (token : Fin 36) (ht : DepthRawToken token) :
    depthRawEmbedding mode gain token (depthCoordinate mode 0) = 1 := by
  rw [depthRawEmbedding_input mode gain token ht]
  split_ifs <;> simp [depthAxis_coordinate]

example : DepthRawToken ⟨9, by decide⟩ := Or.inr (Or.inl rfl)

/-- All later detector and readout channels start at zero on every genuine raw input.
Source: the explicit disjoint 24-coordinate working layout; only axes 0,1,2 are initially occupied. -/
theorem depthRawEmbedding_fresh (mode : Mode) (gain : ℝ) (token : Fin 36) (ht : DepthRawToken token)
    (coordinate : Fin 24) (h0 : coordinate ≠ 0) (h1 : coordinate ≠ 1) (h2 : coordinate ≠ 2) :
    depthRawEmbedding mode gain token (depthCoordinate mode coordinate) = 0 := by
  rw [depthRawEmbedding_input mode gain token ht]
  split_ifs <;> simp [depthAxis_coordinate, h0, h1, h2]

example : DepthRawToken ⟨10, by decide⟩ ∧ (3 : Fin 24) ≠ 0 ∧ (3 : Fin 24) ≠ 1 ∧ (3 : Fin 24) ≠ 2 := by
  exact ⟨Or.inr (Or.inr (Or.inl rfl)), by decide, by decide, by decide⟩

/-- Raw input norms have a genuine lower bound from their protected coordinate and an upper bound of two.
Source: the actual local embedding vectors, independent of the label readout gain. -/
theorem depthRawEmbedding_norm (mode : Mode) (gain : ℝ) (token : Fin 36) (ht : DepthRawToken token) :
    1 ≤ ‖depthRawEmbedding mode gain token‖ ∧ ‖depthRawEmbedding mode gain token‖ ≤ 2 := by
  constructor
  · have h := PiLp.norm_apply_le (depthRawEmbedding mode gain token) (depthCoordinate mode 0)
    rw [depthRawEmbedding_constant mode gain token ht, Real.norm_eq_abs, abs_one] at h
    exact h
  · rw [depthRawEmbedding_input mode gain token ht]
    split_ifs
    · calc ‖depthAxis mode 0 + depthAxis mode 1‖ ≤ ‖depthAxis mode 0‖ + ‖depthAxis mode 1‖ := norm_add_le _ _
        _ = 2 := by rw [depthAxis_norm, depthAxis_norm]; norm_num
    · calc ‖depthAxis mode 0 + depthAxis mode 2‖ ≤ ‖depthAxis mode 0‖ + ‖depthAxis mode 2‖ := norm_add_le _ _
        _ = 2 := by rw [depthAxis_norm, depthAxis_norm]; norm_num
    · rw [add_zero, depthAxis_norm]
      norm_num

example : DepthRawToken ⟨11, by decide⟩ := Or.inr (Or.inr (Or.inr rfl))

end Transformer.GPTMini.Semantics
