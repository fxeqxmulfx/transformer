import Transformer.GPTMini.Semantics.DepthReadoutBlock

/-!
# Actual raw-word depth final state and semantic coordinates

Source: original blockForward at f11b6e2 and complete genuine detector
induction at 14a2f0d. This state computes the actual raw embedding,
one/three ordinary detectors and one original readout block. Its scale
is the squared genuine pre-FFN RMS multiplier of that same computed
residual. Neither state nor scale uses a desired output label.

All final raw and readout coordinates, quantitative scale and norm
bounds derive from the raw word, with no supplied encoder or invariant.
The two flags detect independently defined ordered visible occurrences;
the common coordinate has the same positive true scale. This is the
last substantive hidden-state input for strict tied-label margins.
Full ModelParams/hidden-loop and integer-prefix coupling remain to do.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical BigOperators
open Transformer.Basis

/-- The genuine final raw-word state, computed solely by the unchanged original detector/readout blocks.
Source: token-local embedding, actual detector prefix and original two-residual readout block. -/
noncomputable def depthWordFinalState (mode : Mode) (gain eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) (i : Fin word.length) : EucSpace (depthConfig mode).d_model :=
  blockForward (depthConfig mode) (depthReadoutBlock mode) eps positions
    (depthWordState mode gain eps word positions (depthDetectorCount mode)) i

/-- The true squared prenorm multiplier of the same actual computed readout-attention residual.
Source: original RMSNorm denominator, with no compensating input-dependent matrix parameter. -/
noncomputable def depthWordReadoutScale (mode : Mode) (gain eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) (i : Fin word.length) : ℝ :=
  (depthScale mode eps (depthReadoutAttentionState mode eps positions
    (depthWordState mode gain eps word positions (depthDetectorCount mode)) i)) ^ 2

/-- Every reserved readout axis is truly zero before the final FFN, directly from raw-word computation.
Source: detector-prefix upper-axis freshness and genuine protected readout attention. -/
theorem depthWordReadout_upper_zero (mode : Mode) (gain eps : ℝ) (word : List (Option Bool))
    (positions : Fin word.length → ℝ) (i : Fin word.length) (c : Fin 24) (hc : 15 ≤ c.val) :
    depthReadoutAttentionState mode eps positions
      (depthWordState mode gain eps word positions (depthDetectorCount mode)) i (depthCoordinate mode c) = 0 := by
  rw [depthReadoutAttentionState_protected mode eps positions _ i c (Or.inr hc)]
  exact depthWordState_upper_zero mode gain eps word positions _ (by omega) c hc i

example : 15 ≤ (19 : Fin 24).val := by decide

/-- The entire genuine raw-word final state has the independent visible-occurrence formula without a prepared representation premise.
Source: actual detector induction and proved complete original readout block semantics. -/
theorem depthWordFinalState_formula (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    depthWordFinalState mode gain eps word positions i =
      depthReadoutAttentionState mode eps positions (depthWordState mode gain eps word positions (depthDetectorCount mode)) i +
      ((∑ b : Fin 2, (if DepthReadoutPresence mode word b i then depthWordReadoutScale mode gain eps word positions i else 0) •
        EuclideanSpace.single (depthReadoutOutputCoordinate mode b) 1) +
        depthWordReadoutScale mode gain eps word positions i • depthAxis mode 19) := by
  have h := depthWordState_invariant mode gain eps heps hclip word hT positions (depthDetectorCount mode) (by omega)
  exact depthReadoutBlock_semantics mode eps heps hclip word hT positions _ h i

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- Genuine final constant and raw token types are derived through the entire original stack from the current raw letter.
Source: detector invariant from raw embeddings and preservation through both real readout residuals. -/
theorem depthWordFinalState_raw (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    depthWordFinalState mode gain eps word positions i (depthCoordinate mode 0) = 1 ∧
      ∀ b, depthWordFinalState mode gain eps word positions i (depthTypeCoordinate mode b) =
        if word.get i = some (depthBranchLetter b) then 1 else 0 := by
  have h := depthWordState_invariant mode gain eps heps hclip word hT positions (depthDetectorCount mode) (by omega)
  constructor
  · unfold depthWordFinalState
    rw [depthReadoutBlock_raw mode eps positions _ i 0 (by decide)]
    exact h.1 i
  · intro b
    unfold depthWordFinalState
    change blockForward (depthConfig mode) (depthReadoutBlock mode) eps positions _ i
      (depthCoordinate mode ⟨1 + b.val, by have hb := b.isLt; omega⟩) = _
    rw [depthReadoutBlock_raw mode eps positions _ i _ (by change 1 + b.val < 3; have hb := b.isLt; omega)]
    exact h.2.1 b i

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- Both true final indicators and their common real scale are derived on the actual raw-word output coordinates.
Source: complete genuine state formula, originally empty reserved axes and distinct actual output columns. -/
theorem depthWordFinalState_readout (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    (∀ b, depthWordFinalState mode gain eps word positions i (depthReadoutOutputCoordinate mode b) =
      if DepthReadoutPresence mode word b i then depthWordReadoutScale mode gain eps word positions i else 0) ∧
    depthWordFinalState mode gain eps word positions i (depthCoordinate mode 19) = depthWordReadoutScale mode gain eps word positions i := by
  have hf := depthWordFinalState_formula mode gain eps heps hclip word hT positions i
  have hz := depthWordReadout_upper_zero mode gain eps word positions i
  constructor
  · intro b
    rw [hf]
    simp only [WithLp.ofLp_add, Pi.add_apply, WithLp.ofLp_sum, Finset.sum_apply, PiLp.smul_apply, smul_eq_mul]
    have hzb : depthReadoutAttentionState mode eps positions
        (depthWordState mode gain eps word positions (depthDetectorCount mode)) i (depthReadoutOutputCoordinate mode b) = 0 :=
      hz ⟨17 + b.val, by have hb := b.isLt; omega⟩ (by change 15 ≤ 17 + b.val; omega)
    rw [hzb, zero_add]
    rw [Fintype.sum_eq_single b]
    · simp only [PiLp.single_apply, ite_true, mul_one]
      unfold depthAxis
      have hn : depthReadoutOutputCoordinate mode b ≠ depthCoordinate mode 19 := by
        intro he
        have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
        change 17 + b.val = 19 at hv
        have hb := b.isLt
        omega
      rw [PiLp.single_apply, ite_eq_right hn, mul_zero, add_zero]
    · intro other ho
      have hn : depthReadoutOutputCoordinate mode b ≠ depthReadoutOutputCoordinate mode other :=
        fun he => ho ((depthReadoutOutputCoordinate_injective mode) he).symm
      rw [PiLp.single_apply, ite_eq_right hn, mul_zero]
  · rw [hf]
    simp only [WithLp.ofLp_add, Pi.add_apply, WithLp.ofLp_sum, Finset.sum_apply, PiLp.smul_apply, smul_eq_mul]
    rw [hz 19 (by decide), depthAxis_coordinate]
    have hs : (∑ b : Fin 2, (if DepthReadoutPresence mode word b i then depthWordReadoutScale mode gain eps word positions i else 0) *
        EuclideanSpace.single (depthReadoutOutputCoordinate mode b) 1 (depthCoordinate mode 19)) = 0 := by
      apply Finset.sum_eq_zero
      intro b hb
      have hn : depthCoordinate mode 19 ≠ depthReadoutOutputCoordinate mode b := by
        intro he
        have hv := congrArg (fun c : Fin (depthConfig mode).d_model => c.val) he
        change 19 = 17 + b.val at hv
        have hb := b.isLt
        omega
      rw [PiLp.single_apply, ite_eq_right hn, mul_zero]
    rw [hs]
    norm_num

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- The actual raw-word final norm and genuine common scale satisfy uniform quantitative bounds without any encoder premise.
Source: real detector induction, complete readout growth and actual prenorm scale estimates. -/
theorem depthWordFinalState_bounds (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    (1 ≤ ‖depthWordFinalState mode gain eps word positions i‖ ∧ ‖depthWordFinalState mode gain eps word positions i‖ ≤ 1282) ∧
      depthScaleLower ^ 2 ≤ depthWordReadoutScale mode gain eps word positions i ∧ depthWordReadoutScale mode gain eps word positions i ≤ 128 := by
  have h := depthWordState_invariant mode gain eps heps hclip word hT positions (depthDetectorCount mode) (by omega)
  exact ⟨depthReadoutBlock_norm mode eps heps hclip word hT positions _ h i,
    depthReadoutBlock_scale_bounds mode eps heps hclip word positions _ h i⟩

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

end Transformer.GPTMini.Semantics
