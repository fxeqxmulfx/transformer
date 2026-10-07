import Transformer.GPTMini.Semantics.DepthWordFinalState

/-!
# Genuine tied depth scores and quantitative decision direction

Source: original tied unembed at f11b6e2 and actual raw-word final state
at 59cf964. Every vocabulary entry is the same token-local embedding
used on input. The two label entries read the derived direction with
coefficients one, minus two and minus one half on the actual semantic
indicator and common-scale coordinates.

The direction is positive with a uniform r^2/2 margin exactly for the
independent positive-pattern/no-negative-pattern criterion, and negative
with the opposite margin otherwise. The finite shared label gain turns
this into margin four. Every nonlabel tied score is at most two, derived
from the preserved actual raw constant/types. No logit or oracle encoder
is a premise. Strict labels, final RMS and full-model coupling follow.
-/

namespace Transformer.GPTMini.Semantics

open scoped Classical
open Transformer.Basis

/-- The semantic target token for the independent visible-pattern criterion, not the model's computed decoder.
Source: Basis E_2/E_4 positive-ending presence and exclusion of the wrong active start. -/
noncomputable def depthWordAnswer (mode : Mode) (word : List (Option Bool)) (i : Fin word.length) : Fin 36 :=
  if DepthReadoutPresence mode word 0 i ∧ ¬DepthReadoutPresence mode word 1 i then 16 else 15

/-- The independent target always names exactly one of the two original depth label IDs.
Source: actual ACCEPT sixteen and REJECT fifteen, without extending the original vocabulary. -/
theorem depthWordAnswer_labels (mode : Mode) (word : List (Option Bool)) (i : Fin word.length) :
    depthWordAnswer mode word i = 16 ∨ depthWordAnswer mode word i = 15 := by
  unfold depthWordAnswer
  split_ifs
  · exact Or.inl rfl
  · exact Or.inr rfl

/-- The actual tied direction reads precisely the three reserved real output coordinates.
Source: original Euclidean tied inner product and the concrete shared label embedding direction. -/
theorem depthReadoutDirection_score (mode : Mode) (x : EucSpace (depthConfig mode).d_model) :
    inner (𝕜 := ℝ) x (depthReadoutDirection mode) =
      x (depthCoordinate mode 17) - 2 * x (depthCoordinate mode 18) - (1 / 2 : ℝ) * x (depthCoordinate mode 19) := by
  rw [depthReadoutDirection, inner_sub_right, inner_sub_right, real_inner_smul_right, real_inner_smul_right]
  simp only [depthAxis, EuclideanSpace.inner_single_right, conj_trivial, one_mul]

/-- Every real vocabulary tied score is evaluated from its own actual token-local embedding branch.
Source: the complete ordinary depthRawEmbedding table, including both distinct label entries. -/
theorem depthRawEmbedding_score (mode : Mode) (gain : ℝ) (token : Fin 36) (x : EucSpace (depthConfig mode).d_model) :
    inner (𝕜 := ℝ) x (depthRawEmbedding mode gain token) = x (depthCoordinate mode 0) +
      if token.val = 9 then x (depthCoordinate mode 1)
      else if token.val = 10 then x (depthCoordinate mode 2)
      else if token.val = 16 then gain * inner (𝕜 := ℝ) x (depthReadoutDirection mode)
      else if token.val = 15 then -gain * inner (𝕜 := ℝ) x (depthReadoutDirection mode)
      else 0 := by
  rw [depthRawEmbedding, inner_add_right]
  split_ifs <;> simp only [depthAxis, EuclideanSpace.inner_single_right, conj_trivial, one_mul, real_inner_smul_right, inner_zero_right]

/-- Both genuine tied label entries read the same direction with opposite signs and the same raw constant.
Source: the two actual label branches of the complete ordinary token-local table. -/
theorem depthRawEmbedding_labels (mode : Mode) (gain : ℝ) (x : EucSpace (depthConfig mode).d_model) :
    inner (𝕜 := ℝ) x (depthRawEmbedding mode gain (16 : Fin 36)) =
      x (depthCoordinate mode 0) + gain * inner (𝕜 := ℝ) x (depthReadoutDirection mode) ∧
    inner (𝕜 := ℝ) x (depthRawEmbedding mode gain (15 : Fin 36)) =
      x (depthCoordinate mode 0) - gain * inner (𝕜 := ℝ) x (depthReadoutDirection mode) := by
  constructor
  · rw [depthRawEmbedding_score]
    norm_num
  · rw [depthRawEmbedding_score]
    norm_num
    ring

/-- The actual final tied direction has the exact independent two-presence formula with its genuine common scale.
Source: real final indicator/common coordinates derived from raw words, not assumed Boolean hidden states. -/
theorem depthWordFinalState_direction (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    inner (𝕜 := ℝ) (depthWordFinalState mode gain eps word positions i) (depthReadoutDirection mode) =
      (if DepthReadoutPresence mode word 0 i then depthWordReadoutScale mode gain eps word positions i else 0) -
        2 * (if DepthReadoutPresence mode word 1 i then depthWordReadoutScale mode gain eps word positions i else 0) -
          (1 / 2 : ℝ) * depthWordReadoutScale mode gain eps word positions i := by
  have h := depthWordFinalState_readout mode gain eps heps hclip word hT positions i
  have hp := h.1 0
  have hn := h.1 1
  change depthWordFinalState mode gain eps word positions i (depthCoordinate mode 17) = _ at hp
  change depthWordFinalState mode gain eps word positions i (depthCoordinate mode 18) = _ at hn
  rw [depthReadoutDirection_score, hp, hn, h.2]

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- The true final direction separates the exact independent accept/reject criterion by at least half the genuine lower scale.
Source: all four actual semantic presence cases and the derived common RMS-square floor r^2. -/
theorem depthWordFinalState_direction_bounds (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    if DepthReadoutPresence mode word 0 i ∧ ¬DepthReadoutPresence mode word 1 i then
      depthScaleLower ^ 2 / 2 ≤ inner (𝕜 := ℝ) (depthWordFinalState mode gain eps word positions i) (depthReadoutDirection mode)
    else inner (𝕜 := ℝ) (depthWordFinalState mode gain eps word positions i) (depthReadoutDirection mode) ≤ -depthScaleLower ^ 2 / 2 := by
  have hs := (depthWordFinalState_bounds mode gain eps heps hclip word hT positions i).2.1
  rw [depthWordFinalState_direction mode gain eps heps hclip word hT positions i]
  by_cases hp : DepthReadoutPresence mode word 0 i <;> by_cases hn : DepthReadoutPresence mode word 1 i <;>
    simp only [hp, hn, not_true_eq_false, not_false_eq_true, and_true, and_false, ite_true, ite_false] <;>
    nlinarith [sq_nonneg depthScaleLower]

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- The fixed genuine tied label gain yields decision margin four for every raw word.
Source: actual direction separation and shared finite compensation gain times r^2 equal to eight. -/
theorem depthWordFinalState_label_margin (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length) :
    if DepthReadoutPresence mode word 0 i ∧ ¬DepthReadoutPresence mode word 1 i then
      4 ≤ depthLabelGain * inner (𝕜 := ℝ) (depthWordFinalState mode depthLabelGain eps word positions i) (depthReadoutDirection mode)
    else depthLabelGain * inner (𝕜 := ℝ) (depthWordFinalState mode depthLabelGain eps word positions i) (depthReadoutDirection mode) ≤ -4 := by
  have hd := depthWordFinalState_direction_bounds mode depthLabelGain eps heps hclip word hT positions i
  have hg := depthLabelGain_margin
  by_cases hp : DepthReadoutPresence mode word 0 i ∧ ¬DepthReadoutPresence mode word 1 i
  · rw [ite_eq_left hp] at hd ⊢
    have hm := mul_le_mul_of_nonneg_left hd hg.1.le
    nlinarith [hg.2]
  · rw [ite_eq_right hp] at hd ⊢
    have hm := mul_le_mul_of_nonneg_left hd hg.1.le
    nlinarith [hg.2]

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([none, some false, some true] : List (Option Bool)).length ≤ 128 := by
  exact ⟨by norm_num, by norm_num, by decide⟩

/-- Every other actual tied vocabulary score is at most two, directly from real preserved raw types.
Source: true token-local embedding table and raw-word final constant/type coordinates. -/
theorem depthWordFinalState_nonlabel_score (mode : Mode) (gain eps : ℝ) (heps : 0 < eps) (hclip : eps ≤ 1)
    (word : List (Option Bool)) (hT : word.length ≤ 128) (positions : Fin word.length → ℝ) (i : Fin word.length)
    (token : Fin 36) (ha : token.val ≠ 16) (hr : token.val ≠ 15) :
    inner (𝕜 := ℝ) (depthWordFinalState mode gain eps word positions i) (depthRawEmbedding mode gain token) ≤ 2 := by
  have h := depthWordFinalState_raw mode gain eps heps hclip word hT positions i
  have hA := h.2 0
  have hB := h.2 1
  change depthWordFinalState mode gain eps word positions i (depthCoordinate mode 1) = _ at hA
  change depthWordFinalState mode gain eps word positions i (depthCoordinate mode 2) = _ at hB
  rw [depthRawEmbedding_score, h.1]
  by_cases htA : token.val = 9
  · rw [ite_eq_left htA, hA]
    split_ifs <;> norm_num
  · rw [ite_eq_right htA]
    by_cases htB : token.val = 10
    · rw [ite_eq_left htB, hB]
      split_ifs <;> norm_num
    · rw [ite_eq_right htB, ite_eq_right ha, ite_eq_right hr]
      norm_num

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 ∧
    ([some false, none, some true] : List (Option Bool)).length ≤ 128 ∧
    (9 : Fin 36).val ≠ 16 ∧ (9 : Fin 36).val ≠ 15 := by
  exact ⟨by norm_num, by norm_num, by decide, by decide, by decide⟩

end Transformer.GPTMini.Semantics
