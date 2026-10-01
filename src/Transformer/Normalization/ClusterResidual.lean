/-
# The attention-weight error in variance decay

Appendix C, proof of Theorem 4.3 of arXiv:2510.22026v2, the term `I₂`.
Centering the weight error before Cauchy--Schwarz gives a direct variance bound.
-/

import Transformer.Normalization.ClusterGeometry
import Transformer.Normalization.ClusterWeights
import Mathlib.Algebra.Order.BigOperators.Ring.Finset

open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- The deviation of attention from the mean is the weight error applied
to centered tokens. Source: arXiv:2510.22026v2, Appendix C, the decomposition
of the attention error in `I₂`. -/
theorem attentionVec_id_sub_mean (β : ℝ) (Q K : ParamMatrix d)
    (Θ : Idx n → EucSpace d) (j : Idx n) :
    attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j - tokenMean Θ =
      ∑ k, (attentionWeight β Q K Θ j k - (n : ℝ)⁻¹) • (Θ k - tokenMean Θ) := by
  have hn : (n : ℝ) ≠ 0 := by exact_mod_cast (Fin.pos j).ne'
  simp_rw [smul_sub, sub_smul]
  simp only [Finset.sum_sub_distrib, ← Finset.smul_sum, ← Finset.sum_smul,
    sum_attentionWeight, one_smul, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul, inv_mul_cancel₀ hn, one_smul]
  rw [attentionVec_id_eq_sum]
  change (∑ k, attentionWeight β Q K Θ j k • Θ k) - tokenMean Θ =
    ((∑ k, attentionWeight β Q K Θ j k • Θ k) - tokenMean Θ) -
      (tokenMean Θ - tokenMean Θ)
  simp

/-- A uniform weight error of size `ε/n` bounds the vector error by the
average centered norm. Source: arXiv:2510.22026v2, Appendix C, the norm
estimate on the attention error. -/
theorem norm_attentionVec_id_sub_mean_le (β ε : ℝ) (Q K : ParamMatrix d)
    (Θ : Idx n → EucSpace d) (j : Idx n)
    (hw : ∀ k, |attentionWeight β Q K Θ j k - (n : ℝ)⁻¹| ≤ ε / n) :
    ‖attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j - tokenMean Θ‖ ≤
      (ε / n) * ∑ k, ‖Θ k - tokenMean Θ‖ := by
  rw [attentionVec_id_sub_mean]
  apply (norm_sum_le _ _).trans
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro k _
  rw [norm_smul, Real.norm_eq_abs]
  exact mul_le_mul_of_nonneg_right (hw k) (norm_nonneg _)

/-- Equal scores on a singleton have zero weight error. -/
example : ∀ k : Idx 1,
    |attentionWeight 0 (idParams 1 0) (idParams 1 0)
      (fun _ : Idx 1 => (0 : EucSpace 1)) 0 k - ((1 : ℕ) : ℝ)⁻¹| ≤ (0 : ℝ) / 1 := by
  intro k
  simp [attentionWeight, Perspective.softmaxWeight]

/-- The weighted projection error is bounded by `b ε` times the total
squared centered norm. Source: arXiv:2510.22026v2, Appendix C, the bound
on `I₂`, with inverse speeds bounded by `b`. -/
theorem residual_variance_sum_bound (β ε b : ℝ) (hε : 0 ≤ ε) (hb : 0 ≤ b)
    (Q K : ParamMatrix d) (Θ : Idx n → EucSpace d) (s : Idx n → ℝ)
    (hunit : ∀ j, ‖Θ j‖ = 1) (hs : ∀ j, 0 ≤ s j ∧ s j ≤ b)
    (hw : ∀ j k, |attentionWeight β Q K Θ j k - (n : ℝ)⁻¹| ≤ ε / n) :
    |∑ j, s j * inner (𝕜 := ℝ) (Θ j - tokenMean Θ)
      (proj d (Θ j) (attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j - tokenMean Θ))|
      ≤ b * ε * ∑ j, ‖Θ j - tokenMean Θ‖ ^ 2 := by
  let S : ℝ := ∑ j, ‖Θ j - tokenMean Θ‖
  have hconst : 0 ≤ ε / (n : ℝ) := div_nonneg hε (Nat.cast_nonneg n)
  have hj : ∀ j, |s j * inner (𝕜 := ℝ) (Θ j - tokenMean Θ)
      (proj d (Θ j) (attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j - tokenMean Θ))|
      ≤ b * ‖Θ j - tokenMean Θ‖ * ((ε / n) * S) := by
    intro j
    rw [abs_mul, abs_of_nonneg (hs j).1]
    have h := (abs_real_inner_le_norm (Θ j - tokenMean Θ) _).trans
      (mul_le_mul_of_nonneg_left ((norm_proj_le (hunit j) _).trans
        (norm_attentionVec_id_sub_mean_le β ε Q K Θ j (hw j))) (norm_nonneg _))
    calc s j * _ ≤ b * (‖Θ j - tokenMean Θ‖ * ((ε / n) * S)) :=
        mul_le_mul (hs j).2 h (abs_nonneg _) hb
      _ = _ := by ring
  have hsum := (Finset.abs_sum_le_sum_abs _ Finset.univ).trans
    (Finset.sum_le_sum (s := Finset.univ) fun j _ => hj j)
  have hcs : S ^ 2 ≤ (n : ℝ) * ∑ j, ‖Θ j - tokenMean Θ‖ ^ 2 := by
    have h := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ
      (fun j : Idx n => ‖Θ j - tokenMean Θ‖) (fun _ => (1 : ℝ))
    simpa [S, mul_comm] using h
  by_cases hn : n = 0
  · subst n
    simp
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have heq : (∑ j, b * ‖Θ j - tokenMean Θ‖ * ((ε / n) * S)) = b * (ε / n) * S ^ 2 := by
    simp only [← Finset.sum_mul, ← Finset.mul_sum]
    dsimp [S]
    ring
  rw [heq] at hsum
  apply hsum.trans
  have h := mul_le_mul_of_nonneg_left hcs (mul_nonneg hb hconst)
  convert h using 1
  field_simp

/-- Unit singleton data with unit speeds and zero weight error witnesses
the norm, speed and weight assumptions. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧ (∀ j : Idx 1, ‖(fun _ : Idx 1 => u) j‖ = 1) ∧
      (∀ j : Idx 1, 0 ≤ (fun _ : Idx 1 => (1 : ℝ)) j ∧
        (fun _ : Idx 1 => (1 : ℝ)) j ≤ 1) ∧
      ∀ j k : Idx 1, |attentionWeight 0 (idParams 1 0) (idParams 1 0)
        (fun _ : Idx 1 => u) j k - ((1 : ℕ) : ℝ)⁻¹| ≤ (0 : ℝ) / 1 := by
  simp [attentionWeight, Perspective.softmaxWeight]

end Transformer.Normalization
