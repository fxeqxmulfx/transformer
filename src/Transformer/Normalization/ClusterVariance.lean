/-
# Pointwise variance decay inside a narrow cluster

The uniform-attention contribution and its combination with the
weight error in Appendix C, proof of Theorem 4.3 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.ClusterResidual

open Set
open scoped BigOperators

namespace Transformer.Normalization

variable {d n : ℕ}

/-- The empirical mean of unit directions has norm at most one.
Source: arXiv:2510.22026v2, Appendix C, proof of Theorem 4.3. -/
theorem norm_tokenMean_le_one (hn : 0 < n) (Θ : Idx n → EucSpace d)
    (hunit : ∀ j, ‖Θ j‖ = 1) : ‖tokenMean Θ‖ ≤ 1 := by
  have h := norm_attentionVec_le_one d n 0 (0 : ParamMatrix d) (0 : ParamMatrix d)
    (ContinuousLinearMap.id ℝ _) ContinuousLinearMap.norm_id_le Θ hunit ⟨0, hn⟩
  simpa [attentionVec, tokenMean] using h

/-- A unit singleton meets the hypotheses. -/
example : 0 < 1 ∧ ∀ j : Idx 1, ‖(fun _ : Idx 1 => EuclideanSpace.single (0 : Fin 1)
    (1 : ℝ)) j‖ = 1 := ⟨Nat.one_pos, fun _ => by simp⟩

/-- The projection of the mean contributes `c_j²-‖mean‖²` for unit tokens.
Source: arXiv:2510.22026v2, Appendix C, the identity for `I₁`. -/
theorem inner_center_proj_mean (Θ : Idx n → EucSpace d) (j : Idx n)
    (hj : ‖Θ j‖ = 1) :
    inner (𝕜 := ℝ) (Θ j - tokenMean Θ) (proj d (Θ j) (tokenMean Θ)) =
      inner (𝕜 := ℝ) (Θ j) (tokenMean Θ) ^ 2 - ‖tokenMean Θ‖ ^ 2 := by
  simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_right,
    real_inner_self_eq_norm_sq, hj, real_inner_comm (tokenMean Θ) (Θ j)]
  ring

/-- A unit singleton supplies the norm hypothesis. -/
example : ‖EuclideanSpace.single (0 : Fin 1) (1 : ℝ)‖ = 1 := by simp

/-- Summing the uniform-attention contribution gives bounds proportional
to the variance, with lower cone height `m` and inverse speeds in `[a,b]`.
Source: arXiv:2510.22026v2, Appendix C, the estimates on `I₁`. -/
theorem uniform_variance_sum_bounds (hn : 0 < n) (m a b : ℝ) (ha : 0 ≤ a)
    (hb : 0 ≤ b) (Θ : Idx n → EucSpace d) (s : Idx n → ℝ)
    (hunit : ∀ j, ‖Θ j‖ = 1)
    (hpair : ∀ j k, m ≤ inner (𝕜 := ℝ) (Θ j) (Θ k))
    (hs : ∀ j, a ≤ s j ∧ s j ≤ b) :
    -(b * ∑ j, ‖Θ j - tokenMean Θ‖ ^ 2) ≤
      ∑ j, s j * inner (𝕜 := ℝ) (Θ j - tokenMean Θ) (proj d (Θ j) (tokenMean Θ)) ∧
    (∑ j, s j * inner (𝕜 := ℝ) (Θ j - tokenMean Θ) (proj d (Θ j) (tokenMean Θ))) ≤
      -(a * m * ∑ j, ‖Θ j - tokenMean Θ‖ ^ 2) := by
  let c : Idx n → ℝ := fun j => inner (𝕜 := ℝ) (Θ j) (tokenMean Θ)
  let D : Idx n → ℝ := fun j => ‖tokenMean Θ‖ ^ 2 - c j ^ 2
  have hc : ∀ j, m ≤ c j ∧ c j ≤ 1 := by
    intro j
    refine ⟨?_, ?_⟩
    · have h := inner_tokenMean_ge_of_localCone hn (1 - m) Θ
        (fun i k => by simpa using hpair i k) j
      simpa [c] using h
    · have h := real_inner_le_norm (Θ j) (tokenMean Θ)
      rw [hunit j, one_mul] at h
      exact h.trans (norm_tokenMean_le_one hn Θ hunit)
  have hD : ∀ j, 0 ≤ D j := by
    intro j
    have h := abs_real_inner_le_norm (Θ j) (tokenMean Θ)
    rw [hunit j, one_mul] at h
    dsimp [D, c]
    have hsquare := (sq_le_sq₀ (abs_nonneg _) (norm_nonneg _)).mpr h
    rw [sq_abs] at hsquare
    exact sub_nonneg.mpr hsquare
  have hsumD : ∑ j, D j = ∑ j, c j * (1 - c j) := by
    simp only [D, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    simp_rw [mul_sub, mul_one, ← sq]
    rw [Finset.sum_sub_distrib, show ∑ j, c j = (n : ℝ) * ‖tokenMean Θ‖ ^ 2
      from sum_inner_tokenMean hn Θ]
  have hsumc : ∑ j, (1 - c j) = ∑ j, ‖Θ j - tokenMean Θ‖ ^ 2 := by
    have hvar := intraClusterVar_eq_one_sub_norm_mean_sq hn (fun _ => Θ) 0 hunit
    have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
    change (n : ℝ)⁻¹ * (∑ j, ‖Θ j - tokenMean Θ‖ ^ 2) = _ at hvar
    have hv := congrArg (fun v : ℝ => (n : ℝ) * v) hvar
    rw [← mul_assoc, mul_inv_cancel₀ hn', one_mul] at hv
    simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    rw [show ∑ j, c j = (n : ℝ) * ‖tokenMean Θ‖ ^ 2 from sum_inner_tokenMean hn Θ]
    linarith
  have hsum : m * (∑ j, ‖Θ j - tokenMean Θ‖ ^ 2) ≤ ∑ j, D j ∧
      (∑ j, D j) ≤ ∑ j, ‖Θ j - tokenMean Θ‖ ^ 2 := by
    rw [hsumD, ← hsumc, Finset.mul_sum]
    constructor <;> apply Finset.sum_le_sum <;> intro j _
    · exact mul_le_mul_of_nonneg_right (hc j).1 (sub_nonneg.mpr (hc j).2)
    · simpa using mul_le_mul_of_nonneg_right (hc j).2 (sub_nonneg.mpr (hc j).2)
  have hweighted : a * (∑ j, D j) ≤ ∑ j, s j * D j ∧
      (∑ j, s j * D j) ≤ b * (∑ j, D j) := by
    simp only [Finset.mul_sum]
    constructor <;> apply Finset.sum_le_sum <;> intro j _
    · exact mul_le_mul_of_nonneg_right (hs j).1 (hD j)
    · exact mul_le_mul_of_nonneg_right (hs j).2 (hD j)
  have heq : (∑ j, s j * inner (𝕜 := ℝ) (Θ j - tokenMean Θ)
      (proj d (Θ j) (tokenMean Θ))) = -(∑ j, s j * D j) := by
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro j _
    rw [inner_center_proj_mean Θ j (hunit j)]
    dsimp [D, c]
    ring
  rw [heq]
  constructor
  · exact neg_le_neg (hweighted.2.trans (mul_le_mul_of_nonneg_left hsum.2 hb))
  · have h := (mul_le_mul_of_nonneg_left hsum.1 ha).trans hweighted.1
    simpa only [mul_assoc] using neg_le_neg h

/-- Unit copies, cone height one and unit speeds meet the assumptions. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    0 < 1 ∧ (0 : ℝ) ≤ 1 ∧ (∀ j : Idx 1, ‖(fun _ : Idx 1 => u) j‖ = 1) ∧
      (∀ j k : Idx 1, (1 : ℝ) ≤ inner (𝕜 := ℝ) ((fun _ : Idx 1 => u) j)
        ((fun _ : Idx 1 => u) k)) ∧ ∀ j : Idx 1,
      (1 : ℝ) ≤ (fun _ : Idx 1 => (1 : ℝ)) j ∧ (fun _ : Idx 1 => (1 : ℝ)) j ≤ 1 := by simp

/-- In a cluster of height `3/4`, almost uniform weights and comparable
inverse speeds give a variance derivative between `-4g Var` and `-g Var/9`.
Source: arXiv:2510.22026v2, Appendix C, the combination `Var'=I₁+I₂` in
the proof of Theorem 4.3. -/
theorem variance_decay_of_uniform_weights (hn : 0 < n) (β g : ℝ) (hg : 0 ≤ g)
    (Q K : ParamMatrix d) (Θ : Idx n → EucSpace d) (s : Idx n → ℝ)
    (hunit : ∀ j, ‖Θ j‖ = 1)
    (hpair : ∀ j k, (3 / 4 : ℝ) ≤ inner (𝕜 := ℝ) (Θ j) (Θ k))
    (hs : ∀ j, (2 / 3 : ℝ) * g ≤ s j ∧ s j ≤ (4 / 3 : ℝ) * g)
    (hw : ∀ j k, |attentionWeight β Q K Θ j k - (n : ℝ)⁻¹| ≤ (1 / 3 : ℝ) / n) :
    let V := (n : ℝ)⁻¹ * ∑ j, ‖Θ j - tokenMean Θ‖ ^ 2;
    let v := 2 / (n : ℝ) * ∑ j, inner (𝕜 := ℝ) (Θ j - tokenMean Θ)
      (s j • proj d (Θ j) (attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j));
    -(4 * g * V) ≤ v ∧ v ≤ -((1 / 9 : ℝ) * g * V) := by
  let S : ℝ := ∑ j, ‖Θ j - tokenMean Θ‖ ^ 2
  let I₁ : ℝ := ∑ j, s j * inner (𝕜 := ℝ) (Θ j - tokenMean Θ)
    (proj d (Θ j) (tokenMean Θ))
  let I₂ : ℝ := ∑ j, s j * inner (𝕜 := ℝ) (Θ j - tokenMean Θ)
    (proj d (Θ j) (attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j - tokenMean Θ))
  have hI₁ := uniform_variance_sum_bounds hn (3 / 4) ((2 / 3) * g) ((4 / 3) * g)
    (by positivity) (by positivity) Θ s hunit hpair hs
  have hI₂ := residual_variance_sum_bound β (1 / 3) ((4 / 3) * g) (by norm_num)
    (by positivity) Q K Θ s hunit (fun j => ⟨by linarith [(hs j).1], (hs j).2⟩) hw
  change -((4 / 3) * g * S) ≤ I₁ ∧ I₁ ≤ -((2 / 3) * g * (3 / 4) * S) at hI₁
  change |I₂| ≤ (4 / 3) * g * (1 / 3) * S at hI₂
  have heq : (∑ j, inner (𝕜 := ℝ) (Θ j - tokenMean Θ)
      (s j • proj d (Θ j) (attentionVec d n β Q K (ContinuousLinearMap.id ℝ _) Θ j))) =
      I₁ + I₂ := by
    simp only [I₁, I₂, ← Finset.sum_add_distrib, real_inner_smul_right]
    apply Finset.sum_congr rfl
    intro j _
    simp only [proj, inner_sub_right, real_inner_smul_right]
    ring
  have hS : 0 ≤ S := Finset.sum_nonneg (fun j _ => sq_nonneg _)
  have hprod : 0 ≤ g * S := mul_nonneg hg hS
  have hraw : -(2 * g * S) ≤ I₁ + I₂ ∧ I₁ + I₂ ≤ -((1 / 18 : ℝ) * g * S) := by
    constructor <;> nlinarith [hI₁.1, hI₁.2, le_of_abs_le hI₂, neg_le_of_abs_le hI₂]
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hmul0 : 0 ≤ 2 / (n : ℝ) := by positivity
  have hlo := mul_le_mul_of_nonneg_left hraw.1 hmul0
  have hhi := mul_le_mul_of_nonneg_left hraw.2 hmul0
  dsimp only
  rw [heq]
  constructor
  · convert hlo using 1
    dsimp [S]
    ring
  · convert hhi using 1
    dsimp [S]
    ring

/-- Uniform attention and unit speeds on a unit singleton satisfy all
the pointwise assumptions with `g=1`. -/
example : let u := EuclideanSpace.single (0 : Fin 1) (1 : ℝ)
    (0 : ℝ) ≤ 1 ∧ (∀ j : Idx 1, ‖(fun _ : Idx 1 => u) j‖ = 1) ∧
      (∀ j k : Idx 1, (3 / 4 : ℝ) ≤ inner (𝕜 := ℝ) ((fun _ : Idx 1 => u) j)
        ((fun _ : Idx 1 => u) k)) ∧
      (∀ j : Idx 1, (2 / 3 : ℝ) * 1 ≤ (fun _ : Idx 1 => (1 : ℝ)) j ∧
        (fun _ : Idx 1 => (1 : ℝ)) j ≤ (4 / 3 : ℝ) * 1) ∧
      ∀ j k : Idx 1, |attentionWeight 0 (idParams 1 0) (idParams 1 0)
        (fun _ : Idx 1 => u) j k - ((1 : ℕ) : ℝ)⁻¹| ≤ (1 / 3 : ℝ) / 1 := by
  norm_num [attentionWeight, Perspective.softmaxWeight]

end Transformer.Normalization
