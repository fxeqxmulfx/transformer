/-
# A zero signed Gram action for two collinear original neurons

ArXiv:2601.21366v2, §5.2, `thm: circle`, and §6,
`eq:ratio-const-correct`. At a nonzero original neuron input, an
actual activation pattern determines both signed coefficients.
If the original two-neuron signed Gram action is zero, the ReLU
potential is a constant for equal ray directions, and a quadratic
for opposite ray directions. The original primitive contributes its
actual constant, as proved in the preceding reduction module.
-/

import Transformer.Perceptron.Section5_TwoNeuronReduction

open Real Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- Zero signed Gram action gives the actual scalar coefficient
cancellation after collinear substitution.
Source: arXiv:2601.21366v2, §6, `eq:ratio-const-correct`, scalar zero. -/
theorem collinear_zero_signedGram_coefficient (u : EucSpace 2) (c : ℝ)
    (ω : Idx 2 → ℝ) (J : Finset (Idx 2)) (p : EucSpace 2)
    (hp : inner (𝕜 := ℝ) u p ≠ 0)
    (hz : ∀ x, signedGram ω ![u, c • u] J x = 0) :
    signedBranchWeight ω J 0 + signedBranchWeight ω J 1 * c ^ 2 = 0 := by
  have hu : u ≠ 0 := by intro h; exact hp (by rw [h, inner_zero_left])
  have h := hz p
  rw [signedGram_two_eq] at h
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, real_inner_smul_left, smul_smul] at h
  have heq : ((signedBranchWeight ω J 0 + signedBranchWeight ω J 1 * c ^ 2) *
      inner (𝕜 := ℝ) u p) • u = 0 := by
    calc
      _ = (signedBranchWeight ω J 0 * inner (𝕜 := ℝ) u p) • u +
          (signedBranchWeight ω J 1 * (c * inner (𝕜 := ℝ) u p) * c) • u := by
        rw [← add_smul]
        congr 1
        ring
      _ = 0 := h
  exact (mul_eq_zero.mp ((smul_eq_zero.mp heq).resolve_right hu)).resolve_right hp

/-- The actual pattern turns signed cancellation into the appropriate
original equal-ray or opposite-ray weight identity.
Source: arXiv:2601.21366v2, §5.2 and §6, `eq:ratio-const-correct`. -/
theorem collinear_zero_signedGram_weights (u : EucSpace 2) (c : ℝ)
    (ω : Idx 2 → ℝ) (J : Finset (Idx 2)) (p : EucSpace 2)
    (hpat : IsActivationPattern ![u, c • u] J p) (hp : inner (𝕜 := ℝ) u p ≠ 0)
    (hz : ∀ x, signedGram ω ![u, c • u] J x = 0) :
    if 0 ≤ c then ω 0 + ω 1 * c ^ 2 = 0 else ω 0 - ω 1 * c ^ 2 = 0 := by
  have hcoeff := collinear_zero_signedGram_coefficient u c ω J p hp hz
  by_cases hc0 : c = 0
  · subst c
    simp only [zero_pow (by decide : 2 ≠ 0), mul_zero, add_zero] at hcoeff
    have hw : ω 0 = 0 := (signedBranchWeight_eq_zero_iff ω J 0).mp hcoeff
    simp [hw]
  have hm₀ : (0 : Idx 2) ∈ J ↔ 0 < inner (𝕜 := ℝ) u p := by
    simpa only [Matrix.cons_val_zero] using
      mem_pattern_iff_inner_pos_of_ne ![u, c • u] J p hpat 0 hp
  have hm₁ : (1 : Idx 2) ∈ J ↔ 0 < c * inner (𝕜 := ℝ) u p := by
    simpa only [Matrix.cons_val_one, Matrix.cons_val_zero, real_inner_smul_left] using
      mem_pattern_iff_inner_pos_of_ne ![u, c • u] J p hpat 1
        (by
          change inner (𝕜 := ℝ) (c • u) p ≠ 0
          rw [real_inner_smul_left]
          exact mul_ne_zero hc0 hp)
  by_cases hc : 0 ≤ c
  · rw [ite_eq_left hc]
    have hcpos : 0 < c := lt_of_le_of_ne hc (Ne.symm hc0)
    by_cases hr : 0 < inner (𝕜 := ℝ) u p
    · have h₀ := hm₀.mpr hr
      have h₁ := hm₁.mpr (mul_pos hcpos hr)
      simpa only [signedBranchWeight, ite_eq_left h₀, ite_eq_left h₁] using hcoeff
    · have hrneg : inner (𝕜 := ℝ) u p < 0 := lt_of_le_of_ne (le_of_not_gt hr) hp
      have h₀ : (0 : Idx 2) ∉ J := fun h => hr (hm₀.mp h)
      have h₁ : (1 : Idx 2) ∉ J := by
        intro h
        exact (not_lt_of_ge (mul_neg_of_pos_of_neg hcpos hrneg).le) (hm₁.mp h)
      simp only [signedBranchWeight, ite_eq_right h₀, ite_eq_right h₁] at hcoeff
      linarith
  · rw [ite_eq_right hc]
    have hcneg : c < 0 := lt_of_not_ge hc
    by_cases hr : 0 < inner (𝕜 := ℝ) u p
    · have h₀ := hm₀.mpr hr
      have h₁ : (1 : Idx 2) ∉ J := by
        intro h
        have hn : c * inner (𝕜 := ℝ) u p < 0 := by nlinarith
        exact (not_lt_of_ge hn.le) (hm₁.mp h)
      simp only [signedBranchWeight, ite_eq_left h₀, ite_eq_right h₁] at hcoeff
      linarith
    · have hrneg : inner (𝕜 := ℝ) u p < 0 := lt_of_le_of_ne (le_of_not_gt hr) hp
      have h₀ : (0 : Idx 2) ∉ J := fun h => hr (hm₀.mp h)
      have h₁ := hm₁.mpr (mul_pos_of_neg_of_neg hcneg hrneg)
      simp only [signedBranchWeight, ite_eq_right h₀, ite_eq_left h₁] at hcoeff
      linarith

/-- A collinear two-neuron zero scalar branch with a genuine pattern
point forces the actual original primitive potential to be analytic.
Source: arXiv:2601.21366v2, §5.2, the analytic-potential contradiction
of `thm: circle`, and §6, `eq:ratio-const-correct`, scalar zero.
The original arbitrary primitive is retained with its actual constant. -/
theorem isAnalyticOnSphere_collinear_of_zero_signedGram (φ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (u : EucSpace 2) (c : ℝ)
    (ω : Idx 2 → ℝ) (J : Finset (Idx 2)) (p : EucSpace 2)
    (hpat : IsActivationPattern ![u, c • u] J p) (hp : inner (𝕜 := ℝ) u p ≠ 0)
    (hz : ∀ x, signedGram ω ![u, c • u] J x = 0) :
    IsAnalyticOnSphere φ ω ![u, c • u] := by
  have hw := collinear_zero_signedGram_weights u c ω J p hpat hp hz
  let C := φ 0 * (ω 0 + ω 1)
  by_cases hc : 0 ≤ c
  · rw [ite_eq_left hc] at hw
    refine ⟨fun _ => C, fun _ _ => analyticAt_const, fun x _ => ?_⟩
    rw [potential_reluPrimitive_eq φ hφ]
    simp only [potential, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
      real_inner_smul_left]
    rw [reluSq_mul_nonneg c hc]
    change C = ω 0 * max (inner (𝕜 := ℝ) u x) 0 ^ 2 +
      ω 1 * (c ^ 2 * max (inner (𝕜 := ℝ) u x) 0 ^ 2) + C
    calc
      _ = (ω 0 + ω 1 * c ^ 2) * max (inner (𝕜 := ℝ) u x) 0 ^ 2 + C := by rw [hw]; ring
      _ = _ := by ring
  · rw [ite_eq_right hc] at hw
    refine ⟨fun x => ω 0 * inner (𝕜 := ℝ) u x ^ 2 + C, ?_, fun x _ => ?_⟩
    · intro x _
      exact (analyticAt_const.fun_mul (((innerSL ℝ u).analyticAt x).fun_pow 2)).fun_add analyticAt_const
    · rw [potential_reluPrimitive_eq φ hφ]
      simp only [potential, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
        real_inner_smul_left]
      rw [reluSq_mul_nonpos c (le_of_not_ge hc)]
      have heq : ω 1 * c ^ 2 = ω 0 := by linarith
      change ω 0 * inner (𝕜 := ℝ) u x ^ 2 + C =
        ω 0 * max (inner (𝕜 := ℝ) u x) 0 ^ 2 +
          ω 1 * (c ^ 2 * max (-inner (𝕜 := ℝ) u x) 0 ^ 2) + C
      calc
        _ = ω 0 * (max (inner (𝕜 := ℝ) u x) 0 ^ 2 + max (-inner (𝕜 := ℝ) u x) 0 ^ 2) + C := by
          rw [reluSq_add_neg]
        _ = _ := by rw [← mul_assoc, heq]; ring

/-- The same-ray cancellation hypotheses hold for two actual equal
nonzero circle neurons with opposite output weights. -/
example : (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    IsActivationPattern ![(basePoint 1 : EucSpace 2), (1 : ℝ) • (basePoint 1 : EucSpace 2)]
      Finset.univ (basePoint 1 : EucSpace 2) ∧
    inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) (basePoint 1 : EucSpace 2) ≠ 0 ∧
    ∀ x : EucSpace 2, signedGram (![1, -1] : Idx 2 → ℝ)
      ![(basePoint 1 : EucSpace 2), (1 : ℝ) • (basePoint 1 : EucSpace 2)] Finset.univ x = 0 := by
  refine ⟨hasDerivAt_reluSq, ⟨?_, fun j hj => by simp at hj⟩, ?_, ?_⟩
  · intro j hj
    fin_cases j <;> simp [basePoint]
  · simp [basePoint]
  · intro x
    rw [signedGram_two_eq]
    simp [signedBranchWeight]

/-- Opposite original ray directions with equal weights give the
analytic quadratic alternative, retaining the squared-ReLU primitive. -/
example : IsAnalyticOnSphere (fun s : ℝ => max s 0 ^ 2) (![1, 1] : Idx 2 → ℝ)
    ![(basePoint 1 : EucSpace 2), (-1 : ℝ) • (basePoint 1 : EucSpace 2)] := by
  apply isAnalyticOnSphere_collinear_of_zero_signedGram (fun s => max s 0 ^ 2)
    hasDerivAt_reluSq (basePoint 1 : EucSpace 2) (-1) ![1, 1] {0} (basePoint 1 : EucSpace 2)
  · constructor
    · intro j hj
      fin_cases j <;> simp_all [basePoint]
    · intro j hj
      fin_cases j <;> simp_all [basePoint]
  · simp [basePoint]
  · intro x
    rw [signedGram_two_eq]
    simp [signedBranchWeight]

end Transformer.Perceptron
