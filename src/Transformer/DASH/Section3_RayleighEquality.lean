/-
# DASH — equality and strictness of the Rayleigh bound

arXiv:2602.02016v2, §3.4. Equality holds for every nonzero vector in
the maximal eigenspace, not only for a single distinguished eigenvector.
-/

import Transformer.DASH.Section3_Rayleigh

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.DASH

variable {n : ℕ}

/-- The Rayleigh quotient reaches its spectral upper bound exactly when all
nonzero coordinates lie in the maximal eigenspace. This corrects the source's
strict inequality for every other vector, including repeated eigenvalues.
Source: arXiv:2602.02016v2, §3.4, the paragraph defining `v_PI`. -/
theorem spectralRayleigh_eq_max_iff (s z : Fin n → ℝ) (μ : ℝ)
    (hz : z ≠ 0) (hupper : ∀ i, s i ≤ μ) :
    spectralRayleigh s z = μ ↔ ∀ i, z i ≠ 0 → s i = μ := by
  have hden := (coordinateEnergy_pos z hz).ne'
  constructor
  · intro h i hi
    have hnum : (∑ i, s i * z i ^ 2) = μ * coordinateEnergy z :=
      (div_eq_iff hden).mp h
    have hsum : (∑ i, (μ - s i) * z i ^ 2) = 0 := by
      simp_rw [sub_mul]
      rw [Finset.sum_sub_distrib, ← Finset.mul_sum, hnum]
      simp only [coordinateEnergy, sub_self]
    have hterm : (μ - s i) * z i ^ 2 = 0 :=
      (Finset.sum_eq_zero_iff_of_nonneg
        (fun j _ => mul_nonneg (sub_nonneg.mpr (hupper j)) (sq_nonneg _))).mp
        hsum i (Finset.mem_univ i)
    have hdiff := (mul_eq_zero.mp hterm).resolve_right (pow_ne_zero 2 hi)
    linarith
  · intro h
    have hnum : (∑ i, s i * z i ^ 2) = μ * coordinateEnergy z := by
      rw [coordinateEnergy, Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i hi
      by_cases hzi : z i = 0
      · simp [hzi]
      · rw [h i hzi]
    rw [spectralRayleigh, hnum]
    exact mul_div_cancel_right₀ _ hden

/-- Equality-characterization hypotheses are satisfiable,
arXiv:2602.02016v2, §3.4. -/
example : (fun _ : Fin 1 => (1 : ℝ)) ≠ 0 ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) := by
  constructor
  · intro h
    have := congrFun h 0
    norm_num at this
  · norm_num

/-- Strict inequality requires a nonzero component at a strictly smaller
eigenvalue. Numerical inexactness alone does not imply strictness.
Source: arXiv:2602.02016v2, §3.4, the claimed strict Rayleigh inequality. -/
theorem spectralRayleigh_lt_max_iff (s z : Fin n → ℝ) (μ : ℝ)
    (hz : z ≠ 0) (hupper : ∀ i, s i ≤ μ) :
    spectralRayleigh s z < μ ↔ ∃ i, z i ≠ 0 ∧ s i < μ := by
  constructor
  · intro h
    by_contra hn
    push Not at hn
    have heq : spectralRayleigh s z = μ :=
      (spectralRayleigh_eq_max_iff s z μ hz hupper).2
        (fun i hi => le_antisymm (hupper i) (hn i hi))
    linarith
  · rintro ⟨i, hi, hsi⟩
    apply lt_of_le_of_ne (spectralRayleigh_le s z μ hz hupper)
    intro heq
    have := (spectralRayleigh_eq_max_iff s z μ hz hupper).1 heq i hi
    linarith

/-- Strict-inequality hypotheses are satisfiable with two distinct eigenvalues.
Source: arXiv:2602.02016v2, §3.4. -/
example : (![1, 1] : Fin 2 → ℝ) ≠ 0 ∧
    (∀ i : Fin 2, (![2, 1] : Fin 2 → ℝ) i ≤ 2) := by
  constructor
  · intro h
    have := congrFun h 0
    norm_num at this
  · intro i
    fin_cases i <;> norm_num

/-- Equality characterization for the actual matrix quotient, including
maximal eigenspaces with multiplicity. Source: arXiv:2602.02016v2, §3.4. -/
theorem rayleigh_eq_max_iff (Q : Matrix (Fin n) (Fin n) ℝ)
    (s z : Fin n → ℝ) (μ : ℝ) (hQ : Orthogonal Q)
    (hz : z ≠ 0) (hupper : ∀ i, s i ≤ μ) :
    rayleigh (spectralMatrix Q s) (Q *ᵥ z) = μ ↔ ∀ i, z i ≠ 0 → s i = μ := by
  rw [rayleigh_spectral Q s z hQ]
  exact spectralRayleigh_eq_max_iff s z μ hz hupper

/-- Actual matrix-quotient assumptions are satisfiable,
arXiv:2602.02016v2, §3.4. -/
example : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) ∧
    (fun _ : Fin 1 => (1 : ℝ)) ≠ 0 ∧
    (∀ i : Fin 1, (fun _ => (1 : ℝ)) i ≤ 1) := by
  refine ⟨?_, ?_, by norm_num⟩
  · simp [Orthogonal, Muon.OrthonormalColumns]
  · intro h
    have := congrFun h 0
    norm_num at this

end Transformer.DASH
