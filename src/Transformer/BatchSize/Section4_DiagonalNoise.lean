/-
# Independent-coordinate Gaussian noise

arXiv:2506.12543v1, Section 4.3, Theorem 1.
The model is a genuine product probability measure, rather than an
assumed mean or covariance. Its signed covariance is diagonal.
-/

import Transformer.BatchSize.Section4_Coefficients

open MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.BatchSize

/-- The full minibatch gradient law with independent Gaussian
coordinates, Section 4.3, Theorem 1. -/
def diagonalNoiseLaw {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ) : Measure (Fin d → ℝ) :=
  Measure.pi (fun k => gradientNoiseLaw B (g k) (σ k))

/-- The independent-coordinate law is normalized, Section 4.3. -/
instance diagonalNoiseLaw_probability {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ) :
    IsProbabilityMeasure (diagonalNoiseLaw B g σ) := by
  dsimp [diagonalNoiseLaw]
  infer_instance

/-- Each coordinate has the stipulated Gaussian law, Section 4.3, Theorem 1. -/
theorem diagonalNoiseLaw_coordinate {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ) (k : Fin d) :
    HasLaw (fun z : Fin d → ℝ => z k) (gradientNoiseLaw B (g k) (σ k))
      (diagonalNoiseLaw B g σ) :=
  (measurePreserving_eval (fun r => gradientNoiseLaw B (g r) (σ r)) k).hasLaw

/-- The coordinates are independent, Section 4.3, Theorem 1. -/
theorem diagonalNoiseLaw_independent {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ) :
    iIndepFun (fun k (z : Fin d → ℝ) => z k) (diagonalNoiseLaw B g σ) :=
  iIndepFun_pi (fun _ => aemeasurable_id)

/-- Signed coordinates stay independent, Section 4.3, Theorem 1. -/
theorem diagonalNoiseLaw_sign_independent {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ) :
    iIndepFun (fun k (z : Fin d → ℝ) => Real.sign (z k)) (diagonalNoiseLaw B g σ) :=
  iIndepFun_pi (fun _ => measurable_real_sign.aemeasurable)

/-- The full stochastic gradient is unbiased in each coordinate,
Section 4.3, equation (2). -/
theorem diagonalNoiseLaw_mean {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ) (k : Fin d) :
    (∫ z, z k ∂diagonalNoiseLaw B g σ) = g k := by
  rw [(diagonalNoiseLaw_coordinate B g σ k).integral_eq]
  exact sgd_mean_gradient _ _ _

/-- The vector signed drift is componentwise erf, Section 4.3, equation (3). -/
theorem diagonalNoiseLaw_sign_mean {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ)
    (k : Fin d) (hB : 0 < B) (hσ : 0 < σ k) :
    (∫ z, Real.sign (z k) ∂diagonalNoiseLaw B g σ) = signResponse B (σ k) (g k) := by
  have h := (diagonalNoiseLaw_coordinate B g σ k).integral_comp
    measurable_real_sign.aestronglyMeasurable
  exact h.trans (signResponse_eq_mean B (g k) (σ k) hB hσ)

/-- Nonvacuity of the vector signed drift, Section 4.3. -/
example : 0 < (1 : ℕ) ∧ (0 : ℝ) < (![1] : Fin 1 → ℝ) 0 := by norm_num

/-- The coordinate variance is the exact signed diffusion covariance,
Section 4.3, Theorem 1, equation (3). -/
theorem diagonalNoiseLaw_sign_variance {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ)
    (k : Fin d) (hB : 0 < B) (hσ : 0 < σ k) :
    variance (fun z => Real.sign (z k)) (diagonalNoiseLaw B g σ) =
      1 - signResponse B (σ k) (g k) ^ 2 := by
  have hl := diagonalNoiseLaw_coordinate B g σ k
  have h := variance_map (X := Real.sign) measurable_real_sign.aemeasurable hl.aemeasurable
  rw [hl.map_eq] at h
  exact h.symm.trans (signResponse_variance B (g k) (σ k) hB hσ)

/-- Nonvacuity of the coordinate diffusion covariance, Section 4.3. -/
example : 0 < (64 : ℕ) ∧ (0 : ℝ) < (![1] : Fin 1 → ℝ) 0 := by norm_num

/-- The off-diagonal signed covariance vanishes, giving the diagonal
matrix in Section 4.3, Theorem 1, equation (3). -/
theorem diagonalNoiseLaw_sign_covariance {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ)
    (i j : Fin d) (hij : i ≠ j) :
    covariance (fun z => Real.sign (z i)) (fun z => Real.sign (z j))
      (diagonalNoiseLaw B g σ) = 0 := by
  have h := (diagonalNoiseLaw_sign_independent B g σ).indepFun hij
  apply h.covariance_eq_zero
  · exact MemLp.of_bound (measurable_real_sign.comp (measurable_pi_apply i)).aestronglyMeasurable
      1 (ae_of_all _ (fun z => by simpa [Real.norm_eq_abs] using sign_abs_le_one (z i)))
  · exact MemLp.of_bound (measurable_real_sign.comp (measurable_pi_apply j)).aestronglyMeasurable
      1 (ae_of_all _ (fun z => by simpa [Real.norm_eq_abs] using sign_abs_le_one (z j)))

/-- Nonvacuity of distinct Gaussian coordinates, Section 4.3. -/
example : (0 : Fin 2) ≠ 1 := by decide

/-- The covariance of the signed Gaussian vector is precisely the
diagonal matrix displayed in Theorem 1, equation (3), Section 4.3. -/
theorem diagonalNoiseLaw_sign_covariance_matrix {d : ℕ} (B : ℕ) (g σ : Fin d → ℝ)
    (hB : 0 < B) (hσ : ∀ k, 0 < σ k) :
    (Matrix.of (fun i j => covariance (fun z => Real.sign (z i))
      (fun z => Real.sign (z j)) (diagonalNoiseLaw B g σ))) =
        Matrix.diagonal (fun k => 1 - signResponse B (σ k) (g k) ^ 2) := by
  ext i j
  by_cases hij : i = j
  · subst j
    simp only [Matrix.of_apply, Matrix.diagonal_apply_eq]
    have hm : AEMeasurable (fun z : Fin d → ℝ => Real.sign (z i))
        (diagonalNoiseLaw B g σ) :=
      (measurable_real_sign.comp (measurable_pi_apply i)).aemeasurable
    rw [covariance_self hm,
      diagonalNoiseLaw_sign_variance B g σ i hB (hσ i)]
  · simp only [Matrix.of_apply, Matrix.diagonal_apply_ne _ hij]
    exact diagonalNoiseLaw_sign_covariance B g σ i j hij

/-- Nonvacuity of diagonal Gaussian noise, Section 4.3, Theorem 1. -/
example : 0 < (64 : ℕ) ∧ ∀ k : Fin 2, (0 : ℝ) < (![1, 2] : Fin 2 → ℝ) k := by
  constructor
  · norm_num
  · intro k
    fin_cases k <;> norm_num

end Transformer.BatchSize
