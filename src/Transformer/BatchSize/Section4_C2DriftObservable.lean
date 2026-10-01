/-
# Lipschitz drift observables for the weak comparison

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Bounded first and second test derivatives, together with the genuine
bounded Lipschitz drift, control the change in its first-order generator.
-/

import Transformer.BatchSize.Section4_ComposedDerivativeLipschitz

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- The actual first-order drift observable in the optimizer
generator, Section 4.3 (2)--(3). -/
def driftTest {d : ℕ} (b : EucSpace d → EucSpace d) (φ : EucSpace d → ℝ)
    (x : EucSpace d) : ℝ := fderiv ℝ φ x (b x)

/-- A global C2 Hessian bound controls the actual derivative's
Lipschitz constant, Section 4.3, Theorem 1. -/
theorem C2_fderiv_lipschitz {d : ℕ} (φ : EucSpace d → ℝ) (R : NNReal)
    (hφ : ContDiff ℝ 2 φ) (hH : ∀ x, ‖iteratedFDeriv ℝ 2 φ x‖ ≤ R) :
    LipschitzWith R (fderiv ℝ φ) := by
  apply lipschitzWith_of_nnnorm_fderiv_le
    ((hφ.fderiv_right (m := 1) (by norm_num)).differentiable (by norm_num))
  intro x
  change ‖fderiv ℝ (fderiv ℝ φ) x‖ ≤ R
  rw [← norm_iteratedFDeriv_one (fderiv ℝ φ), norm_iteratedFDeriv_fderiv]
  exact hH x

/-- The actual drift observable is globally bounded on bounded
drifts and bounded-derivative tests, Section 4.3, Theorem 1. -/
theorem driftTest_abs_le {d : ℕ} (b : EucSpace d → EucSpace d) (φ : EucSpace d → ℝ)
    (R M : NNReal) (hd : ∀ x, ‖fderiv ℝ φ x‖ ≤ R) (hb : ∀ x, ‖b x‖ ≤ M)
    (x : EucSpace d) : |driftTest b φ x| ≤ (R : ℝ) * M := by
  have h := (fderiv ℝ φ x).le_opNorm (b x)
  change ‖fderiv ℝ φ x (b x)‖ ≤ _
  exact h.trans (mul_le_mul (hd x) (hb x) (norm_nonneg _) R.coe_nonneg)

/-- The first-order generator observable has Lipschitz constant
R*(M+K), with the actual derivative and drift norms,
Section 4.3 (2)--(3), Theorem 1. -/
theorem driftTest_lipschitz {d : ℕ} (b : EucSpace d → EucSpace d) (φ : EucSpace d → ℝ)
    (R M K : NNReal) (hφ : ContDiff ℝ 2 φ) (hK : LipschitzWith K b)
    (hb : ∀ x, ‖b x‖ ≤ M) (hd : ∀ x, ‖fderiv ℝ φ x‖ ≤ R)
    (hH : ∀ x, ‖iteratedFDeriv ℝ 2 φ x‖ ≤ R) :
    LipschitzWith (R * (M + K)) (driftTest b φ) := by
  have hD := C2_fderiv_lipschitz φ R hφ hH
  apply LipschitzWith.of_dist_le_mul
  intro x y
  rw [dist_eq_norm, dist_eq_norm]
  change |fderiv ℝ φ x (b x) - fderiv ℝ φ y (b y)| ≤ _
  have he : fderiv ℝ φ x (b x) - fderiv ℝ φ y (b y) =
      (fderiv ℝ φ x - fderiv ℝ φ y) (b x) + fderiv ℝ φ y (b x - b y) := by
    simp only [sub_apply, map_sub]
    ring
  rw [he]
  calc
    _ ≤ |(fderiv ℝ φ x - fderiv ℝ φ y) (b x)| + |fderiv ℝ φ y (b x - b y)| := abs_add_le _ _
    _ ≤ ‖fderiv ℝ φ x - fderiv ℝ φ y‖ * ‖b x‖ + ‖fderiv ℝ φ y‖ * ‖b x - b y‖ := by
      exact add_le_add ((fderiv ℝ φ x - fderiv ℝ φ y).le_opNorm (b x))
        ((fderiv ℝ φ y).le_opNorm (b x - b y))
    _ ≤ ((R : ℝ) * ‖x - y‖) * M + R * (K * ‖x - y‖) := by
      gcongr
      · exact hD.norm_sub_le x y
      · exact hb x
      · exact hd y
      · exact hK.norm_sub_le x y
    _ = _ := by simp only [NNReal.coe_mul, NNReal.coe_add]; ring

/-- Joint nonvacuity of the drift-observable hypotheses,
Section 4.3: zero bounded Lipschitz drift and a nonzero constant C2 test. -/
example : ContDiff ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) ∧
    LipschitzWith 0 (fun _ : EucSpace 1 => (0 : EucSpace 1)) ∧
    (∀ x : EucSpace 1, ‖(fun _ : EucSpace 1 => (0 : EucSpace 1)) x‖ ≤ (1 : NNReal)) ∧
    (∀ x : EucSpace 1, ‖fderiv ℝ (fun _ : EucSpace 1 => (1 : ℝ)) x‖ ≤ (1 : NNReal)) ∧
    (∀ x : EucSpace 1, ‖iteratedFDeriv ℝ 2 (fun _ : EucSpace 1 => (1 : ℝ)) x‖ ≤ (1 : NNReal)) := by
  refine ⟨contDiff_const, LipschitzWith.const _, by simp, ?_, ?_⟩
  · intro x
    rw [fderiv_const_apply]; simp
  · intro x
    rw [iteratedFDeriv_const_of_ne (by norm_num)]; simp

end Transformer.BatchSize
