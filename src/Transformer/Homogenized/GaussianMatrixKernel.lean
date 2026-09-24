/-
# Homogenized Transformers — the Gaussian matrix covariance identity

For bounded query/key-dependent vectors `a,b`, the Gaussian value matrix
satisfies `E[(V a) ⟨V b,v⟩] = σ_V² E[⟨a,b⟩] v`.  This is the vector form of the
calculation used in `lem:lemma_app`, before identifying `a,b` with softmax
barycenters.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`.
-/

import Transformer.Homogenized.GaussianCrossIntegrable
import Mathlib.MeasureTheory.SpecificCodomains.WithLp

open scoped BigOperators NNReal
open MeasureTheory

namespace Transformer
namespace Homogenized

variable {d : ℕ}

/-- The Gaussian value-matrix covariance is a scalar multiple of the identity
when its input vectors depend only on the independent query/key matrix.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`. -/
theorem integral_gaussian_value_matrix_cross {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {Y₁ Y₂ : Matrix (Fin d) (Fin d) ℝ → EucSpace d}
    (hY₁ : Measurable Y₁) (hY₂ : Measurable Y₂)
    {C₁ C₂ : ℝ} (hC₁ : ∀ A, ‖Y₁ A‖ ≤ C₁) (hC₂ : ∀ A, ‖Y₂ A‖ ≤ C₂)
    (v : EucSpace d) :
    ∫ θ, (inner (𝕜 := ℝ) (Matrix.toEuclideanLin θ.1 (Y₂ θ.2)) v) •
        Matrix.toEuclideanLin θ.1 (Y₁ θ.2) ∂ρ =
      ((σV : ℝ) ^ 2 * ∫ θ, inner (𝕜 := ℝ) (Y₁ θ.2) (Y₂ θ.2) ∂ρ) • v := by
  let f : HeadParam d → EucSpace d := fun θ =>
    (inner (𝕜 := ℝ) (Matrix.toEuclideanLin θ.1 (Y₂ θ.2)) v) •
      Matrix.toEuclideanLin θ.1 (Y₁ θ.2)
  have hint : ∀ k : Fin d, Integrable (fun θ => f θ k) ρ := by
    intro k
    have h := integrable_gaussian_value_cross hρ hY₁ hY₂ hC₁ hC₂
      (EuclideanSpace.single k (1 : ℝ)) v
    convert h using 1
    ext θ
    simp [f, EuclideanSpace.inner_single_left, real_inner_comm, mul_comm]
  ext k
  rw [show (∫ θ, (inner (𝕜 := ℝ) (Matrix.toEuclideanLin θ.1 (Y₂ θ.2)) v) •
      Matrix.toEuclideanLin θ.1 (Y₁ θ.2) ∂ρ) k = ∫ θ, f θ k ∂ρ from
    eval_integral_piLp hint k]
  have h := integral_gaussian_value_cross hρ hY₁ hY₂ hC₁ hC₂
    (EuclideanSpace.single k (1 : ℝ)) v
  simpa [f, EuclideanSpace.inner_single_left, real_inner_comm, mul_comm, mul_assoc,
    mul_left_comm]
    using h

/-- The matrix identity has a witness at the zero Gaussian law. -/
example (d : ℕ) :
    IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      Measurable (fun _ : Matrix (Fin d) (Fin d) ℝ => (0 : EucSpace d)) ∧
      (∀ _A : Matrix (Fin d) (Fin d) ℝ, ‖(0 : EucSpace d)‖ ≤ 0) :=
  ⟨isGaussianHeadLaw_dirac_zero d, measurable_const, fun _ => by simp⟩

end Homogenized
end Transformer
