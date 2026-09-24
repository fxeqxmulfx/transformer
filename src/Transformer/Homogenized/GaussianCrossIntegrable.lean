/-
# Homogenized Transformers — integrability of mixed Gaussian value moments

The scalar products in the matrix identity of `lem:lemma_app` are integrable
when the query/key-dependent vectors are bounded.  This complements the
integral identity in `GaussianCrossMoments.lean` and permits coordinatewise
evaluation of the Bochner integral defining the covariance kernel.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`.
-/

import Transformer.Homogenized.GaussianCrossMoments

open scoped BigOperators NNReal
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {d : ℕ}
    {V : Ω → Matrix (Fin d) (Fin d) ℝ} {Y : Ω → EucSpace d × EucSpace d}

/-- Mixed Gaussian linear forms with bounded coefficients are integrable.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`. -/
theorem integrable_inner_toEuclideanLin_mul_inner [IsFiniteMeasure P]
    (hY : Measurable Y)
    (hint : ∀ p q : Fin d × Fin d,
      Integrable (fun ω => V ω p.1 p.2 * V ω q.1 q.2) P)
    {C C' : ℝ} (hC : ∀ ω, ‖(Y ω).1‖ ≤ C) (hC' : ∀ ω, ‖(Y ω).2‖ ≤ C')
    (u v : EucSpace d) :
    Integrable (fun ω => inner (𝕜 := ℝ) u (Matrix.toEuclideanLin (V ω) (Y ω).1) *
      inner (𝕜 := ℝ) v (Matrix.toEuclideanLin (V ω) (Y ω).2)) P := by
  simp_rw [inner_toEuclideanLin_eq_sum]
  exact integrable_sum_mul_mul_sum_mul hY hint
    (measurable_cross_coeff u) (measurable_cross_coeff_right v)
    (fun ω p => abs_coef_le (hC ω) u p)
    (fun ω p => abs_coef_le (hC' ω) v p)

/-- The mixed scalar product is integrable under assumption (G).

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`. -/
theorem integrable_gaussian_value_cross {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {Y₁ Y₂ : Matrix (Fin d) (Fin d) ℝ → EucSpace d}
    (hY₁ : Measurable Y₁) (hY₂ : Measurable Y₂)
    {C₁ C₂ : ℝ} (hC₁ : ∀ A, ‖Y₁ A‖ ≤ C₁) (hC₂ : ∀ A, ‖Y₂ A‖ ≤ C₂)
    (u v : EucSpace d) :
    Integrable (fun θ => inner (𝕜 := ℝ) u (Matrix.toEuclideanLin θ.1 (Y₁ θ.2)) *
      inner (𝕜 := ℝ) v (Matrix.toEuclideanLin θ.1 (Y₂ θ.2))) ρ := by
  obtain ⟨Ω, _, P, Vr, Wr, Wr', _, hV, hW, hW', hind, hmV, -, -, rfl⟩ := hρ
  let A : Ω → Matrix (Fin d) (Fin d) ℝ := fun ω => Wr ω * (Wr' ω).transpose
  have hA : Measurable A := measurable_qk hW hW'
  have hVA : Measurable fun ω => ((Vr ω, A ω) : HeadParam d) := hV.prodMk hA
  have hY : Measurable fun ω => ((Y₁ (A ω), Y₂ (A ω)) : EucSpace d × EucSpace d) :=
    (hY₁.comp hA).prodMk (hY₂.comp hA)
  have hF : Measurable fun θ : HeadParam d =>
      inner (𝕜 := ℝ) u (Matrix.toEuclideanLin θ.1 (Y₁ θ.2)) *
        inner (𝕜 := ℝ) v (Matrix.toEuclideanLin θ.1 (Y₂ θ.2)) := by
    have hleft : Measurable fun θ : HeadParam d =>
        inner (𝕜 := ℝ) u (Matrix.toEuclideanLin θ.1 (Y₁ θ.2)) := by
      simp_rw [inner_toEuclideanLin_eq_sum]
      apply Finset.measurable_sum
      intro p _
      exact (Matrix.measurable_apply.comp measurable_fst).mul
        ((measurable_pi_apply p).comp ((measurable_coef u).comp
          (hY₁.comp measurable_snd)))
    have hright : Measurable fun θ : HeadParam d =>
        inner (𝕜 := ℝ) v (Matrix.toEuclideanLin θ.1 (Y₂ θ.2)) := by
      simp_rw [inner_toEuclideanLin_eq_sum]
      apply Finset.measurable_sum
      intro p _
      exact (Matrix.measurable_apply.comp measurable_fst).mul
        ((measurable_pi_apply p).comp ((measurable_coef v).comp
          (hY₂.comp measurable_snd)))
    exact hleft.mul hright
  apply (integrable_map_measure hF.aestronglyMeasurable hVA.aemeasurable).2
  exact integrable_inner_toEuclideanLin_mul_inner hY
    (integrable_value_mul_value hV hmV)
    (fun ω => hC₁ (A ω)) (fun ω => hC₂ (A ω)) u v

/-- The integrability hypotheses are satisfiable at the zero Gaussian law. -/
example (d : ℕ) :
    IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      Measurable (fun _ : Matrix (Fin d) (Fin d) ℝ => (0 : EucSpace d)) ∧
      (∀ _A : Matrix (Fin d) (Fin d) ℝ, ‖(0 : EucSpace d)‖ ≤ 0) :=
  ⟨isGaussianHeadLaw_dirac_zero d, measurable_const, fun _ => by simp⟩

end Homogenized
end Transformer
