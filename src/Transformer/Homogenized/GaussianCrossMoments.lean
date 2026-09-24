/-
# Homogenized Transformers — mixed second moments of a Gaussian value matrix

The matrix identity used in the proof of `lem:lemma_app` of
arXiv:2604.01978v1: for bounded vectors `a,b` determined by the query/key
matrix and independent of the value matrix `V`,

  `E[⟨u,Va⟩ ⟨v,Vb⟩] = σ_V² ⟨u,v⟩ E[⟨a,b⟩]`.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`.
-/

import Transformer.Homogenized.GaussianMoments
import Transformer.Homogenized.GaussianEntries
import Mathlib.MeasureTheory.Function.SpecialFunctions.Inner

open scoped BigOperators NNReal
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {d : ℕ}
    {V : Ω → Matrix (Fin d) (Fin d) ℝ} {Y : Ω → EucSpace d × EucSpace d}

/-- The diagonal coefficient sum factors into two Euclidean inner products. -/
theorem sum_cross_coeff (u v a b : EucSpace d) :
    (∑ p : Fin d × Fin d, (u p.1 * a p.2) * (v p.1 * b p.2)) =
      inner (𝕜 := ℝ) u v * inner (𝕜 := ℝ) a b := by
  simp only [EuclideanSpace.inner_eq_star_dotProduct, star_trivial,
    dotProduct, Fintype.sum_prod_type, Finset.sum_mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

/-- The coefficient functions for two linear forms are measurable. -/
theorem measurable_cross_coeff (u : EucSpace d) :
    Measurable fun (z : EucSpace d × EucSpace d) (p : Fin d × Fin d) =>
      u p.1 * z.1 p.2 :=
  Measurable.of_eval fun p => by fun_prop

/-- The second coefficient function is measurable. -/
theorem measurable_cross_coeff_right (v : EucSpace d) :
    Measurable fun (z : EucSpace d × EucSpace d) (p : Fin d × Fin d) =>
      v p.1 * z.2 p.2 :=
  Measurable.of_eval fun p => by fun_prop

/-- The mixed-moment identity `E[⟨u,Va⟩⟨v,Vb⟩] =
`σ² ⟨u,v⟩ E[⟨a,b⟩]` for bounded `a,b` independent of `V`.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`. -/
theorem integral_inner_toEuclideanLin_mul_inner [IsFiniteMeasure P]
    (hV : Measurable V) (hY : Measurable Y) (hVY : IndepFun V Y P) {σ2 : ℝ}
    (hmom : ∀ p q : Fin d × Fin d,
      ∫ ω, V ω p.1 p.2 * V ω q.1 q.2 ∂P = if p = q then σ2 else 0)
    (hint : ∀ p q : Fin d × Fin d,
      Integrable (fun ω => V ω p.1 p.2 * V ω q.1 q.2) P)
    {C C' : ℝ} (hC : ∀ ω, ‖(Y ω).1‖ ≤ C) (hC' : ∀ ω, ‖(Y ω).2‖ ≤ C')
    (u v : EucSpace d) :
    ∫ ω, inner (𝕜 := ℝ) u (Matrix.toEuclideanLin (V ω) (Y ω).1) *
        inner (𝕜 := ℝ) v (Matrix.toEuclideanLin (V ω) (Y ω).2) ∂P =
      σ2 * inner (𝕜 := ℝ) u v *
        ∫ ω, inner (𝕜 := ℝ) (Y ω).1 (Y ω).2 ∂P := by
  have hflat : Measurable fun (M : Matrix (Fin d) (Fin d) ℝ) (p : Fin d × Fin d) =>
      M p.1 p.2 := Measurable.of_eval fun _ => Matrix.measurable_apply
  have hleft : ∀ ω (p : Fin d × Fin d), |u p.1 * (Y ω).1 p.2| ≤ ‖u‖ * C :=
    fun ω p => abs_coef_le (hC ω) u p
  have hright : ∀ ω (p : Fin d × Fin d), |v p.1 * (Y ω).2 p.2| ≤ ‖v‖ * C' :=
    fun ω p => abs_coef_le (hC' ω) v p
  have key := integral_sum_mul_mul_sum_mul
    (X := fun ω (p : Fin d × Fin d) => V ω p.1 p.2) (Y := Y)
    (c := fun z p => u p.1 * z.1 p.2) (c' := fun z p => v p.1 * z.2 p.2)
    (hflat.comp hV) hY (hVY.comp hflat measurable_id) hmom hint
    (measurable_cross_coeff u) (measurable_cross_coeff_right v) hleft hright
  simp_rw [← inner_toEuclideanLin_eq_sum] at key
  simp_rw [sum_cross_coeff] at key
  simpa only [integral_const_mul, mul_assoc] using key

/-- The mixed-moment hypotheses are satisfiable even in the degenerate
Gaussian case: a zero value matrix and zero query/key-dependent vectors. -/
example (d : ℕ) :
    Measurable (fun _ : Unit => (0 : Matrix (Fin d) (Fin d) ℝ)) ∧
      Measurable (fun _ : Unit => ((0, 0) : EucSpace d × EucSpace d)) ∧
      IndepFun (fun _ : Unit => (0 : Matrix (Fin d) (Fin d) ℝ))
        (fun _ : Unit => ((0, 0) : EucSpace d × EucSpace d)) (Measure.dirac ()) ∧
      (∀ p q : Fin d × Fin d,
        ∫ _ω, (0 : ℝ) * 0 ∂(Measure.dirac ()) = if p = q then 0 else 0) ∧
      (∀ _p _q : Fin d × Fin d,
        Integrable (fun _ω : Unit => (0 : ℝ) * 0) (Measure.dirac ())) ∧
      (∀ _ω : Unit, ‖(0 : EucSpace d)‖ ≤ 0) :=
  ⟨measurable_const, measurable_const, indepFun_const_left _ _, fun _ _ => by simp,
    fun _ _ => integrable_const _, fun _ => by simp⟩

/-- The mixed second moment of the value matrix under assumption (G), with
two bounded vectors determined by the query/key matrix.  This is the matrix
expectation computed in the proof of `lem:lemma_app`.

Source: arXiv:2604.01978v1, proof of `lem:lemma_app`. -/
theorem integral_gaussian_value_cross {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {Y₁ Y₂ : Matrix (Fin d) (Fin d) ℝ → EucSpace d}
    (hY₁ : Measurable Y₁) (hY₂ : Measurable Y₂)
    {C₁ C₂ : ℝ} (hC₁ : ∀ A, ‖Y₁ A‖ ≤ C₁) (hC₂ : ∀ A, ‖Y₂ A‖ ≤ C₂)
    (u v : EucSpace d) :
    ∫ θ, inner (𝕜 := ℝ) u (Matrix.toEuclideanLin θ.1 (Y₁ θ.2)) *
        inner (𝕜 := ℝ) v (Matrix.toEuclideanLin θ.1 (Y₂ θ.2)) ∂ρ =
      (σV : ℝ) ^ 2 * inner (𝕜 := ℝ) u v *
        ∫ θ, inner (𝕜 := ℝ) (Y₁ θ.2) (Y₂ θ.2) ∂ρ := by
  obtain ⟨Ω, _, P, Vr, Wr, Wr', _, hV, hW, hW', hind, hmV, -, -, rfl⟩ := hρ
  let A : Ω → Matrix (Fin d) (Fin d) ℝ := fun ω => Wr ω * (Wr' ω).transpose
  have hA : Measurable A := measurable_qk hW hW'
  have hVA : Measurable fun ω => ((Vr ω, A ω) : HeadParam d) := hV.prodMk hA
  have hY : Measurable fun ω => ((Y₁ (A ω), Y₂ (A ω)) : EucSpace d × EucSpace d) :=
    (hY₁.comp hA).prodMk (hY₂.comp hA)
  have hindY : IndepFun Vr
      (fun ω => ((Y₁ (A ω), Y₂ (A ω)) : EucSpace d × EucSpace d)) P :=
    (indepFun_value_qk hV hW hW' hind).comp measurable_id (hY₁.prodMk hY₂)
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
  have hG : Measurable fun θ : HeadParam d => inner (𝕜 := ℝ) (Y₁ θ.2) (Y₂ θ.2) :=
    (hY₁.comp measurable_snd).inner (hY₂.comp measurable_snd)
  rw [integral_map hVA.aemeasurable hF.aestronglyMeasurable,
    integral_map hVA.aemeasurable hG.aestronglyMeasurable]
  exact integral_inner_toEuclideanLin_mul_inner hV hY hindY
    (integral_value_mul_value hV hind hmV) (integrable_value_mul_value hV hmV)
    (fun ω => hC₁ (A ω)) (fun ω => hC₂ (A ω)) u v

/-- Assumption (G) and the bounded-vector hypotheses are simultaneously
satisfied at the zero Gaussian law. -/
example (d : ℕ) :
    IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) ∧
      Measurable (fun _ : Matrix (Fin d) (Fin d) ℝ => (0 : EucSpace d)) ∧
      (∀ _A : Matrix (Fin d) (Fin d) ℝ, ‖(0 : EucSpace d)‖ ≤ 0) :=
  ⟨isGaussianHeadLaw_dirac_zero d, measurable_const, fun _ => by simp⟩

end Homogenized
end Transformer
