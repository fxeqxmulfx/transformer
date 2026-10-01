/-
# Integrated Gaussian generator identity with genuine random coefficients

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The bounded coefficients and bounded past observable give an integrable
joint domination, so expectation and time integration commute.
-/

import Transformer.BatchSize.Section4_GaussianParameters
import Transformer.BatchSize.Section4_VectorGaussianIdentity

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- Bounded actual amplitudes give the uniform frozen-generator
bound used in Section 4.3's martingale problem. -/
theorem frozenGaussianGenerator_uniform_bound {d : ℕ} (b : EucSpace d) (A : Fin d → ℝ)
    (M S : NNReal) (hb : ‖b‖ ≤ M) (hA : ∀ k, |A k| ≤ S)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) (y : EucSpace d) :
    |frozenGaussianGenerator b A φ y| ≤ M + (d : ℝ) * (S : ℝ) ^ 2 / 2 := by
  have hsum : (∑ k, (A k) ^ 2) ≤ (d : ℝ) * (S : ℝ) ^ 2 := by
    calc
      _ ≤ ∑ _ : Fin d, (S : ℝ) ^ 2 := Finset.sum_le_sum fun k _ => by
        have h := pow_le_pow_left₀ (abs_nonneg (A k)) (hA k) 2
        simpa only [sq_abs] using h
      _ = _ := by simp
  exact (frozenGaussianGenerator_abs_le b A φ hφ y).trans
    (add_le_add hb (div_le_div_of_nonneg_right hsum (by norm_num)))

/-- The genuine frozen Gaussian generator identity survives averaging
over arbitrary measurable random coefficients and a bounded past
observable, Section 4.3 (2)--(3). This is an unconditional identity for
the actual Gaussian integrals, with all Fubini hypotheses proved. -/
theorem randomGaussianFlow_generator_identity {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsFiniteMeasure P] {d : ℕ}
    (X b : Ω → EucSpace d) (A : Ω → Fin d → ℝ) (F : Ω → ℝ)
    (hX : Measurable X) (hb : Measurable b) (hA : ∀ k, Measurable (fun ω => A ω k))
    (hF : Measurable F) (M S : NNReal)
    (hbM : ∀ ω, ‖b ω‖ ≤ M) (hAS : ∀ ω k, |A ω k| ≤ S) (hF1 : ∀ ω, |F ω| ≤ 1)
    (φ : EucSpace d → ℝ) (hφ : BoundedSmoothTest 2 φ) (t : ℝ) (ht : 0 ≤ t) :
    (∫ ω, F ω * (gaussianVectorFlow φ (X ω) (b ω) (A ω) t - φ (X ω)) ∂P) =
      ∫ u in (0 : ℝ)..t, ∫ ω, F ω * gaussianVectorFlow
        (frozenGaussianGenerator (b ω) (A ω) φ) (X ω) (b ω) (A ω) u ∂P := by
  let K : ℝ := M + (d : ℝ) * (S : ℝ) ^ 2 / 2
  let G (u : ℝ) (ω : Ω) : ℝ := F ω * gaussianVectorFlow
    (frozenGaussianGenerator (b ω) (A ω) φ) (X ω) (b ω) (A ω) u
  have hGbound (u : ℝ) (ω : Ω) : ‖G u ω‖ ≤ K := by
    have hg := gaussianVectorFlow_abs_le (frozenGaussianGenerator (b ω) (A ω) φ)
      (X ω) (b ω) (A ω) K
      (frozenGaussianGenerator_uniform_bound (b ω) (A ω) M S (hbM ω) (hAS ω) φ hφ) u
    simpa only [G, Real.norm_eq_abs, abs_mul, one_mul] using
      (mul_le_mul (hF1 ω) hg (abs_nonneg _) (by norm_num : (0 : ℝ) ≤ 1))
  have hGmeas : Measurable (Function.uncurry G) :=
    (hF.comp measurable_snd).mul (gaussianGeneratorFlow_measurable_parameters
      (fun p : ℝ × Ω => X p.2) (fun p => b p.2) (fun p => A p.2) Prod.fst
      (hX.comp measurable_snd) (hb.comp measurable_snd)
      (fun k => (hA k).comp measurable_snd) measurable_fst φ hφ)
  let : IsFiniteMeasure ((volume : Measure ℝ).restrict (Set.uIoc 0 t)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_uIoc]; exact ENNReal.ofReal_lt_top⟩
  have hGi : Integrable (Function.uncurry G)
      (((volume : Measure ℝ).restrict (Set.uIoc 0 t)).prod P) :=
    (integrable_const K).mono' hGmeas.stronglyMeasurable.aestronglyMeasurable
      (Eventually.of_forall fun p => hGbound p.1 p.2)
  calc
    _ = ∫ ω, (∫ u in (0 : ℝ)..t, G u ω) ∂P := by
      apply integral_congr_ae
      exact Eventually.of_forall fun ω => by
        dsimp only [G]
        rw [gaussianVectorFlow_generator_identity φ hφ (X ω) (b ω) (A ω) t ht,
          add_sub_cancel_left, intervalIntegral.integral_const_mul]
    _ = _ := (intervalIntegral_integral_swap hGi).symm

/-- Joint nonvacuity of bounded random-generator hypotheses,
Section 4.3: a nonzero state and weight with bounded constant drift
and positive unit amplitudes. -/
example : Measurable (fun _ : Unit => EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) ∧
    Measurable (fun _ : Unit => (0 : EucSpace 1)) ∧
    (∀ k : Fin 1, Measurable (fun _ : Unit => (k : ℝ) + 1)) ∧
    Measurable (fun _ : Unit => (1 : ℝ)) ∧
    (‖(0 : EucSpace 1)‖ ≤ (1 : NNReal)) ∧
    (∀ k : Fin 1, |(k : ℝ) + 1| ≤ (1 : NNReal)) ∧
    |(1 : ℝ)| ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    BoundedSmoothTest 2 (fun _ : EucSpace 1 => (1 : ℝ)) := by
  refine ⟨measurable_const, measurable_const, fun _ => measurable_const,
    measurable_const, by simp, (fun k => by fin_cases k; norm_num), by norm_num, by norm_num, ?_⟩
  refine ⟨contDiff_const, ?_⟩
  intro j hj y
  cases j with
  | zero => simp [norm_iteratedFDeriv_zero]
  | succ j => rw [iteratedFDeriv_const_of_ne (by omega)]; simp

end Transformer.BatchSize
