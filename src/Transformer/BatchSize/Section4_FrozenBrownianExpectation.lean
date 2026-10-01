/-
# Actual frozen Brownian expectations equal their Gaussian averages

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The observable can depend on both the random left state and the future
state, as needed for the variable frozen generator.
-/

import Transformer.BatchSize.Section4_FrozenBrownianState

open MeasureTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A bounded observable of the actual frozen Brownian state has
exactly its Gaussian conditional average, even after multiplying by
an arbitrary bounded jointly adapted past observable,
Section 4.3 (2)--(3). -/
theorem frozenBrownianState_weighted_expectation {d : ℕ}
    (b : EucSpace d → EucSpace d) (a : EucSpace d → Fin d → ℝ)
    (hb : Continuous b) (ha : ∀ k, Continuous (fun x => a x k))
    (s t : ℝ≥0) (hst : s ≤ t) (X : BrownianSample d → EucSpace d)
    (hX : StronglyMeasurable[brownianFiltration d s] X) (F : BrownianSample d → ℝ)
    (hF : StronglyMeasurable[brownianFiltration d s] F) (hF1 : ∀ ω, |F ω| ≤ 1)
    (Ψ : EucSpace d × EucSpace d → ℝ) (hΨ : Measurable Ψ)
    (C : ℝ) (hC : ∀ x y, |Ψ (x, y)| ≤ C) :
    (∫ ω, F ω * Ψ (X ω, frozenBrownianState b a X s t ω) ∂brownianNoiseLaw d) =
      ∫ ω, F ω * gaussianVectorFlow (fun y => Ψ (X ω, y))
        (X ω) (b (X ω)) (a (X ω)) ((t : ℝ) - s) ∂brownianNoiseLaw d := by
  let δ : ℝ := (t : ℝ) - s
  let H (ω : BrownianSample d) := (X ω, F ω)
  let g (p : (EucSpace d × ℝ) × (Fin d → ℝ)) : ℝ :=
    p.1.2 * Ψ (p.1.1, gaussianAffineState (p.1.1 + δ • b p.1.1) (a p.1.1) p.2)
  have hgm : Measurable g := by
    have hx : Measurable (fun p : (EucSpace d × ℝ) × (Fin d → ℝ) => p.1.1) :=
      measurable_fst.comp measurable_fst
    have hs := gaussianAffineState_measurable_parameters _ _ _
      (hx.add ((hb.measurable.comp hx).const_smul δ))
      (fun k => (ha k).measurable.comp hx)
      (fun k => (measurable_pi_apply k).comp measurable_snd)
    exact (measurable_snd.comp measurable_fst).mul (hΨ.comp (hx.prodMk hs))
  have hgb (ω : BrownianSample d) (z : Fin d → ℝ) : |g (H ω, z)| ≤ C := by
    exact (abs_mul _ _).trans_le (by
      simpa only [one_mul] using mul_le_mul (hF1 ω) (hC _ _) (abs_nonneg _)
        (by norm_num : (0 : ℝ) ≤ 1))
  have heq := vectorBrownianIncrement_integral_adapted s t hst H (hX.prodMk hF) g hgm C hgb
  change (∫ ω, F ω * Ψ (X ω, frozenBrownianState b a X s t ω) ∂brownianNoiseLaw d) = _ at heq
  rw [heq]
  apply integral_congr_ae
  exact Eventually.of_forall fun ω => by
    dsimp only [g, H, δ]
    simp_rw [gaussianAffineState_rescale]
    exact integral_const_mul (F ω) _

/-- Joint nonvacuity of weighted frozen-transition hypotheses,
Section 4.3: a genuine past state, nonzero bounded weight and a
bounded nonconstant observable of the future state. -/
example : (1 : ℝ≥0) ≤ 2 ∧
    StronglyMeasurable[brownianFiltration 1 1] (vectorBrownian 1 1) ∧
    StronglyMeasurable[brownianFiltration 1 1] (fun _ : BrownianSample 1 => (1 : ℝ)) ∧
    |(1 : ℝ)| ≤ 1 ∧
    Measurable (fun p : EucSpace 1 × EucSpace 1 => Real.sin (p.2 0)) ∧
    (∀ y : EucSpace 1, |Real.sin (y 0)| ≤ 1) := by
  refine ⟨by norm_num, Filtration.stronglyAdapted_natural
    (fun t => (vectorBrownian_measurable 1 t).stronglyMeasurable) 1,
    stronglyMeasurable_const, by norm_num, ?_, fun _ => Real.abs_sin_le_one _⟩
  exact Real.continuous_sin.measurable.comp
    ((PiLp.continuous_apply 2 (fun _ : Fin 1 => ℝ) 0).measurable.comp measurable_snd)

end Transformer.BatchSize
