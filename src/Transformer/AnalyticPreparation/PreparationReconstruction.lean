/-
# Real analytic preparation: PreparationReconstruction

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.PreparedFunctions

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Pointwise reconstruction once sequence division and convergence are
available.  Keeping the convergence hypotheses explicit makes this lemma
reusable for both the public existence and uniqueness arguments.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparation_factorization_of_sequence_factorization {n d : ℕ}
    {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (hr0 : 0 < r)
    (htop : originWeightedCoeffs p r hr d ≠ 0)
    (z : Base n) (w : ℝ)
    (hfac : monomialSeq d =
      convolution (normalizedPreparationQuotient p r hr d z)
        (analyticNormalizedCoefficientMap p r hr d z) +
          normalizedPreparationRemainder p r hr d z)
    (hsupp : seqHighShift d (normalizedPreparationRemainder p r hr d z) = 0)
    (hw : ‖(r : ℝ)⁻¹ * w‖ < 1)
    (hrecon : evalL1PowerSeries ((weightedCoefficientSeries p r).sum z)
        ((r : ℝ)⁻¹ * w) = f (z, w))
    (hQne : preparationQuotientEval p r hr d (z, w) ≠ 0) :
    f (z, w) = preparationUnit p r hr d (z, w) *
      preparedPolynomial d (preparationCoefficient p r hr d) (z, w) := by
  let t : ℝ := (r : ℝ)⁻¹ * w
  let q : OriginSeq := normalizedPreparationQuotient p r hr d z
  let rem : OriginSeq := normalizedPreparationRemainder p r hr d z
  let C : OriginSeq := (weightedCoefficientSeries p r).sum z
  let D : ℝ := originWeightedCoeffs p r hr d
  have hD : D ≠ 0 := htop
  have hEvalFac := congrArg (fun a : OriginSeq ↦ evalL1PowerSeries a t) hfac
  have hG : analyticNormalizedCoefficientMap p r hr d z = D⁻¹ • C := rfl
  have hseries :
      t ^ d = evalL1PowerSeries q t * (D⁻¹ * evalL1PowerSeries C t) +
        ∑ i : Fin d, rem i * t ^ (i : ℕ) := by
    rw [evalL1PowerSeries_monomialSeq d hw, evalL1PowerSeries_add,
      evalL1PowerSeries_convolution _ _ hw, hG, evalL1PowerSeries_smul,
      evalL1PowerSeries_eq_sum_fin_of_highShift_eq_zero d rem hsupp hw] at hEvalFac
    simpa only [q, rem] using hEvalFac
  rw [preparedPolynomial_preparationCoefficient p r hr d hr0 z w]
  rw [← hrecon]
  simp only [preparationUnit, preparationQuotientEval]
  rw [← mul_pow]
  change evalL1PowerSeries C t =
    (D * (r : ℝ)⁻¹ ^ d) * (evalL1PowerSeries q t)⁻¹ *
      ((r : ℝ) ^ d * (t ^ d - ∑ i : Fin d, rem i * t ^ (i : ℕ)))
  have hrC : (r : ℝ) ≠ 0 := by exact_mod_cast hr0.ne'
  have hQ : evalL1PowerSeries q t ≠ 0 := hQne
  let R : ℝ := ∑ i : Fin d, rem i * t ^ (i : ℕ)
  calc
    evalL1PowerSeries C t =
        D * (evalL1PowerSeries q t)⁻¹ * (t ^ d - R) := by
      rw [hseries]
      field_simp [hD, hQ]
      ring
    _ = (D * (r : ℝ)⁻¹ ^ d) * (evalL1PowerSeries q t)⁻¹ *
        ((r : ℝ) ^ d * (t ^ d - R)) := by
      have hrpow : (r : ℝ) ^ d * (r : ℝ)⁻¹ ^ d = 1 := by
        rw [← mul_pow, mul_inv_cancel₀ hrC, one_pow]
      calc
        D * (evalL1PowerSeries q t)⁻¹ * (t ^ d - R) =
            D * (evalL1PowerSeries q t)⁻¹ * (1 * (t ^ d - R)) := by rw [one_mul]
        _ = _ := by rw [← hrpow]; ring

/-- The existence half of classical real Weierstrass preparation, in the
exact public predicate.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_isWeierstrassPreparation {n d : ℕ} {f : Ambient n → ℝ}
    (hf : AnalyticAt ℝ f 0) (horder : ExactOrderInLastVariable f d) :
    ∃ (a : Fin d → Base n → ℝ) (u : Ambient n → ℝ),
      IsWeierstrassPreparation f d a u := by
  obtain ⟨p, hp⟩ := hf
  obtain ⟨r, hr, hr0, htop, hq, hrem, hfac, hrem0, hq0⟩ :=
    exists_normalizedPreparationSequences p hp horder
  let a : Fin d → Base n → ℝ := preparationCoefficient p r hr d
  let u : Ambient n → ℝ := preparationUnit p r hr d
  refine ⟨a, u, ?_, ?_, ?_, ?_, ?_⟩
  · intro i
    exact analyticAt_preparationCoefficient p r hr d hrem i
  · intro i
    exact preparationCoefficient_zero p r hr d hrem0 i
  · exact analyticAt_preparationUnit p r hr d hq hq0
  · exact preparationUnit_zero_ne p r hr d hr0 htop hq0
  · have hfac' : ∀ᶠ x : Ambient n in 𝓝 0,
        monomialSeq d = convolution (normalizedPreparationQuotient p r hr d x.1)
            (analyticNormalizedCoefficientMap p r hr d x.1) +
              normalizedPreparationRemainder p r hr d x.1 ∧
        seqHighShift d (normalizedPreparationRemainder p r hr d x.1) = 0 :=
      (continuousAt_fst : ContinuousAt (fun x : Ambient n ↦ x.1) 0).eventually hfac
    have hrecon :=
      eventually_eval_weightedCoefficientSeries_eq_of_hasFPowerSeriesAt
        p r hp hr0 hr
    have hscaled : ∀ᶠ x : Ambient n in 𝓝 0, ‖(r : ℝ)⁻¹ * x.2‖ < 1 := by
      have hc : ContinuousAt (fun x : Ambient n ↦ (r : ℝ)⁻¹ * x.2) 0 := by
        fun_prop
      have hb := Metric.ball_mem_nhds (0 : ℝ) (by norm_num : (0 : ℝ) < 1)
      have hb' : Metric.ball (0 : ℝ) 1 ∈
          nhds ((fun x : Ambient n ↦ (r : ℝ)⁻¹ * x.2) 0) := by
        simpa [ambient_zero_eq] using hb
      have he := hc.eventually hb'
      simpa only [Metric.mem_ball, dist_zero_right] using he
    have hQne : ∀ᶠ x : Ambient n in 𝓝 0,
        preparationQuotientEval p r hr d x ≠ 0 := by
      have hQa := analyticAt_preparationQuotientEval p r hr d hq
      apply hQa.continuousAt.eventually_ne
      rw [preparationQuotientEval_zero, hq0]
      exact one_ne_zero
    filter_upwards [hfac', hrecon, hscaled, hQne] with x hx hrec hxw hxQ
    simpa only [a, u] using
      preparation_factorization_of_sequence_factorization p r hr hr0 htop
        x.1 x.2 hx.1 hx.2 hxw hrec hxQ

end Transformer.AnalyticPreparation
