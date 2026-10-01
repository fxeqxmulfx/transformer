/-
# Joint witnesses for real analytic Weierstrass division

The prepared divisor is `y² - z₀`. An exponential dividend exercises the
convergent-series construction, and a cubic dividend has the nonzero
remainder `z₀ y`. Auxiliary examples for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.AnalyticPreparation.AnalyticDivisionUniqueness
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Filter
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
namespace Transformer.AnalyticPreparation

/-- A genuinely analytic, nonpolynomial dividend and a nonconstant
distinguished divisor jointly satisfy the sequence and function division
hypotheses. Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 2 → Base 1 → ℝ := fun i z => if i = 0 then -z 0 else 0
    ∃ (p : FormalMultilinearSeries ℝ (Ambient 1) ℝ) (r : ℝ≥0),
      HasFPowerSeriesAt (fun x : Ambient 1 => Real.exp x.2) p 0 ∧
      0 < r ∧ (r : ℝ≥0∞) < p.radius ∧
      (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      preparedTailSeq r 2 a 0 = 0 ∧
      seqHighShift 2 (preparedTailSeq r 2 a 0) = 0 ∧
      ‖(divisionInput p r a 0).1‖ < 1 ∧
      AnalyticAt ℝ (divisionInput p r a) 0 ∧
      AnalyticAt ℝ (quotientSeq p r a) 0 ∧
      AnalyticAt ℝ (remainderSeq p r a) 0 ∧
      AnalyticAt ℝ (analyticDivisionQuotient p r a) 0 ∧
      (∀ i, AnalyticAt ℝ (analyticDivisionRemainderCoefficient p r a i) 0) ∧
      (∀ᶠ x in 𝓝 (0 : Ambient 1), Real.exp x.2 =
        analyticDivisionQuotient p r a x * preparedPolynomial 2 a x +
          ∑ i : Fin 2, analyticDivisionRemainderCoefficient p r a i x.1 * x.2 ^ (i : ℕ)) := by
  intro a
  have ha : ∀ i, AnalyticAt ℝ (a i) 0 := by
    intro i
    dsimp only [a]
    split_ifs
    · exact ((ContinuousLinearMap.proj (0 : Fin 1) : Base 1 →L[ℝ] ℝ).analyticAt 0).neg
    · exact analyticAt_const
  have ha0 : ∀ i, a i 0 = 0 := by intro i; simp only [a]; split_ifs <;> simp
  have hh : AnalyticAt ℝ (fun x : Ambient 1 => Real.exp x.2) 0 :=
    analyticAt_rexp.comp analyticAt_snd
  obtain ⟨p, hp⟩ := hh
  obtain ⟨r, hr0E, hrp⟩ := ENNReal.lt_iff_exists_nnreal_btwn.mp hp.radius_pos
  have hr0 : 0 < r := by exact_mod_cast hr0E
  have hdiv := analyticWeierstrassDivision_fixedRadius p hp r hr0 hrp a ha ha0
  exact ⟨p, r, hp, hr0, hrp, ha, ha0, preparedTailSeq_zero r a ha0,
    seqHighShift_preparedTailSeq r a 0, norm_divisionInput_fst_zero_lt_one r p a ha0,
    analyticAt_divisionInput p r hrp a ha, analyticAt_quotientSeq p r hrp a ha ha0,
    analyticAt_remainderSeq p r hrp a ha ha0, hdiv.1, hdiv.2.1, hdiv.2.2⟩

/-- Division of `y³` by `y² - z₀` computes the quotient `y` and the
nonzero remainder `z₀ y`, as analytic germs. This tests existence and
uniqueness together, including the coefficient identities. Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 2 → Base 1 → ℝ := fun i z => if i = 0 then -z 0 else 0
    let b : Fin 2 → Base 1 → ℝ := fun i z => if i = 0 then 0 else z 0
    ∃ (q : Ambient 1 → ℝ) (remainder : Fin 2 → Base 1 → ℝ),
      AnalyticAt ℝ q 0 ∧ (∀ i, AnalyticAt ℝ (remainder i) 0) ∧
      (∀ᶠ x in 𝓝 (0 : Ambient 1), x.2 ^ 3 =
        q x * preparedPolynomial 2 a x +
          ∑ i : Fin 2, remainder i x.1 * x.2 ^ (i : ℕ)) ∧
      q =ᶠ[𝓝 0] (fun x => x.2) ∧ ∀ i, remainder i =ᶠ[𝓝 0] b i := by
  intro a b
  have ha : ∀ i, AnalyticAt ℝ (a i) 0 := by
    intro i
    dsimp only [a]
    split_ifs
    · exact ((ContinuousLinearMap.proj (0 : Fin 1) : Base 1 →L[ℝ] ℝ).analyticAt 0).neg
    · exact analyticAt_const
  have ha0 : ∀ i, a i 0 = 0 := by intro i; simp only [a]; split_ifs <;> simp
  obtain ⟨q, remainder, hq, hr, hfactor⟩ := exists_analyticWeierstrassDivision
    (fun x : Ambient 1 => x.2 ^ 3) (analyticAt_snd.fun_pow 3) a ha ha0
  have hcomputed : ∀ᶠ x in 𝓝 (0 : Ambient 1), x.2 ^ 3 =
      x.2 * preparedPolynomial 2 a x + ∑ i : Fin 2, b i x.1 * x.2 ^ (i : ℕ) := by
    apply Eventually.of_forall
    intro x
    simp [preparedPolynomial, a, b, Fin.sum_univ_succ]
    ring
  have hunique := analyticWeierstrassDivision_unique
    (fun x : Ambient 1 => x.2 ^ 3) q (fun x => x.2) remainder b a hq
    analyticAt_snd ha ha0 hfactor hcomputed
  exact ⟨q, remainder, hq, hr, hfactor, hunique.1, hunique.2⟩

/-- Monomial coefficients satisfy the nonzero-radius and evaluation
uniqueness inputs, and give two actual sequence division factorizations.
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let f := monomialSeq 2
    let q := monomialSeq 1
    ‖(0 : L1Sequence)‖ < 1 ∧
      f = seqLowShift 1 q + convolution q 0 + 0 ∧
      seqHighShift 1 (0 : L1Sequence) = 0 ∧
      (q = seqDivisionQuotient 1 0 (by simp) f) ∧
      ((fun w => evalL1PowerSeries q w) =ᶠ[𝓝 0]
        (fun w => evalL1PowerSeries (monomialSeq 1) w)) ∧
      q = monomialSeq 1 := by
  intro f q
  have hfac : f = seqLowShift 1 q + convolution q 0 + 0 := by
    apply lp.ext
    funext k
    change f k = seqLowShift 1 q k + convolution q (0 : L1Sequence) k + 0
    by_cases hk : k < 1
    · have hk0 : k = 0 := by omega
      subst k
      simp [f, q, seqLowShift_apply_of_lt, convolution_apply]
    · rw [seqLowShift_apply_of_le 1 q (Nat.le_of_not_gt hk)]
      by_cases hk2 : k = 2
      · subst k
        simp [f, q, convolution_apply]
      · have hk1 : k - 1 ≠ 1 := by omega
        simp [f, q, convolution_apply, monomialSeq_apply_ne hk2,
          monomialSeq_apply_ne hk1]
  have hs : seqHighShift 1 (0 : L1Sequence) = 0 := by
    apply lp.ext
    funext k
    rfl
  have heval : (fun w => evalL1PowerSeries q w) =ᶠ[𝓝 0]
      (fun w => evalL1PowerSeries (monomialSeq 1) w) := EventuallyEq.rfl
  exact ⟨by simp, hfac, hs, seqDivisionQuotient_unique 1 0 (by simp) f q 0 hfac hs,
    heval, eq_of_evalL1PowerSeries_eventuallyEq _ _ heval⟩

end Transformer.AnalyticPreparation
