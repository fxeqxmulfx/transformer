/-
# Preparation along an analytic curve in the base parameters

An analytic base curve gives a two-variable analytic slice. Real
preparation identifies all nearby zeros in each scalar fiber with the
roots of an analytic monic polynomial family.
-/

import Transformer.Normalization.EnergyLevelPreparation
import Transformer.Normalization.PolynomialFamily

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- Restriction to an analytic base curve retains genuine joint
analyticity in the curve parameter and distinguished variable. Auxiliary
for the fiber minimum step of Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_parameter_slice_composition {n : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0)
    (gamma : ℝ → Base n) (hgamma : AnalyticAt ℝ gamma 0) (hgamma0 : gamma 0 = 0) :
    AnalyticAt ℝ (fun x : ℝ × ℝ => F (gamma x.1, x.2)) 0 := by
  let path : ℝ × ℝ → Ambient n := fun x => (gamma x.1, x.2)
  have hpath : AnalyticAt ℝ path 0 :=
    (hgamma.comp (f := fun x : ℝ × ℝ => x.1) (x := 0) analyticAt_fst).prod analyticAt_snd
  have hpath0 : path 0 = 0 := by simp [path, hgamma0, Prod.mk_zero_zero]
  exact (by simpa only [hpath0] using hF : AnalyticAt ℝ F (path 0)).comp
    (f := path) (x := 0) hpath

/-- Real preparation along an arbitrary analytic base curve identifies
the original analytic zero fibers with polynomial roots on one fixed
parameter-variable box. The lower coefficients are analytic and vanish
at the origin. Both directions of the zero equivalence hold for every
point of the box, which is essential for universal minimum comparisons.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_prepared_analytic_slice {n d : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0) (horder : ExactOrderInLastVariable F d)
    (gamma : ℝ → Base n) (hgamma : AnalyticAt ℝ gamma 0) (hgamma0 : gamma 0 = 0) :
    ∃ (a : Fin d → ℝ → ℝ) (r : ℝ), 0 < r ∧
      (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
        ∀ t y : ℝ, |t| < r → |y| < r →
          (F (gamma t, y) = 0 ↔ polynomialFamily d a (t, y) = 0) := by
  obtain ⟨A, U, hprep⟩ := exists_isWeierstrassPreparation hF horder
  let a : Fin d → ℝ → ℝ := fun i t => A i (gamma t)
  have ha (i : Fin d) : AnalyticAt ℝ (a i) 0 :=
    (by simpa only [hgamma0] using hprep.1 i : AnalyticAt ℝ (A i) (gamma 0)).comp
      (f := gamma) (x := 0) hgamma
  have ha0 (i : Fin d) : a i 0 = 0 := by simpa only [a, hgamma0] using hprep.2.1 i
  let path : ℝ × ℝ → Ambient n := fun x => (gamma x.1, x.2)
  have hpath : AnalyticAt ℝ path 0 :=
    (hgamma.comp (f := fun x : ℝ × ℝ => x.1) (x := 0) analyticAt_fst).prod analyticAt_snd
  have hpath0 : path 0 = 0 := by simp [path, hgamma0, Prod.mk_zero_zero]
  have ht : Tendsto path (nhds 0) (nhds 0) := by
    simpa only [hpath0] using hpath.continuousAt.tendsto
  have hzero : ∀ᶠ x in nhds (0 : ℝ × ℝ),
      F (gamma x.1, x.2) = 0 ↔ polynomialFamily d a x = 0 := by
    have hunit : ∀ᶠ x in nhds (0 : Ambient n), U x ≠ 0 :=
      hprep.2.2.1.continuousAt.eventually_ne hprep.2.2.2.1
    filter_upwards [ht.eventually hprep.2.2.2.2, ht.eventually hunit] with x hx hu
    change F (gamma x.1, x.2) = U (path x) * polynomialFamily d a x at hx
    rw [hx, mul_eq_zero, or_iff_right hu]
  obtain ⟨r, hr, hbox⟩ := Metric.eventually_nhds_iff.mp hzero
  refine ⟨a, r, hr, ha, ha0, fun t y ht hy => hbox (y := (t, y)) ?_⟩
  simpa only [dist_zero_right, Prod.norm_def, Real.norm_eq_abs] using max_lt ht hy

/-- The singular equation `y²=z₀` and the base curve `z₀=t` jointly
satisfy the slice-analyticity and preparation hypotheses. The conclusion
identifies all local roots, rather than a chosen root alone. Auxiliary
example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : AnalyticAt ℝ (fun x : ℝ × ℝ => x.2 ^ 2 - x.1) 0 ∧
    ∃ (a : Fin 2 → ℝ → ℝ) (r : ℝ), 0 < r ∧
    (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      ∀ t y : ℝ, |t| < r → |y| < r →
        (y ^ 2 - t = 0 ↔ polynomialFamily 2 a (t, y) = 0) := by
  let F : Ambient 1 → ℝ := fun x => x.2 ^ 2 - x.1 0
  let L : Ambient 1 →L[ℝ] ℝ :=
    (ContinuousLinearMap.proj (0 : Fin 1)).comp (ContinuousLinearMap.fst ℝ (Base 1) ℝ)
  have hF : AnalyticAt ℝ F 0 := (analyticAt_snd.fun_pow 2).fun_sub (L.analyticAt 0)
  have horder : ExactOrderInLastVariable F 2 := by
    have hslice : lastSlice F = fun t : ℝ => t ^ 2 := by funext t; simp [lastSlice, F]
    rw [ExactOrderInLastVariable, hslice]
    have hord : analyticOrderAt (fun t : ℝ => t ^ 2) 0 = 2 := by
      simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 2
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 2)).mp hord
  let gamma : ℝ → Base 1 := fun t _ => t
  have hgamma : AnalyticAt ℝ gamma 0 := AnalyticAt.pi (fun _ => analyticAt_id)
  have hs := analytic_parameter_slice_composition F hF gamma hgamma rfl
  obtain ⟨a, r, hr, ha, ha0, hzero⟩ := exists_prepared_analytic_slice F hF horder gamma hgamma rfl
  exact ⟨hs, a, r, hr, ha, ha0, hzero⟩

end Transformer.Normalization
