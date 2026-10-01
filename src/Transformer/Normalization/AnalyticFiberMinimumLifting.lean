/-
# Lifting minima over the distinguished-variable analytic fibers

Real preparation, complete root coverage, and stable analytic branch
comparisons construct actual constrained minima on the scalar fibers
over an arbitrary analytic base curve in any parameter dimension.
-/

import Transformer.Normalization.AnalyticPreparedSlice
import Transformer.Normalization.PolynomialFiberMinimum

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- Over an analytic base curve, a finite-order analytic equation admits
an analytic minimum of any analytic objective after ramification. The
comparison quantifies over every admissible distinguished-variable root
in a fixed neighborhood. Only feasible accumulating roots are assumed;
no minimizer curve is given. This removes one scalar fiber quantifier in
arbitrary parameter dimension. Selecting a base curve for a projected
set remains a separate step of Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_fiber_minimum_lifting {n d l : ℕ}
    (F : Ambient n → ℝ) (hF : AnalyticAt ℝ F 0) (horder : ExactOrderInLastVariable F d)
    (gamma : ℝ → Base n) (hgamma : AnalyticAt ℝ gamma 0) (hgamma0 : gamma 0 = 0)
    (V : Ambient n → ℝ) (hV : AnalyticAt ℝ V 0)
    (C : Fin l → Ambient n → ℝ) (hC : ∀ i, AnalyticAt ℝ (C i) 0)
    (requirement : Fin l → AnalyticSignRequirement)
    (hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ F (gamma x.1, x.2) = 0 ∧
      ∀ i, (requirement i).Holds (C i (gamma x.1, x.2))) :
    ∃ (q : ℕ) (g : ℝ → ℝ) (r : ℝ), 0 < q ∧ 0 < r ∧
      AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
      (∀ᶠ s in nhds (0 : ℝ), F (gamma (s ^ q), g s) = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), |g s| < r ∧
        (∀ i, (requirement i).Holds (C i (gamma (s ^ q), g s))) ∧
          ∀ y : ℝ, |y| < r → F (gamma (s ^ q), y) = 0 →
            (∀ i, (requirement i).Holds (C i (gamma (s ^ q), y))) →
              V (gamma (s ^ q), g s) ≤ V (gamma (s ^ q), y) := by
  obtain ⟨a, r, hr, ha, ha0, hzero⟩ :=
    exists_prepared_analytic_slice F hF horder gamma hgamma hgamma0
  let constraints : Fin l → ℝ × ℝ → ℝ := fun i x => C i (gamma x.1, x.2)
  have hc (i : Fin l) : AnalyticAt ℝ (constraints i) 0 :=
    analytic_parameter_slice_composition (C i) (hC i) gamma hgamma hgamma0
  have hv : AnalyticAt ℝ (fun x : ℝ × ℝ => V (gamma x.1, x.2)) 0 :=
    analytic_parameter_slice_composition V hV gamma hgamma hgamma0
  have hsmall : ∀ᶠ x in nhds (0 : ℝ × ℝ), |x.1| < r ∧ |x.2| < r := by
    filter_upwards [Metric.ball_mem_nhds (0 : ℝ × ℝ) hr] with x hx
    simpa only [Metric.mem_ball, dist_zero_right, Prod.norm_def, Real.norm_eq_abs,
      max_lt_iff] using hx
  have hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ,
      polynomialFamily d a (t, y) = 0 ∧ ∀ i,
        (requirement i).Holds (constraints i (t, y)) := by
    by_contra h
    have hno : ∀ᶠ t in nhds (0 : ℝ), 0 < t → ¬ ∃ y : ℝ,
        polynomialFamily d a (t, y) = 0 ∧ ∀ i,
          (requirement i).Holds (constraints i (t, y)) :=
      eventually_nhdsWithin_iff.mp (not_frequently.mp h)
    have hfst : Tendsto (fun x : ℝ × ℝ => x.1) (nhds 0) (nhds 0) :=
      continuous_fst.tendsto 0
    obtain ⟨x, hx⟩ := (hacc.and_eventually (hsmall.and (hfst.eventually hno))).exists
    exact hx.2.2 hx.1.1 ⟨x.2,
      (hzero x.1 x.2 hx.2.1.1 hx.2.1.2).mp hx.1.2.1, hx.1.2.2⟩
  obtain ⟨q, g, hq, hg, hg0, hroot, hminimum⟩ :=
    analytic_polynomial_fiber_minimum a ha ha0 (fun x => V (gamma x.1, x.2)) hv
      constraints hc requirement hroots
  have hnear : ∀ᶠ s in nhds (0 : ℝ), |s ^ q| < r ∧ |g s| < r := by
    have hpower : ContinuousAt (fun s : ℝ => |s ^ q|) 0 := (continuousAt_id.pow q).abs
    have hpow0 : |(0 : ℝ) ^ q| < r := by simpa [hq.ne'] using hr
    have hgabs0 : |g 0| < r := by simpa only [hg0, abs_zero] using hr
    exact (hpower.eventually (Iio_mem_nhds hpow0)).and
      (hg.continuousAt.abs.eventually (Iio_mem_nhds hgabs0))
  refine ⟨q, g, r, hq, hr, hg, hg0, ?_, ?_⟩
  · filter_upwards [hroot, hnear] with s hs hn
    exact (hzero (s ^ q) (g s) hn.1 hn.2).mpr hs
  · filter_upwards [hminimum, hnear.filter_mono nhdsWithin_le_nhds] with s hs hn
    refine ⟨hn.2, hs.1, ?_⟩
    intro y hy hFy hCy
    exact hs.2 y ((hzero (s ^ q) y hn.1 hy).mp hFy) hCy

/-- The equation `(y²-z₀)²=0` has repeated real roots over `z₀=t`.
Minimizing `y` with a nonnegative base constraint exercises every
preparation, analyticity, accumulation and minimum-lifting hypothesis.
Auxiliary example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ (q : ℕ) (g : ℝ → ℝ) (r : ℝ), 0 < q ∧ 0 < r ∧
    AnalyticAt ℝ g 0 ∧ g 0 = 0 ∧
    (∀ᶠ s in nhds (0 : ℝ), ((g s) ^ 2 - s ^ q) ^ 2 = 0) ∧
      ∀ᶠ s in nhdsWithin (0 : ℝ) (Ioi 0), |g s| < r ∧ (0 ≤ s ^ q) ∧
        ∀ y : ℝ, |y| < r → (y ^ 2 - s ^ q) ^ 2 = 0 → 0 ≤ s ^ q → g s ≤ y := by
  let F : Ambient 1 → ℝ := fun x => (x.2 ^ 2 - x.1 0) ^ 2
  let L : Ambient 1 →L[ℝ] ℝ :=
    (ContinuousLinearMap.proj (0 : Fin 1)).comp (ContinuousLinearMap.fst ℝ (Base 1) ℝ)
  have hF : AnalyticAt ℝ F 0 := ((analyticAt_snd.fun_pow 2).fun_sub (L.analyticAt 0)).fun_pow 2
  have horder : ExactOrderInLastVariable F 4 := by
    have hslice : lastSlice F = fun t : ℝ => t ^ 4 := by
      funext t
      simp only [lastSlice, F, Pi.zero_apply, sub_zero]
      ring
    rw [ExactOrderInLastVariable, hslice]
    have hord : analyticOrderAt (fun t : ℝ => t ^ 4) 0 = 4 := by
      simpa [Pi.pow_def] using analyticOrderAt_pow (analyticAt_id (𝕜 := ℝ) (z := (0 : ℝ))) 4
    exact (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero (analyticAt_id.fun_pow 4)).mp hord
  let gamma : ℝ → Base 1 := fun t _ => t
  have hgamma : AnalyticAt ℝ gamma 0 := AnalyticAt.pi (fun _ => analyticAt_id)
  have hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ F (gamma x.1, x.2) = 0 ∧
      ∀ i : Fin 1, AnalyticSignRequirement.nonnegative.Holds x.1 := by
    have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    let path : ℝ → ℝ × ℝ := fun t => (t ^ 2, t)
    have ht : Tendsto path (nhdsWithin 0 (Ioi 0)) (nhds (0 : ℝ × ℝ)) := by
      have hpath : ContinuousAt path 0 := by fun_prop
      simpa [path, Prod.mk_zero_zero] using hpath.tendsto.mono_left nhdsWithin_le_nhds
    have hp : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply ht.frequently
    apply hp.frequently.mono
    intro t ht
    exact ⟨sq_pos_of_pos ht, by simp [F, gamma, path], fun _ => sq_nonneg t⟩
  simpa only [F, gamma, AnalyticSignRequirement.Holds, forall_const] using
    analytic_fiber_minimum_lifting F hF horder gamma hgamma rfl
      (fun x => x.2) analyticAt_snd (fun (_ : Fin 1) x => x.1 0)
      (fun _ => L.analyticAt 0) (fun _ => .nonnegative) hacc

end Transformer.Normalization
