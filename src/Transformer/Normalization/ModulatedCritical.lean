/-
# The divergent clock and criticality of a limiting trajectory

Appendix D.1 of arXiv:2510.22026v2: divergence of the primitive forces
local integrability, and a bounded-energy limit must be critical.
-/

import Transformer.Normalization.ModulatedEnergy
import Mathlib.MeasureTheory.Integral.IntervalIntegral.DerivIntegrable
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Analysis.Calculus.ContDiff.Comp

open MeasureTheory

namespace Transformer.Normalization

/-- Divergence of the primitive in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2 forces local integrability on the positive half-line.
A nonintegrable interval would make every sufficiently long primitive zero. -/
theorem divergent_clock_intervalIntegrable (lam : ℝ → ℝ)
    (hint : Filter.Tendsto (fun T : ℝ => ∫ s in (0 : ℝ)..T, lam s)
      Filter.atTop Filter.atTop) (a b : ℝ) (ha : 0 ≤ a) (hab : a ≤ b) :
    IntervalIntegrable lam volume a b := by
  obtain ⟨T, hT⟩ := Filter.eventually_atTop.1
    ((Filter.eventually_ge_atTop b).and (hint.eventually (Filter.eventually_gt_atTop 0)))
  obtain ⟨hTb, hT⟩ := hT T le_rfl
  have hT0 : 0 ≤ T := ha.trans (hab.trans hTb)
  have hfull : IntervalIntegrable lam volume 0 T := by
    by_contra hn
    rw [intervalIntegral.integral_undef hn] at hT
    exact lt_irrefl 0 hT
  apply hfull.mono_set
  rw [Set.uIcc_of_le hab, Set.uIcc_of_le hT0]
  exact Set.Icc_subset_Icc ha hTb

/-- The clock hypotheses are satisfiable with `λ = 1`, whose primitive
is the identity; Appendix D.1 of arXiv:2510.22026v2. -/
example : Filter.Tendsto (fun T : ℝ => ∫ _s in (0 : ℝ)..T, (1 : ℝ))
    Filter.atTop Filter.atTop ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by
  refine ⟨?_, le_rfl, zero_le_one⟩
  convert (Filter.tendsto_id : Filter.Tendsto (fun T : ℝ => T)
    Filter.atTop Filter.atTop) using 1
  ext T
  simp

/-- Removing a finite initial interval preserves the divergent clock
in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem divergent_clock_tail (lam : ℝ → ℝ)
    (hint : Filter.Tendsto (fun T : ℝ => ∫ s in (0 : ℝ)..T, lam s)
      Filter.atTop Filter.atTop) (a : ℝ) (ha : 0 ≤ a) :
    Filter.Tendsto (fun T : ℝ => ∫ s in a..T, lam s) Filter.atTop Filter.atTop := by
  have hia := divergent_clock_intervalIntegrable lam hint 0 a le_rfl ha
  have h := Filter.tendsto_atTop_add_const_right Filter.atTop
    (-(∫ s in (0 : ℝ)..a, lam s)) hint
  apply h.congr'
  filter_upwards [Filter.eventually_ge_atTop a] with T hT
  have heq := intervalIntegral.integral_add_adjacent_intervals hia
    (divergent_clock_intervalIntegrable lam hint a T ha hT)
  linarith

/-- A positive constant clock also witnesses the tail-divergence
hypothesis; Appendix D.1 of arXiv:2510.22026v2. -/
example : Filter.Tendsto (fun T : ℝ => ∫ _s in (0 : ℝ)..T, (1 : ℝ))
    Filter.atTop Filter.atTop ∧ (0 : ℝ) ≤ 0 := by
  refine ⟨?_, le_rfl⟩
  convert (Filter.tendsto_id : Filter.Tendsto (fun T : ℝ => T)
    Filter.atTop Filter.atTop) using 1
  ext T
  simp

/-- Under the dissipation inequality of Appendix D.1, Step 2, of
arXiv:2510.22026v2, a positive multiple of the elapsed clock is bounded
by the energy drop. Continuity of `λ` and of the derivative is unnecessary. -/
theorem clock_dissipation_bound (lam f f' : ℝ → ℝ) (a b delta : ℝ)
    (hab : a ≤ b) (hdelta : 0 ≤ delta)
    (hlam : ∀ t ∈ Set.Icc a b, 0 ≤ lam t)
    (hint : IntervalIntegrable lam volume a b)
    (hf : ∀ t ∈ Set.Icc a b, HasDerivAt f (f' t) t)
    (hbound : ∀ t ∈ Set.Icc a b, f' t ≤ -delta * lam t) :
    delta * (∫ t in a..b, lam t) ≤ f a - f b := by
  have hd t (ht : t ∈ Set.Icc a b) :
      HasDerivAt (fun s => -f s) (-f' t) t := by
    convert (hf t ht).neg using 1
  have hmono : MonotoneOn (fun t => -f t) (Set.uIcc a b) := by
    rw [Set.uIcc_of_le hab]
    apply monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc a b)
    · intro t ht
      exact (hd t ht).continuousAt.continuousWithinAt
    · intro t ht
      exact (hd t (interior_subset ht)).hasDerivWithinAt
    · intro t ht
      have hb := hbound t (interior_subset ht)
      have hn := mul_nonneg hdelta (hlam t (interior_subset ht))
      linarith
  have hi := hmono.intervalIntegrable_deriv
  have heq : (∫ t in a..b, deriv (fun s => -f s) t) = f a - f b := by
    have h := intervalIntegral.integral_eq_sub_of_hasDerivAt
      (fun t ht => (hd t (by simpa only [Set.uIcc_of_le hab] using ht)).differentiableAt.hasDerivAt) hi
    convert h using 1
    simp only [neg_sub_neg]
  have hle := intervalIntegral.integral_mono_on hab (hint.const_mul delta) hi
    (fun t ht => by
      rw [(hd t ht).deriv]
      have hb := hbound t ht
      linarith)
  rwa [intervalIntegral.integral_const_mul, heq] at hle

/-- The clock-dissipation hypotheses hold for `λ = 1`, `f(t) = -t`
and `δ = 1` on `[0,1]`; Appendix D.1 of arXiv:2510.22026v2. -/
example : (0 : ℝ) ≤ 1 ∧
    IntervalIntegrable (fun _ : ℝ => (1 : ℝ)) volume 0 1 ∧
    (∀ t ∈ Set.Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => -s) (-1) t) ∧
    (∀ t ∈ Set.Icc (0 : ℝ) 1, (-1 : ℝ) ≤ -1 * 1) := by
  refine ⟨zero_le_one, intervalIntegrable_const, fun t _ => ?_, fun _ _ => by norm_num⟩
  exact hasDerivAt_neg' t

/-- An analytic energy has a continuous gradient near each point.
This supplies the gradient continuity used in Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem analytic_gradient_continuousAt {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) (hE : AnalyticAt ℝ E z) : ContinuousAt (gradient E) z := by
  have hf : ContinuousAt (fderiv ℝ E) z :=
    (hE.contDiffAt : ContDiffAt ℝ 1 E z).continuousAt_fderiv (by norm_num)
  exact (InnerProductSpace.toDual ℝ (EucSpace N)).symm.continuous.continuousAt.comp hf

/-- A constant analytic energy witnesses the gradient-continuity
hypothesis; Appendix D.1 of arXiv:2510.22026v2. -/
example : AnalyticAt ℝ (fun _ : EucSpace 1 => (0 : ℝ)) 0 := analyticAt_const

/-- Any limit of a bounded-below modulated gradient flow is critical,
under the lower matrix bound and divergent clock of Appendix D.1,
`lem: loj`, in arXiv:2510.22026v2. The proof uses the integral energy
estimate and continuity of the gradient at the limiting point. -/
theorem modulated_limit_critical {N : ℕ} (E : EucSpace N → ℝ)
    (x : ℝ → EucSpace N) (M : ℝ → ParamMatrix N) (lam : ℝ → ℝ)
    (L : ℝ) (z : EucSpace N)
    (hE : ∀ t, 0 ≤ t → DifferentiableAt ℝ E (x t))
    (hx : ∀ t, 0 ≤ t → HasDerivAt x (-(M t (gradient E (x t)))) t)
    (hlam : ∀ t, 0 ≤ t → 0 < lam t)
    (hM : ∀ t, 0 ≤ t → ∀ v : EucSpace N,
      lam t * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) (M t v) v)
    (hint : Filter.Tendsto (fun T : ℝ => ∫ s in (0 : ℝ)..T, lam s)
      Filter.atTop Filter.atTop)
    (hL : ∀ t, 0 ≤ t → L ≤ E (x t))
    (hgrad : ContinuousAt (gradient E) z)
    (hlim : Filter.Tendsto x Filter.atTop (nhds z)) : gradient E z = 0 := by
  by_contra hn
  let delta : ℝ := ‖gradient E z‖ / 2
  have hd : 0 < delta := half_pos (norm_pos_iff.mpr hn)
  have hg : delta < ‖gradient E z‖ := by dsimp [delta]; linarith [norm_pos_iff.mpr hn]
  have ht := (hgrad.norm.tendsto.comp hlim).eventually (lt_mem_nhds hg)
  obtain ⟨a, ha⟩ := Filter.eventually_atTop.1
    ((Filter.eventually_ge_atTop (0 : ℝ)).and ht)
  have ha0 : 0 ≤ a := (ha a le_rfl).1
  have hclock := divergent_clock_tail lam hint a ha0
  obtain ⟨b, hb⟩ := Filter.eventually_atTop.1
    ((Filter.eventually_ge_atTop a).and
      (hclock.eventually (Filter.eventually_gt_atTop ((E (x a) - L) / delta ^ 2))))
  obtain ⟨hab, hbclock⟩ := hb b le_rfl
  have hbd : delta * (∫ t in a..b, lam t) * delta ≤ E (x a) - E (x b) := by
    have h := clock_dissipation_bound lam (fun t => E (x t))
      (fun t => -inner (𝕜 := ℝ) (M t (gradient E (x t))) (gradient E (x t)))
      a b (delta ^ 2) hab (sq_nonneg delta)
      (fun t ht => (hlam t (ha0.trans ht.1)).le)
      (divergent_clock_intervalIntegrable lam hint a b ha0 hab)
      (fun t ht => modulated_energy_hasDerivAt E x M t
        (hE t (ha0.trans ht.1)) (hx t (ha0.trans ht.1)))
      (fun t ht => by
        have hMt := hM t (ha0.trans ht.1) (gradient E (x t))
        have hgt := (ha t ht.1).2
        change delta < ‖gradient E (x t)‖ at hgt
        have hlt := hlam t (ha0.trans ht.1)
        have hsq : delta ^ 2 ≤ ‖gradient E (x t)‖ ^ 2 := by nlinarith
        have hmul := mul_le_mul_of_nonneg_left hsq hlt.le
        nlinarith)
    nlinarith [h]
  have hlarge : E (x a) - L < delta ^ 2 * (∫ t in a..b, lam t) :=
    by nlinarith [(div_lt_iff₀ (sq_pos_of_pos hd)).mp hbclock]
  have hLb := hL b (ha0.trans hab)
  nlinarith

/-- Constant energy, identity modulation, constant positive clock, and
the zero trajectory satisfy all critical-limit hypotheses in
Appendix D.1 of arXiv:2510.22026v2. -/
example :
    (∀ t : ℝ, 0 ≤ t → DifferentiableAt ℝ (fun _ : EucSpace 1 => (0 : ℝ)) 0) ∧
    (∀ t : ℝ, 0 ≤ t → HasDerivAt (fun _ : ℝ => (0 : EucSpace 1)) 0 t) ∧
    (∀ v : EucSpace 1, (1 : ℝ) * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) v v) ∧
    (∀ t : ℝ, 0 ≤ t → (0 : ℝ) ≤ 0) ∧
    ContinuousAt (gradient (fun _ : EucSpace 1 => (0 : ℝ))) 0 ∧
    Filter.Tendsto (fun _ : ℝ => (0 : EucSpace 1)) Filter.atTop (nhds 0) := by
  exact ⟨fun _ _ => differentiableAt_const 0, fun t _ => hasDerivAt_const t 0,
    fun v => by simp, fun _ _ => le_rfl,
    analytic_gradient_continuousAt _ _ analyticAt_const, tendsto_const_nhds⟩

end Transformer.Normalization
