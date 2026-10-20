module

public import Transformer.Causal.Section4_ScalarLimits

/-!
# Bounded scalar components with an expanding coefficient

ArXiv:2411.04990v2, §4, a reduction for the exceptional case of `thm2`.
After dividing a transverse leading component by a negative longitudinal
component, the common radial coefficient cancels. The quotient has a
uniformly positive linear coefficient and a forcing tending to zero.
Every bounded such solution tends to zero. Derivatives are needed only
on the forward tail, since the denominator may vanish earlier.
-/

@[expose] public section

open Real Set Filter
open scoped Topology

namespace Transformer.Causal

/-- A bounded scalar cannot enter a region with uniformly positive
forward drift. The derivative is required only after `T`.
Source: arXiv:2411.04990v2, §4, the barrier and escape argument for
`thm1`, applied to the expanding transverse equation of `thm2`. -/
theorem csaScalar_lt_of_tail_growth (r r' : ℝ → ℝ) (M a c T : ℝ)
    (hd : ∀ t, T ≤ t → HasDerivAt r (r' t) t)
    (hr : ∀ t, T ≤ t → r t ≤ M) (hc : 0 < c)
    (hg : ∀ t, T ≤ t → a ≤ r t → c ≤ r' t) :
    ∀ s, T ≤ s → r s < a := by
  intro s hTs
  by_contra hs
  have hs : a ≤ r s := le_of_not_gt hs
  have hbar (t : ℝ) (hst : s ≤ t) : a ≤ r t := by
    have h := image_le_of_deriv_right_lt_deriv_boundary
      (f := fun u => -r u) (f' := fun u => -r' u)
      (B := fun _ => -a) (B' := fun _ => 0)
      (fun x hx => (hd x (hTs.trans hx.1)).neg.continuousAt.continuousWithinAt)
      (fun x hx => (hd x (hTs.trans hx.1)).neg.hasDerivWithinAt)
      (neg_le_neg hs) (fun x => hasDerivAt_const x (-a))
      (fun x hx he => by
        have hp := hc.trans_le (hg x (hTs.trans hx.1) (by linarith))
        linarith) (show t ∈ Icc s t from ⟨hst, le_rfl⟩)
    exact neg_le_neg_iff.mp h
  let u := s + (max (M - r s) 0 + 1) / c
  have hsu : s ≤ u := le_add_of_nonneg_right (div_nonneg (by positivity) hc.le)
  have hlin : c * (u - s) ≤ r u - r s := by
    apply (convex_Ici s).mul_sub_le_image_sub_of_le_deriv
      (fun t ht => (hd t (hTs.trans ht)).continuousAt.continuousWithinAt)
      (fun t ht => (hd t (hTs.trans (interior_subset ht))).differentiableAt.differentiableWithinAt)
      ?_ s (by simp) u hsu hsu
    intro t ht
    rw [(hd t (hTs.trans (interior_subset ht))).deriv]
    exact hg t (hTs.trans (interior_subset ht)) (hbar t (interior_subset ht))
  have he : c * (u - s) = max (M - r s) 0 + 1 := by dsimp [u]; field_simp; ring
  rw [he] at hlin
  linarith [hr u (hTs.trans hsu), le_max_left (M - r s) 0]

/-- A bounded solution of an expanding scalar equation with vanishing
additive forcing tends to zero. No sign condition on the forcing or
the solution is imposed. Source: arXiv:2411.04990v2, §4, the transverse
quotient reduction of `thm2`; this is an auxiliary scalar consequence. -/
theorem tendsto_zero_of_bounded_expanding (a b z : ℝ → ℝ) (M c T : ℝ)
    (hd : ∀ t, T ≤ t → HasDerivAt z (a t * z t + b t) t)
    (hz : ∀ t, T ≤ t → |z t| ≤ M) (hc : 0 < c)
    (ha : ∀ t, T ≤ t → c ≤ a t) (hb : Tendsto b atTop (nhds 0)) :
    Tendsto z atTop (nhds 0) := by
  apply Metric.tendsto_nhds.mpr
  intro ε hε
  have he : 0 < c * ε / 2 := div_pos (mul_pos hc hε) (by norm_num)
  obtain ⟨S, hS⟩ := eventually_atTop.mp
    ((Metric.tendsto_nhds.mp hb) (c * ε / 2) he)
  have hb' (t : ℝ) (ht : max T S ≤ t) : |b t| < c * ε / 2 := by
    simpa only [Real.dist_eq, sub_zero] using hS t ((le_max_right _ _).trans ht)
  have hu : ∀ s, max T S ≤ s → z s < ε := by
    apply csaScalar_lt_of_tail_growth z (fun t => a t * z t + b t) M ε (c * ε / 2) (max T S)
      (fun t ht => hd t ((le_max_left _ _).trans ht))
      (fun t ht => (le_abs_self _).trans (hz t ((le_max_left _ _).trans ht))) he
    intro t ht hεz
    have hmul := mul_le_mul (ha t ((le_max_left _ _).trans ht)) hεz hε.le
      (hc.le.trans (ha t ((le_max_left _ _).trans ht)))
    linarith [(abs_lt.mp (hb' t ht)).1]
  have hl : ∀ s, max T S ≤ s → -z s < ε := by
    apply csaScalar_lt_of_tail_growth (fun t => -z t) (fun t => -(a t * z t + b t))
      M ε (c * ε / 2) (max T S)
      (fun t ht => (hd t ((le_max_left _ _).trans ht)).neg)
      (fun t ht => (neg_le_abs _).trans (hz t ((le_max_left _ _).trans ht))) he
    intro t ht hεz
    have hmul := mul_le_mul (ha t ((le_max_left _ _).trans ht)) hεz hε.le
      (hc.le.trans (ha t ((le_max_left _ _).trans ht)))
    nlinarith [(abs_lt.mp (hb' t ht)).2]
  filter_upwards [eventually_ge_atTop (max T S)] with t ht
  rw [Real.dist_eq, sub_zero, abs_lt]
  exact ⟨by linarith [hl t ht], hu t ht⟩

/-- A uniformly positive longitudinal forcing and a bounded radial
coefficient either force eventual nonnegative overlap or keep the
overlap uniformly negative. Source: arXiv:2411.04990v2, §4, the
exceptional-trajectory reduction for `thm2`. -/
theorem eventually_nonneg_or_uniformly_negative (h f q : ℝ → ℝ)
    (hq : ∀ t, HasDerivAt q (f t + h t * q t) t)
    (hbound : ∀ t, -1 ≤ q t ∧ q t ≤ 1) (M c T : ℝ)
    (hM : 0 < M) (hc : 0 < c)
    (hh : ∀ t, T ≤ t → |h t| ≤ M) (hf : ∀ t, T ≤ t → c ≤ f t) :
    ∃ δ : ℝ, 0 < δ ∧
      ((∀ᶠ t in atTop, 0 ≤ q t) ∨ (∀ᶠ t in atTop, q t < -δ)) := by
  let δ := c / (2 * M)
  have hδ : 0 < δ := div_pos hc (mul_pos (by norm_num) hM)
  have he : M * δ = c / 2 := by dsimp [δ]; field_simp
  refine ⟨δ, hδ, ?_⟩
  by_cases hex : ∃ s, T ≤ s ∧ -δ ≤ q s
  · left
    obtain ⟨s, hTs, hs⟩ := hex
    apply csaScalar_escape_strip q (fun t => f t + h t * q t) hq hbound
      (-δ) 0 T (c / 2) s (by linarith) (by positivity) hTs hs
    intro t ht hlo hhi
    have hmul := mul_le_mul_of_nonpos_right ((le_abs_self (h t)).trans (hh t ht)) hhi
    have hlow := mul_le_mul_of_nonneg_left hlo hM.le
    nlinarith [hf t ht]
  · right
    push Not at hex
    exact (eventually_ge_atTop T).mono fun t ht => hex t ht

/-- A zero transverse solution with a positive coefficient jointly
realizes the bounded expanding scalar hypotheses. The forcing
need not share the sign of the solution in the general theorem.
Source: arXiv:2411.04990v2, §4, the exceptional branch of `thm2`. -/
example : ∃ a b z : ℝ → ℝ,
    (∀ t, (0 : ℝ) ≤ t → HasDerivAt z (a t * z t + b t) t) ∧
    (∀ t, (0 : ℝ) ≤ t → |z t| ≤ 1) ∧ (0 : ℝ) < 1 ∧
    (∀ t, (0 : ℝ) ≤ t → 1 ≤ a t) ∧ Tendsto b atTop (nhds 0) := by
  exact ⟨fun _ => 1, fun _ => 0, fun _ => 0,
    fun t _ => by simpa using hasDerivAt_const t (0 : ℝ),
    fun _ _ => by norm_num, one_pos, fun _ _ => le_rfl, tendsto_const_nhds⟩

/-- A bounded constant below the forbidden level realizes every tail
growth premise. The growth condition is imposed only above that level,
as the comparison theorem requires. Source: arXiv:2411.04990v2, §4,
the barrier argument underlying `thm1` and the `thm2` reduction. -/
example : ∃ r r' : ℝ → ℝ,
    (∀ t, (0 : ℝ) ≤ t → HasDerivAt r (r' t) t) ∧
    (∀ t, (0 : ℝ) ≤ t → r t ≤ 1) ∧ (0 : ℝ) < 1 ∧
    (∀ t, (0 : ℝ) ≤ t → (1 : ℝ) ≤ r t → 1 ≤ r' t) := by
  exact ⟨fun _ => 0, fun _ => 0, fun t _ => hasDerivAt_const t (0 : ℝ),
    fun _ _ => zero_le_one, one_pos, fun _ _ h => by linarith⟩

/-- The scalar negative equilibrium realizes the longitudinal equation,
the uniform forcing, and the coefficient and overlap bounds together.
Source: arXiv:2411.04990v2, §4, the exceptional branch of `thm2`. -/
example : ∃ h f q : ℝ → ℝ,
    (∀ t, HasDerivAt q (f t + h t * q t) t) ∧
    (∀ t, -1 ≤ q t ∧ q t ≤ 1) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (∀ t, (0 : ℝ) ≤ t → |h t| ≤ 1) ∧
    (∀ t, (0 : ℝ) ≤ t → 1 ≤ f t) := by
  exact ⟨fun _ => 1, fun _ => 1, fun _ => -1,
    fun t => by simpa using hasDerivAt_const t (-1 : ℝ),
    fun _ => by norm_num, one_pos, one_pos,
    fun _ _ => by norm_num, fun _ _ => le_rfl⟩

end Transformer.Causal
