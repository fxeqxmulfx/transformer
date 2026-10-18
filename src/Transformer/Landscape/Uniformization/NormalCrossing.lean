/-
# Local uniformization: functions with normal crossings

A unit times a coordinate monomial centred anywhere has normal crossings at
every point where the unit does not vanish; hence normal crossings is an open
condition, is stable under products, and never holds for a germ vanishing
identically. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Landscape.Uniformization.Defs

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base)

variable {n : ℕ}

/-- A monomial at a constant vector is a power. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem monomial_const (p : Fin n → ℕ) (s : ℝ) :
    monomial p (fun _ => s) = s ^ ∑ i, p i := by
  simp only [monomial, Finset.prod_pow_eq_pow_sum]

/-- A centred monomial is analytic. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem analyticAt_monomial_sub (p : Fin n → ℕ) (a w : Base n) :
    AnalyticAt ℝ (fun x => monomial p (x - a)) w :=
  (analyticAt_monomial p (w - a)).comp (f := fun x : Base n => x - a) (x := w)
    (analyticAt_id.fun_sub analyticAt_const)

theorem NormalCrossingAt.congr {f g : Base n → ℝ} {w : Base n} (hf : NormalCrossingAt f w)
    (hfg : f =ᶠ[𝓝 w] g) : NormalCrossingAt g w := by
  obtain ⟨p, u, hu, huw, h⟩ := hf
  refine ⟨p, u, hu, huw, ?_⟩
  filter_upwards [hfg, h] with x h1 h2
  rw [← h1, h2]

/-- A unit times a monomial centred at any point `a` has normal crossings at
`w`: the coordinates with `w j ≠ a j` move into the unit. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem normalCrossingAt_of_eq_mul_monomial {f u : Base n → ℝ} {w : Base n} (a : Base n)
    (p : Fin n → ℕ) (hu : AnalyticAt ℝ u w) (huw : u w ≠ 0)
    (hf : ∀ᶠ x in 𝓝 w, f x = u x * monomial p (x - a)) : NormalCrossingAt f w := by
  classical
  refine ⟨fun j => if w j = a j then p j else 0,
    fun x => u x * ∏ j, if w j = a j then 1 else (x j - a j) ^ p j, ?_, ?_, ?_⟩
  · refine hu.fun_mul (Finset.analyticAt_fun_prod _ fun j _ => ?_)
    split_ifs
    · exact analyticAt_const
    · exact ((((ContinuousLinearMap.proj j : Base n →L[ℝ] ℝ).analyticAt w).sub
        analyticAt_const).fun_pow _)
  · refine mul_ne_zero huw (Finset.prod_ne_zero_iff.2 fun j _ => ?_)
    split_ifs with h
    · exact one_ne_zero
    · exact pow_ne_zero _ (sub_ne_zero.2 h)
  · filter_upwards [hf] with x hx
    rw [hx, mul_assoc, monomial, monomial, ← Finset.prod_mul_distrib]
    congr 1
    refine Finset.prod_congr rfl fun j _ => ?_
    split_ifs with h
    · simp [h]
    · simp

/-- Normal crossings is an open condition. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem NormalCrossingAt.eventually {f : Base n → ℝ} {w : Base n}
    (h : NormalCrossingAt f w) : ∀ᶠ w' in 𝓝 w, NormalCrossingAt f w' := by
  obtain ⟨p, u, hu, huw, hf⟩ := h
  filter_upwards [hu.eventually_analyticAt, hu.continuousAt.eventually_ne huw,
    hf.eventually_nhds] with w' hu' huw' hf'
  exact normalCrossingAt_of_eq_mul_monomial w p hu' huw' hf'

/-- A function not vanishing at `w` has normal crossings there. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem normalCrossingAt_of_ne_zero {u : Base n → ℝ} {w : Base n} (hu : AnalyticAt ℝ u w)
    (huw : u w ≠ 0) : NormalCrossingAt u w :=
  ⟨0, u, hu, huw, Eventually.of_forall fun x => by rw [monomial_zero_exp, mul_one]⟩

theorem NormalCrossingAt.mul {f g : Base n → ℝ} {w : Base n} (hf : NormalCrossingAt f w)
    (hg : NormalCrossingAt g w) : NormalCrossingAt (fun x => f x * g x) w := by
  obtain ⟨p, u, hu, huw, hf⟩ := hf
  obtain ⟨q, v, hv, hvw, hg⟩ := hg
  refine ⟨p + q, fun x => u x * v x, hu.fun_mul hv, mul_ne_zero huw hvw, ?_⟩
  filter_upwards [hf, hg] with x hfx hgx
  rw [hfx, hgx, monomial_add_exp]
  ring

theorem NormalCrossingAt.analyticAt {f : Base n → ℝ} {w : Base n}
    (h : NormalCrossingAt f w) : AnalyticAt ℝ f w := by
  obtain ⟨p, u, hu, huw, hf⟩ := h
  exact (hu.fun_mul (analyticAt_monomial_sub p w w)).congr (hf.mono fun x hx => hx.symm)

/-- A function with normal crossings does not vanish identically near the
point: along the diagonal `w + (s, …, s)` it is a unit times `s ^ |p|`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem NormalCrossingAt.not_eventuallyEq_zero {f : Base n → ℝ} {w : Base n}
    (h : NormalCrossingAt f w) : ¬ f =ᶠ[𝓝 w] 0 := by
  obtain ⟨p, u, hu, huw, hf⟩ := h
  intro h0
  have hc : Tendsto (fun s : ℝ => w + fun _ => s) (𝓝[≠] 0) (𝓝 w) := by
    have h1 : Continuous (fun s : ℝ => w + fun _ => s) := by fun_prop
    have h2 := (h1.tendsto 0).mono_left (nhdsWithin_le_nhds (s := {0}ᶜ))
    have h3 : (w + fun _ : Fin n => (0 : ℝ)) = w := by ext; simp
    rwa [h3] at h2
  obtain ⟨s, ⟨hus, hfs, h0s⟩, hs⟩ := ((hc.eventually ((hu.continuousAt.eventually_ne huw).and
    (hf.and h0))).and self_mem_nhdsWithin).exists
  rw [h0s, Pi.zero_apply, add_sub_cancel_left, monomial_const] at hfs
  exact mul_ne_zero hus (pow_ne_zero _ hs) hfs.symm

end Transformer.Landscape.Uniformization
