/-
# Root accumulation passes through analytic preparation

A nonvanishing preparation unit can be cancelled at roots accumulating
from positive parameters. The resulting prepared polynomial has real roots
frequently on the same positive side.
-/

import Transformer.Normalization.PolynomialFamily

open Filter Set

namespace Transformer.Normalization

/-- A local preparation at a central root preserves positive-side
accumulation of real roots. The unit is checked to be nonzero on a
neighborhood before cancellation. Auxiliary for the degree induction in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem frequent_roots_of_polynomial_preparation {d m : ℕ}
    (a : Fin d → ℝ → ℝ) (b : Fin m → ℝ → ℝ) (r : ℝ) (u : ℝ × ℝ → ℝ)
    (hu : ContinuousAt u 0) (hu0 : u 0 ≠ 0)
    (hidentity : ∀ᶠ x in nhds (0 : ℝ × ℝ),
      polynomialFamily d a (x.1, r + x.2) = u x * polynomialFamily m b x)
    (hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
      polynomialFamily d a (x.1, r + x.2) = 0) :
    ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ, polynomialFamily m b (t, y) = 0 := by
  by_contra h
  have hno : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ¬ ∃ y : ℝ, polynomialFamily m b (t, y) = 0 := not_frequently.mp h
  have hno0 : ∀ᶠ t in nhds (0 : ℝ), 0 < t →
      ¬ ∃ y : ℝ, polynomialFamily m b (t, y) = 0 := eventually_nhdsWithin_iff.mp hno
  have hfst : Tendsto (fun x : ℝ × ℝ => x.1) (nhds 0) (nhds 0) :=
    continuous_fst.tendsto 0
  have hunit : ∀ᶠ x in nhds (0 : ℝ × ℝ), u x ≠ 0 := hu.eventually_ne hu0
  obtain ⟨x, hx⟩ := (hacc.and_eventually
    (hidentity.and (hunit.and (hfst.eventually hno0)))).exists
  have hroot : polynomialFamily m b x = 0 :=
    (mul_eq_zero.mp (hx.2.1.symm.trans hx.1.2)).resolve_left hx.2.2.1
  exact hx.2.2.2 hx.1.1 ⟨x.2, hroot⟩

/-- The family `y² - t` already is its own preparation with unit one.
Positive roots accumulate at the origin, so all cancellation and
accumulation hypotheses hold jointly. Auxiliary example for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 2 → ℝ → ℝ := fun i t => if i = 0 then -t else 0
    ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), ∃ y : ℝ, polynomialFamily 2 a (t, y) = 0 := by
  have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
  have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
  let gamma : ℝ → ℝ × ℝ := fun t => (t ^ 2, t)
  have hgamma : Tendsto gamma (nhdsWithin 0 (Ioi 0)) (nhds (0 : ℝ × ℝ)) := by
    have hc : ContinuousAt gamma 0 := by fun_prop
    simpa [gamma, Prod.mk_zero_zero] using hc.tendsto.mono_left nhdsWithin_le_nhds
  have hacc : ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
      polynomialFamily 2 (fun i t => if i = 0 then -t else 0) x = 0 := by
    apply hgamma.frequently
    apply hpositive.frequently.mono
    intro t ht
    refine ⟨sq_pos_of_pos ht, ?_⟩
    simp [gamma, polynomialFamily]
  apply frequent_roots_of_polynomial_preparation
    (d := 2) (m := 2)
    (a := fun i t => if i = 0 then -t else 0)
    (b := fun i t => if i = 0 then -t else 0) (r := 0) (u := fun _ => 1)
    (hu := continuousAt_const) (hu0 := one_ne_zero)
  · exact Eventually.of_forall (fun x => by simp)
  · simpa only [zero_add] using hacc

end Transformer.Normalization
