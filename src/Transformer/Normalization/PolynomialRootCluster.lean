/-
# Central limits of real polynomial roots

Uniform monic root bounds give a convergent subsequence of real roots at
parameters approaching zero from above. Its limit is a root of the actual
central polynomial, and the shifted roots accumulate at the origin.
-/

import Transformer.Normalization.PolynomialRootBounds
import Mathlib.Topology.Sequences
import Mathlib.Order.Filter.AtTopBot.CountablyGenerated
import Mathlib.Analysis.Real.Sqrt

open Filter Set

namespace Transformer.Normalization

/-- Frequently occurring positive-parameter real roots of an analytic monic
family accumulate at a root of its central polynomial. The joint shifted
root set accumulates at the origin, retaining the positive-side condition.
Auxiliary for the degree induction in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_polynomial_root_cluster {n : ℕ} (a : Fin (n + 1) → ℝ → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (hroots : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      ∃ y : ℝ, polynomialFamily (n + 1) a (t, y) = 0) :
    ∃ r : ℝ, (parameterPolynomial (n + 1) a 0).IsRoot r ∧
      ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
        polynomialFamily (n + 1) a (x.1, r + x.2) = 0 := by
  obtain ⟨B, hB, hbound⟩ := analytic_polynomial_roots_bounded a ha
  have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
  have hbounded : ∃ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0),
      0 < t ∧ ∃ y : ℝ, polynomialFamily (n + 1) a (t, y) = 0 ∧ |y| ≤ B := by
    apply (hroots.and_eventually
      ((hbound.filter_mono nhdsWithin_le_nhds).and hpositive)).mono
    intro t ht
    obtain ⟨y, hy⟩ := ht.1
    exact ⟨ht.2.2, y, hy, ht.2.1 y hy⟩
  obtain ⟨ts, hts, hseq⟩ := frequently_iff_seq_forall.mp hbounded
  choose ys hys hysB using fun k => (hseq k).2
  have hysIcc (k : ℕ) : ys k ∈ Icc (-B) B := abs_le.mp (hysB k)
  obtain ⟨r, hr, phi, hphi, hyr⟩ := (isCompact_Icc : IsCompact (Icc (-B) B)).tendsto_subseq hysIcc
  have htzero : Tendsto (fun k => ts (phi k)) atTop (nhds (0 : ℝ)) :=
    (hts.mono_right nhdsWithin_le_nhds).comp hphi.tendsto_atTop
  let gamma : ℕ → ℝ × ℝ := fun k => (ts (phi k), ys (phi k) - r)
  have hgamma : Tendsto gamma atTop (nhds (0 : ℝ × ℝ)) := by
    simpa only [gamma, Function.comp_def, sub_self, Prod.mk_zero_zero] using
      htzero.prodMk_nhds (hyr.sub_const r)
  have hgammazero (k : ℕ) : polynomialFamily (n + 1) a
      ((gamma k).1, r + (gamma k).2) = 0 := by
    simpa only [gamma, add_sub_cancel] using hys (phi k)
  have hvalue : polynomialFamily (n + 1) a (0, r) = 0 := by
    have hlim := (polynomialFamily_analyticAt a ha r).continuousAt.tendsto.comp hgamma
    have heq : (fun k => polynomialFamily (n + 1) a ((gamma k).1, r + (gamma k).2)) =
        fun _ => (0 : ℝ) := by funext k; exact hgammazero k
    have hzlim : Tendsto (fun k => polynomialFamily (n + 1) a
        ((gamma k).1, r + (gamma k).2)) atTop (nhds (0 : ℝ)) := by
      rw [heq]
      exact tendsto_const_nhds
    simpa only [Prod.fst_zero, Prod.snd_zero, add_zero] using tendsto_nhds_unique hlim hzlim
  refine ⟨r, by simpa only [Polynomial.IsRoot, parameterPolynomial_eval] using hvalue, ?_⟩
  apply hgamma.frequently
  apply Eventually.frequently
  exact Eventually.of_forall (fun k => ⟨(hseq (phi k)).1, hgammazero k⟩)

/-- The family `y² - (1 + t)` has real roots at every positive parameter.
Its central roots are nonzero, so the example exercises root accumulation
away from the original distinguished-variable origin. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let a : Fin 2 → ℝ → ℝ := fun i t => if i = 0 then -(1 + t) else 0
    ∃ r : ℝ, (parameterPolynomial 2 a 0).IsRoot r ∧
      ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ polynomialFamily 2 a (x.1, r + x.2) = 0 := by
  apply analytic_polynomial_root_cluster
  · intro i
    split_ifs
    · exact (analyticAt_const.fun_add analyticAt_id).fun_neg
    · exact analyticAt_const
  · have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply hpositive.frequently.mono
    intro t ht
    refine ⟨Real.sqrt (1 + t), ?_⟩
    simpa [polynomialFamily, Fin.sum_univ_succ, sub_eq_add_neg] using
      sub_eq_zero.mpr (Real.sq_sqrt (by linarith : 0 ≤ 1 + t))

end Transformer.Normalization
