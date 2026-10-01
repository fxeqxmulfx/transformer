/-
# Arithmetic comparison on every chart of a finite analytic source

One threshold is compared with all points on the same energy level
in the entire source, including charts outside the candidate's chart.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticBoxMinimumQuery
import Transformer.Normalization.AnalyticAtlasSource
import Transformer.Normalization.GradientEnergyImage

noncomputable section
open scoped BigOperators

namespace Transformer.Normalization

open AnalyticPreparation

/-- Preparation and division on a finite analytic coordinate family
eliminate the scalar competitor in every chart simultaneously. The source
is the entire union of constrained energy-graph chart images, so the comparison is not
restricted to the candidate's chart. Degrees, dimensions, and constraints
may differ between charts. No analytic base curve or selected minimum is
assumed, and all base quantifiers remain explicit. Auxiliary for the
global minimum projection step of Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem analytic_atlas_lower_bounds_arithmetic {k N : ℕ}
    (n d m : Fin k → ℕ) (E V : EucSpace N → ℝ)
    (phi : ∀ j, Ambient (n j) → EucSpace N)
    (hphi : ∀ j, AnalyticAt ℝ (phi j) 0)
    (hE : ∀ j, AnalyticAt ℝ E (phi j 0))
    (hV : ∀ j, AnalyticAt ℝ V (phi j 0))
    (level : ∀ j, Base (n j) → ℝ) (hlevel : ∀ j, AnalyticAt ℝ (level j) 0)
    (horder : ∀ j, ExactOrderInLastVariable
      (fun x => E (phi j x) - level j x.1) (d j))
    (G : ∀ j, Fin (m j) → Ambient (n j) → ℝ)
    (hG : ∀ j i, AnalyticAt ℝ (G j i) 0)
    (requirement : ∀ j, Fin (m j) → AnalyticSignRequirement) :
    ∃ (a : ∀ j, Fin (d j) → Base (n j) → ℝ)
      (b : ∀ j, Fin (m j) → Fin (d j) → Base (n j) → ℝ)
      (v : ∀ j, Fin (d j) → Base (n j) → ℝ) (r : Fin k → ℝ),
      (∀ j, 0 < r j) ∧ (∀ j i, AnalyticAt ℝ (a j i) 0) ∧
      (∀ j i, a j i 0 = 0) ∧ (∀ j i t, AnalyticAt ℝ (b j i t) 0) ∧
      (∀ j i, AnalyticAt ℝ (v j i) 0) ∧
      ∀ c threshold : ℝ,
        (∀ w ∈ analyticEnergyAtlasSource n m E phi level G requirement r,
          E w = c → threshold ≤ V w) ↔
        ∀ j, ∀ z : Base (n j), ‖z‖ < r j → level j z = c →
          fiberLowerBoundQuery (a j) (b j) (v j) (requirement j) z threshold (r j) = 0 := by
  have hF (j : Fin k) :
      AnalyticAt ℝ (fun x : Ambient (n j) => E (phi j x) - level j x.1) 0 :=
    ((hE j).comp (hphi j)).sub
      ((hlevel j).comp (f := fun x : Ambient (n j) => x.1) (x := 0) analyticAt_fst)
  choose a b v r hr ha ha0 hb hv hvalues using fun j =>
    analytic_fiber_polynomial_values _ (hF j) (horder j) (G j) (hG j)
      (fun x => V (phi j x)) ((hV j).comp (hphi j))
  refine ⟨a, b, v, r, hr, ha, ha0, hb, hv, ?_⟩
  intro c threshold
  constructor
  · intro hbound j z hz hzc
    apply (fiberLowerBoundQuery_iff (a j) (b j) (v j) (requirement j) z threshold (r j)).mp
    intro t ht hPt hGt
    have hzt := hvalues j z hz t ht
    have horiginal : ∀ i, (requirement j i).Holds (G j i (z, t)) := by
      intro i
      rw [hzt.2.1 hPt i]
      exact hGt i
    have hmem : phi j (z, t) ∈ analyticEnergyAtlasSource n m E phi level G requirement r :=
      (mem_analyticEnergyAtlasSource_iff n m E phi level G requirement r _).mpr
        ⟨j, (z, t), hz, ht, horiginal, sub_eq_zero.mp (hzt.1.mpr hPt), rfl⟩
    have heq : E (phi j (z, t)) = c :=
      (sub_eq_zero.mp (hzt.1.mpr hPt)).trans hzc
    have hbnd := hbound _ hmem heq
    rwa [hzt.2.2 hPt] at hbnd
  · intro hqueries w hw hwc
    obtain ⟨j, x, hz, ht, hGx, hgraph, rfl⟩ :=
      (mem_analyticEnergyAtlasSource_iff n m E phi level G requirement r w).mp hw
    have hxc := hvalues j x.1 hz x.2 ht
    have hlevelc : level j x.1 = c := hgraph.symm.trans hwc
    have hP : preparedPolynomial (d j) (a j) x = 0 :=
      hxc.1.mp (sub_eq_zero.mpr (hwc.trans hlevelc.symm))
    have hpoly : ∀ i, (requirement j i).Holds
        (∑ t : Fin (d j), b j i t x.1 * x.2 ^ (t : ℕ)) := by
      intro i
      rw [← hxc.2.1 hP i]
      exact hGx i
    have hbnd := (fiberLowerBoundQuery_iff (a j) (b j) (v j) (requirement j)
      x.1 threshold (r j)).mpr (hqueries j x.1 hz hlevelc) x.2 ht hP hpoly
    rwa [← hxc.2.2 hP] at hbnd

end Transformer.Normalization
