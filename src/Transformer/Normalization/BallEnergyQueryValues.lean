/-
# Exact scalar queries on prepared closed-ball energy patches

Both the original ball constraint and the actual squared gradient norm
retain their values at every prepared root on the whole box.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.BallEnergyPatchSource
import Transformer.Normalization.PolynomialFiberThreshold

noncomputable section
open scoped BigOperators

namespace Transformer.Normalization

open AnalyticPreparation

/-- The genuine simultaneous value relations supplied by preparation
and division on an energy-graph chart. This predicate asserts only the
equation and actual remainder values; it contains no minimum or selection
claim. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def BallEnergyQueryValues {n d : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w center : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1))
    (R r : ℝ) (a b v : Fin d → Base (n + 1) → ℝ) : Prop :=
  ∀ z : Base (n + 1), ‖z‖ < r → ∀ y : ℝ, |y| < r →
    (E (energyChartMap w L (z, y)) - energyChartLevel E w z = 0 ↔
      preparedPolynomial d a (z, y) = 0) ∧
    (preparedPolynomial d a (z, y) = 0 →
      energyChartBallConstraint w center L R (z, y) = ∑ i : Fin d, b i z * y ^ (i : ℕ)) ∧
    (preparedPolynomial d a (z, y) = 0 →
      squaredGradientNorm E (energyChartMap w L (z, y)) = ∑ i : Fin d, v i z * y ^ (i : ℕ))

/-- The arithmetic test for a common squared-gradient threshold on one
ball-constrained energy patch. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
def ballEnergyLowerBoundQuery {n d : ℕ} (a b v : Fin d → Base (n + 1) → ℝ)
    (z : Base (n + 1)) (threshold r : ℝ) : ℤ :=
  fiberLowerBoundQuery a (fun _ : Fin 1 => b) v (fun _ => .nonnegative) z threshold r

/-- The entire prepared patch satisfies a squared-gradient lower bound
on the chosen energy level exactly when all its scalar queries vanish.
No source point is replaced by a branch or by the candidate's own base
coordinates. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem ball_energy_patch_lower_bound_iff {n d : ℕ} (E : EucSpace (n + 1) → ℝ)
    (w center : EucSpace (n + 1)) (L : Ambient n ≃L[ℝ] EucSpace (n + 1))
    (R r : ℝ) (a b v : Fin d → Base (n + 1) → ℝ)
    (hvalues : BallEnergyQueryValues E w center L R r a b v) (c threshold : ℝ) :
    (∀ x ∈ ballEnergyPatchSource E w center L R r,
      E x = c → threshold ≤ squaredGradientNorm E x) ↔
    ∀ z : Base (n + 1), ‖z‖ < r → energyChartLevel E w z = c →
      ballEnergyLowerBoundQuery a b v z threshold r = 0 := by
  constructor
  · intro hbound z hz hlevel
    apply (fiberLowerBoundQuery_iff a (fun _ : Fin 1 => b) v (fun _ => .nonnegative)
      z threshold r).mp
    intro y hy hP hG
    have hzy := hvalues z hz y hy
    have hconstraint : 0 ≤ energyChartBallConstraint w center L R (z, y) := by
      rw [hzy.2.1 hP]
      exact hG 0
    have hgraph := sub_eq_zero.mp (hzy.1.mpr hP)
    have hmem : energyChartMap w L (z, y) ∈ ballEnergyPatchSource E w center L R r :=
      ⟨(z, y), ⟨hz, hy, hconstraint, hgraph⟩, rfl⟩
    have hbnd := hbound _ hmem (hgraph.trans hlevel)
    rwa [hzy.2.2 hP] at hbnd
  · intro hqueries x hx hxc
    obtain ⟨u, hu, rfl⟩ := hx
    obtain ⟨hz, hy, hG, hgraph⟩ := hu
    have huz := hvalues u.1 hz u.2 hy
    have hP := huz.1.mp (sub_eq_zero.mpr hgraph)
    have hlevel : energyChartLevel E w u.1 = c := hgraph.symm.trans hxc
    have hconstraint : ∀ i : Fin 1, AnalyticSignRequirement.nonnegative.Holds
        (∑ t : Fin d, b t u.1 * u.2 ^ (t : ℕ)) := by
      intro i
      rw [← huz.2.1 hP]
      exact hG
    have hbnd := (fiberLowerBoundQuery_iff a (fun _ : Fin 1 => b) v (fun _ => .nonnegative)
      u.1 threshold r).mpr (hqueries u.1 hz hlevel) u.2 hy hP hconstraint
    rwa [← huz.2.2 hP] at hbnd

end Transformer.Normalization
