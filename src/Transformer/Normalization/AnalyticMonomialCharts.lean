/-
# Compact monomial charts of analytic energies

Local uniformization (`Transformer.Uniformization`) gives every analytic energy that is
not locally constant finitely many compact monomial charts at every point, the
hypothesis of `analytic_gradient_inequality_of_compact_monomial_charts`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.CompactMonomialCharts
import Transformer.Uniformization

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Normalization

open Transformer.AnalyticPreparation (Base)

/-- Every analytic energy that is not locally constant at `z` has compact monomial
charts at `z`. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem hasCompactMonomialCharts_of_analyticAt {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) (hE : AnalyticAt ℝ E z) (hconstant : ¬ ∀ᶠ y in 𝓝 z, E y = E z) :
    HasCompactMonomialCharts E z := by
  classical
  let T : EucSpace N ≃L[ℝ] Base N := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin N => ℝ)
  have hTA : ∀ y, AnalyticAt ℝ T y := fun y => (T : EucSpace N →L[ℝ] Base N).analyticAt y
  have hTsA : ∀ x, AnalyticAt ℝ T.symm x :=
    fun x => (T.symm : Base N →L[ℝ] EucSpace N).analyticAt x
  have hmon : ∀ (p : Fin N → ℕ) (v : EucSpace N),
      Uniformization.monomial p (T v) = coordinateMonomial p v := fun _ _ => rfl
  let f : Base N → ℝ := fun x => E (T.symm x) - E z
  have hf : AnalyticAt ℝ f (T z) :=
    (hE.comp_of_eq (hTsA (T z)) (T.symm_apply_apply z)).fun_sub analyticAt_const
  have hf0 : ¬ f =ᶠ[𝓝 (T z)] 0 := by
    intro h
    apply hconstant
    filter_upwards [T.continuous.continuousAt.tendsto.eventually h] with y hy
    have hy' : E (T.symm (T y)) - E z = 0 := hy
    rw [T.symm_apply_apply] at hy'
    linarith
  obtain ⟨S, hS, hc, hcover⟩ := Uniformization.hasMonomialCharts_of_analyticAt hf hf0
  let c : Fin hS.toFinset.card → (Base N → Base N) × Set (Base N) :=
    fun j => (hS.toFinset.equivFin.symm j).1
  have hcS : ∀ j, c j ∈ S := fun j => hS.mem_toFinset.1 (hS.toFinset.equivFin.symm j).2
  refine ⟨N, hS.toFinset.card, fun j y => T.symm ((c j).1 (T y)), fun j => T.symm '' (c j).2,
    fun j => (hc _ (hcS j)).1.image T.symm.continuous, fun j w hw => ?_,
    fun j w hw hwz => ?_, ?_⟩
  · obtain ⟨a, ha, rfl⟩ := hw
    have h1 : AnalyticAt ℝ (c j).1 (T (T.symm a)) := by
      rw [T.apply_symm_apply]
      exact (hc _ (hcS j)).2.1 a ha
    exact (hTsA _).comp (h1.comp (hTA _))
  · obtain ⟨a, ha, rfl⟩ := hw
    obtain ⟨p, u, hu, hu0, hfu⟩ := (hc _ (hcS j)).2.2 a ha
    simp only [T.apply_symm_apply] at hwz
    have hTt : Tendsto T (𝓝 (T.symm a)) (𝓝 a) := by
      have h := T.continuous.continuousAt.tendsto (x := T.symm a)
      rwa [T.apply_symm_apply] at h
    refine ⟨p, fun y => u (T y), Nat.pos_of_ne_zero fun hp => hu0 ?_,
      hu.comp_of_eq (hTA _) (T.apply_symm_apply a), by simpa only [T.apply_symm_apply], ?_⟩
    · have hp0 : p = 0 := funext fun i => Finset.sum_eq_zero_iff.1 hp i (Finset.mem_univ i)
      have h := hfu.self_of_nhds
      simp only [Function.comp_apply, f, hwz, sub_self, hp0, Uniformization.monomial_zero_exp,
        mul_one] at h
      exact h.symm
    · filter_upwards [hTt.eventually hfu] with y hy
      have hsub : T y - a = T (y - T.symm a) := by rw [map_sub, T.apply_symm_apply]
      show E (T.symm ((c j).1 (T y))) - E z = u (T y) * coordinateMonomial p (y - T.symm a)
      rw [← hmon, ← hsub]
      exact hy
  · filter_upwards [T.continuous.continuousAt.tendsto.eventually hcover] with y hy
    obtain ⟨c', hc'S, a, ha, hca⟩ := Set.mem_iUnion₂.1 hy
    have hj : c (hS.toFinset.equivFin ⟨c', hS.mem_toFinset.2 hc'S⟩) = c' := by
      simp only [c, Equiv.symm_apply_apply]
    refine ⟨hS.toFinset.equivFin ⟨c', hS.mem_toFinset.2 hc'S⟩, T.symm a,
      ⟨a, by rw [hj]; exact ha, rfl⟩, ?_⟩
    simp only [hj, T.apply_symm_apply, hca, T.symm_apply_apply]

/-- The energy `y₀²` on `ℝ¹` is analytic and not locally constant at the origin, so it has
compact monomial charts there. Appendix D.1 of arXiv:2510.22026v2. -/
example : HasCompactMonomialCharts (coordinateMonomial (fun _ : Fin 1 => 2)) 0 := by
  refine hasCompactMonomialCharts_of_analyticAt _ 0 (coordinateMonomial_analytic _ _)
    fun h => ?_
  obtain ⟨ε, hε, hball⟩ := Metric.eventually_nhds_iff.1 h
  have hy := hball (y := PiLp.single 2 (0 : Fin 1) (ε / 2)) (by
    rw [dist_zero_right, PiLp.norm_single, Real.norm_eq_abs, abs_of_pos (half_pos hε)]
    exact half_lt_self hε)
  simp [coordinateMonomial] at hy
  linarith

end Transformer.Normalization
