/-
# Coordinate monomials and directional derivatives

Elementary local models for the analytic gradient estimate used in
Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Basic
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Analytic.Constructions
import Mathlib.Analysis.Analytic.Linear
import Mathlib.Data.Finset.Max

open Filter Set
open scoped BigOperators

namespace Transformer.Normalization

/-- Coordinate monomials model normal-crossing energy germs for the analytic
input of Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def coordinateMonomial {N : ℕ} (p : Fin N → ℕ) (v : EucSpace N) : ℝ :=
  ∏ j, (v j) ^ p j

/-- Scaling one coordinate multiplies a coordinate monomial by the
corresponding power. Auxiliary identity for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma coordinateMonomial_scale_coordinate {N : ℕ} (p : Fin N → ℕ)
    (v : EucSpace N) (i : Fin N) (t : ℝ) :
    coordinateMonomial p (v + t • PiLp.single 2 i (v i)) =
      (1 + t) ^ p i * coordinateMonomial p v := by
  classical
  have heq : ∏ j ∈ Finset.univ.erase i,
      (v + t • PiLp.single 2 i (v i) : EucSpace N).ofLp j ^ p j =
      ∏ j ∈ Finset.univ.erase i, (v j) ^ p j := by
    apply Finset.prod_congr rfl
    intro j hj
    have hji : j ≠ i := (Finset.mem_erase.mp hj).1
    simp [hji]
  unfold coordinateMonomial
  rw [← Finset.prod_erase_mul Finset.univ
    (fun j : Fin N =>
      (v + t • (PiLp.single 2 i (v.ofLp i) : EucSpace N)).ofLp j ^ p j)
    (Finset.mem_univ i), heq]
  simp only [PiLp.add_apply, PiLp.smul_apply, PiLp.single_apply, ite_true, smul_eq_mul]
  rw [show v i + t * v i = (1 + t) * v i by ring, mul_pow]
  rw [← Finset.prod_erase_mul Finset.univ (fun j : Fin N => (v j) ^ p j)
    (Finset.mem_univ i)]
  ring

/-- The smallest active coordinate controls the absolute monomial from
below by its total-degree power. Auxiliary estimate for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma coordinateMonomial_min_power {N : ℕ} (p : Fin N → ℕ)
    (hp : 0 < ∑ j, p j) (v : EucSpace N) :
    ∃ i : Fin N, 0 < p i ∧ |v i| ^ (∑ j, p j) ≤ |coordinateMonomial p v| := by
  classical
  let S : Finset (Fin N) := Finset.univ.filter fun i => 0 < p i
  have hS : S.Nonempty := by
    obtain ⟨i, hi, hpi⟩ := Finset.sum_pos_iff.1 hp
    exact ⟨i, Finset.mem_filter.mpr ⟨hi, hpi⟩⟩
  obtain ⟨i, hi, hmin⟩ := S.exists_min_image (fun j => |v j|) hS
  refine ⟨i, (Finset.mem_filter.mp hi).2, ?_⟩
  rw [coordinateMonomial, Finset.abs_prod]
  simp_rw [abs_pow]
  rw [← Finset.prod_pow_eq_pow_sum Finset.univ p |v i|]
  apply Finset.prod_le_prod₀
  · intro j hj
    positivity
  · intro j hj
    by_cases hpj : 0 < p j
    · exact pow_le_pow_left₀ (abs_nonneg _) (hmin j (Finset.mem_filter.mpr ⟨hj, hpj⟩)) _
    · simp [show p j = 0 by omega]

/-- Differentiating in one coordinate direction separates the monomial
term from the unit derivative. Auxiliary identity for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma coordinateMonomial_unit_direction {N : ℕ}
    (p : Fin N → ℕ) (E u : EucSpace N → ℝ) (z y : EucSpace N) (c : ℝ)
    (hE : DifferentiableAt ℝ E y) (hu : DifferentiableAt ℝ u y)
    (heq : ∀ᶠ w in nhds y, E w - c = u w * coordinateMonomial p (w - z))
    (i : Fin N) :
    inner (𝕜 := ℝ) (gradient E y) (PiLp.single 2 i ((y - z) i)) =
      ((p i : ℝ) * u y +
        inner (𝕜 := ℝ) (gradient u y) (PiLp.single 2 i ((y - z) i))) *
        coordinateMonomial p (y - z) := by
  let v : EucSpace N := PiLp.single 2 i ((y - z) i)
  let a : ℝ → EucSpace N := fun t => y + t • v
  have ha : HasDerivAt a v 0 := by
    simpa [a] using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add y
  have hu' : HasDerivAt (fun t => u (a t))
      (inner (𝕜 := ℝ) (gradient u y) v) 0 := by
    have hu0 : HasFDerivAt u (fderiv ℝ u y) (a 0) := by
      simpa [a] using hu.hasFDerivAt
    simpa only [Function.comp_def, hu.hasGradientAt.fderiv_apply] using
      hu0.comp_hasDerivAt (0 : ℝ) ha
  have hE' : HasDerivAt (fun t => E (a t) - c)
      (inner (𝕜 := ℝ) (gradient E y) v) 0 := by
    have hE0 : HasFDerivAt E (fderiv ℝ E y) (a 0) := by
      simpa [a] using hE.hasFDerivAt
    simpa only [Function.comp_def, hE.hasGradientAt.fderiv_apply] using
      (hE0.comp_hasDerivAt (0 : ℝ) ha).sub_const c
  have hp' : HasDerivAt (fun t : ℝ => (1 + t) ^ p i) (p i : ℝ) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).const_add 1).fun_pow (p i)
  have hprod : HasDerivAt
      (fun t => u (a t) * ((1 + t) ^ p i * coordinateMonomial p (y - z)))
      (((p i : ℝ) * u y + inner (𝕜 := ℝ) (gradient u y) v) *
        coordinateMonomial p (y - z)) 0 := by
    convert hu'.mul (hp'.mul_const (coordinateMonomial p (y - z))) using 1
    simp [a]
    ring
  have hat : Tendsto a (nhds 0) (nhds y) := by
    simpa [a] using ha.continuousAt.tendsto
  have hident : (fun t => E (a t) - c) =ᶠ[nhds 0]
      (fun t => u (a t) * ((1 + t) ^ p i * coordinateMonomial p (y - z))) := by
    filter_upwards [hat.eventually heq] with t ht
    rw [ht, show a t - z = (y - z) + t • v by dsimp [a]; abel]
    dsimp only [v]
    rw [coordinateMonomial_scale_coordinate]
  exact hE'.unique (hprod.congr_of_eventuallyEq hident)

/-- Coordinate monomials are analytic local models for the energy in
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem coordinateMonomial_analytic {N : ℕ} (p : Fin N → ℕ) (z : EucSpace N) :
    AnalyticAt ℝ (coordinateMonomial p) z := by
  unfold coordinateMonomial
  apply Finset.univ.analyticAt_fun_prod
  intro i hi
  exact ((EuclideanSpace.proj i : EucSpace N →L[ℝ] ℝ).analyticAt z).fun_pow (p i)

/-- The positive-degree and derivative hypotheses above hold simultaneously
for a squared coordinate, the constant unit one, and a nonzero state;
Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ (p : Fin 1 → ℕ) (E u : EucSpace 1 → ℝ) (z y : EucSpace 1) (c : ℝ),
    0 < ∑ j, p j ∧ DifferentiableAt ℝ E y ∧ DifferentiableAt ℝ u y ∧
      (∀ᶠ w in nhds y, E w - c = u w * coordinateMonomial p (w - z)) := by
  refine ⟨fun _ => 2, coordinateMonomial (fun _ => 2), fun _ => 1, 0,
    PiLp.single 2 (0 : Fin 1) 1, 0, by simp, ?_, differentiableAt_const _, ?_⟩
  · exact (coordinateMonomial_analytic _ _).differentiableAt
  · exact Filter.Eventually.of_forall (fun w => by simp)

end Transformer.Normalization
