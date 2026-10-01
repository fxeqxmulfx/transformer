/-
# Bounds for a monomial times a nonvanishing unit

The directional derivative and the smallest active coordinate give the
power estimate needed by Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.CoordinateMonomial
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open Filter Set
open scoped BigOperators

namespace Transformer.Normalization

/-- A sufficiently small unit variation gives an energy-gradient bound
in an active coordinate direction. Auxiliary estimate for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma coordinateMonomial_unit_gradient_bound {N : ℕ}
    (p : Fin N → ℕ) (E u : EucSpace N → ℝ) (z y : EucSpace N) (c : ℝ)
    (hE : DifferentiableAt ℝ E y) (hu : DifferentiableAt ℝ u y)
    (heq : ∀ᶠ w in nhds y, E w - c = u w * coordinateMonomial p (w - z))
    (i : Fin N) (hpi : 0 < p i)
    (hsmall : ‖gradient u y‖ * |(y - z) i| ≤ |u y| / 2) :
    |E y - c| ≤ 2 * |(y - z) i| * ‖gradient E y‖ := by
  let v : EucSpace N := PiLp.single 2 i ((y - z) i)
  let d : ℝ := inner (𝕜 := ℝ) (gradient u y) v
  let a : ℝ := (p i : ℝ) * u y + d
  have hd : inner (𝕜 := ℝ) (gradient E y) v =
      a * coordinateMonomial p (y - z) :=
    coordinateMonomial_unit_direction p E u z y c hE hu heq i
  have hdnorm : |d| ≤ ‖gradient u y‖ * |(y - z) i| := by
    simpa [d, v] using norm_inner_le_norm (𝕜 := ℝ) (gradient u y) v
  have htriangle : |(p i : ℝ) * u y| ≤ |a| + |d| := by
    simpa [a, abs_neg] using abs_add_le ((p i : ℝ) * u y + d) (-d)
  have hpR : (1 : ℝ) ≤ p i := by exact_mod_cast Nat.succ_le_iff.mpr hpi
  have hpu : |u y| ≤ |(p i : ℝ) * u y| := by
    rw [abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ p i)]
    nlinarith [abs_nonneg (u y)]
  have hcoef : |u y| ≤ 2 * |a| := by linarith
  calc
    |E y - c| = |u y| * |coordinateMonomial p (y - z)| := by
      rw [heq.self_of_nhds, abs_mul]
    _ ≤ (2 * |a|) * |coordinateMonomial p (y - z)| :=
      mul_le_mul_of_nonneg_right hcoef (abs_nonneg _)
    _ = 2 * |inner (𝕜 := ℝ) (gradient E y) v| := by rw [hd, abs_mul]; ring
    _ ≤ 2 * (‖gradient E y‖ * ‖v‖) := by
      exact mul_le_mul_of_nonneg_left
        (norm_inner_le_norm (𝕜 := ℝ) (gradient E y) v) (by norm_num)
    _ = 2 * |(y - z) i| * ‖gradient E y‖ := by simp [v]; ring

/-- Combining coordinate, unit, and gradient bounds gives a power
exponent strictly below one. Auxiliary scalar estimate for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma monomial_gradient_power_bound (m : ℕ) (b r P F g : ℝ)
    (hb : 0 < b) (hr : 0 ≤ r) (hr1 : r ≤ 1) (hF : 0 ≤ F) (hg : 0 ≤ g)
    (hpower : r ^ m ≤ P) (hunit : b * P ≤ F) (hgrad : F ≤ 2 * r * g) :
    F ^ (1 - 1 / (max m 2 : ℕ) : ℝ) ≤ (2 / b ^ (1 / (max m 2 : ℕ) : ℝ)) * g := by
  let M : ℕ := max m 2
  let beta : ℝ := 1 / M
  have hM : (2 : ℝ) ≤ M := by exact_mod_cast Nat.le_max_right m 2
  have hMpos : (0 : ℝ) < M := by linarith
  have hbeta : 0 < beta := div_pos zero_lt_one hMpos
  have hbeta1 : beta < 1 := (div_lt_one hMpos).2 (by linarith)
  change F ^ (1 - beta) ≤ (2 / b ^ beta) * g
  by_cases hzero : F = 0
  · rw [hzero, Real.zero_rpow (ne_of_gt (by linarith : 0 < 1 - beta))]
    positivity
  have hFpos : 0 < F := lt_of_le_of_ne hF (Ne.symm hzero)
  have hP : P ≤ F / b := (le_div_iff₀ hb).2 (by simpa [mul_comm] using hunit)
  have hbase : r ^ M ≤ F / b :=
    (pow_le_pow_of_le_one hr hr1 (Nat.le_max_left m 2)).trans (hpower.trans hP)
  have hrbeta := Real.rpow_le_rpow (pow_nonneg hr M) hbase hbeta.le
  have hroot : (r ^ M) ^ beta = r := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hr]
    have hexp : (M : ℝ) * beta = 1 := by dsimp [beta]; field_simp
    rw [hexp, Real.rpow_one]
  rw [hroot, Real.div_rpow hF hb.le beta] at hrbeta
  rw [Real.rpow_sub hFpos 1 beta, Real.rpow_one]
  apply (div_le_iff₀ (Real.rpow_pos_of_pos hFpos beta)).2
  calc
    F ≤ 2 * r * g := hgrad
    _ ≤ 2 * (F ^ beta / b ^ beta) * g :=
      mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_left hrbeta (by norm_num)) hg
    _ = (2 / b ^ beta * g) * F ^ beta := by ring

/-- The directional estimate applies to a squared coordinate at a nonzero
state with unit one; Appendix D.1 of arXiv:2510.22026v2. -/
example : |coordinateMonomial (fun _ : Fin 1 => 2) (PiLp.single 2 (0 : Fin 1) 1)| ≤
    2 * ‖gradient (coordinateMonomial (fun _ : Fin 1 => 2))
      (PiLp.single 2 (0 : Fin 1) 1)‖ := by
  simpa using coordinateMonomial_unit_gradient_bound (fun _ : Fin 1 => 2)
    (coordinateMonomial (fun _ => 2)) (fun _ => 1) 0 (PiLp.single 2 (0 : Fin 1) 1) 0
    (coordinateMonomial_analytic _ _).differentiableAt (differentiableAt_const _)
    (Filter.Eventually.of_forall (fun w => by simp)) 0 (by norm_num) (by simp)

/-- The scalar power-bound hypotheses are simultaneously satisfied by
`m = 4` and all scalar inputs equal to one; Appendix D.1 of
arXiv:2510.22026v2. -/
example : (1 : ℝ) ^ (1 - 1 / (max 4 2 : ℕ) : ℝ) ≤
    (2 / (1 : ℝ) ^ (1 / (max 4 2 : ℕ) : ℝ)) * 1 := by
  apply monomial_gradient_power_bound 4 1 1 1 1 1 <;> norm_num

end Transformer.Normalization
